#include "../common/Stopwatch.hpp"

#include <boost/functional/hash.hpp>
#include <boost/mp11.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <llama/llama.hpp>
#include <numeric>
#include <omp.h>
#include <string_view>

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
} // namespace tag

using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, float>,
        llama::Field<tag::Z, float>
    >>,
    llama::Field<tag::Vel, llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, float>,
        llama::Field<tag::Z, float>
    >>,
    llama::Field<tag::Mass, float>
>;
// clang-format on

namespace llamaex
{
    using namespace llama;

    template <std::size_t Dim, typename Func>
    void parallelForEachADCoord(ArrayDims<Dim> adSize, std::size_t numThreads, Func&& func)
    {
#pragma omp parallel for num_threads(numThreads)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(adSize[0]); i++)
        {
            if constexpr (Dim > 1)
                forEachADCoord(internal::popFront(adSize), std::forward<Func>(func), static_cast<std::size_t>(i));
            else
                std::forward<Func>(func)(ArrayDims<Dim>{static_cast<std::size_t>(i)});
        }
    }
} // namespace llamaex

template <typename Mapping1, typename BlobType1, typename Mapping2, typename BlobType2>
void naive_copy(
    const llama::View<Mapping1, BlobType1>& srcView,
    llama::View<Mapping2, BlobType2>& dstView,
    std::size_t numThreads = 1)
{
    static_assert(std::is_same_v<typename Mapping1::RecordDim, typename Mapping2::RecordDim>);

    if (srcView.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    llamaex::parallelForEachADCoord(
        srcView.mapping.arrayDims(),
        numThreads,
        [&](auto ad)
        {
            llama::forEachLeaf<typename Mapping1::RecordDim>(
                [&](auto coord)
                {
                    dstView(ad)(coord) = srcView(ad)(coord);
                    // std::memcpy(
                    //    &dstView(ad)(coord),
                    //    &srcView(ad)(coord),
                    //    sizeof(llama::GetType<typename Mapping1::RecordDim, decltype(coord)>));
                });
        });
}

template <typename Mapping1, typename BlobType1, typename Mapping2, typename BlobType2>
void std_copy(
    const llama::View<Mapping1, BlobType1>& srcView,
    llama::View<Mapping2, BlobType2>& dstView,
    std::size_t numThreads = 1)
{
    static_assert(std::is_same_v<typename Mapping1::RecordDim, typename Mapping2::RecordDim>);

    if (srcView.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    std::copy(srcView.begin(), srcView.end(), dstView.begin());
}

void parallel_memcpy(std::byte* dst, const std::byte* src, std::size_t size, std::size_t numThreads = 1)
{
    const auto sizePerThread = size / numThreads;
    const auto sizeLastThread = sizePerThread + size % numThreads;

#pragma omp parallel num_threads(numThreads)
    {
        const auto id = static_cast<std::size_t>(omp_get_thread_num());
        const auto sizeThisThread = id == numThreads - 1 ? sizeLastThread : sizePerThread;
        std::memcpy(dst + id * sizePerThread, src + id * sizePerThread, sizeThisThread);
    }
}

template <
    bool ReadOpt,
    typename ArrayDims,
    typename RecordDim,
    std::size_t LanesSrc,
    typename BlobType1,
    std::size_t LanesDst,
    typename BlobType2>
void aosoa_copy(
    const llama::View<
        llama::mapping::AoSoA<ArrayDims, RecordDim, LanesSrc, llama::mapping::LinearizeArrayDimsCpp>,
        BlobType1>& srcView,
    llama::View<
        llama::mapping::AoSoA<ArrayDims, RecordDim, LanesDst, llama::mapping::LinearizeArrayDimsCpp>,
        BlobType2>& dstView,
    std::size_t numThreads = 1)
{
    static_assert(decltype(srcView.storageBlobs)::rank == 1);
    static_assert(decltype(dstView.storageBlobs)::rank == 1);

    if (srcView.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    const auto arrayDims = dstView.mapping.arrayDims();
    const auto flatSize = std::reduce(std::begin(arrayDims), std::end(arrayDims), std::size_t{1}, std::multiplies<>{});

    const std::byte* src = &srcView.storageBlobs[0][0];
    std::byte* dst = &dstView.storageBlobs[0][0];

    // the same as AoSoA::blobNrAndOffset but takes a flat array index
    auto map = [](std::size_t flatArrayIndex, auto coord, std::size_t Lanes)
    {
        const auto blockIndex = flatArrayIndex / Lanes;
        const auto laneIndex = flatArrayIndex % Lanes;
        const auto offset = (llama::sizeOf<RecordDim> * Lanes) * blockIndex
            + llama::offsetOf<RecordDim, decltype(coord)> * Lanes
            + sizeof(llama::GetType<RecordDim, decltype(coord)>) * laneIndex;
        return offset;
    };

    if constexpr (ReadOpt)
    {
        // optimized for linear reading
        const auto elementsPerThread = ((flatSize / LanesSrc) / numThreads) * LanesSrc;
#pragma omp parallel num_threads(numThreads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto start = id * elementsPerThread;
            const auto stop = id == numThreads - 1 ? flatSize : (id + 1) * elementsPerThread;
            auto* threadSrc = src + map(start, llama::RecordCoord<>{}, LanesSrc);

            for (std::size_t i = start; i < stop; i += LanesSrc)
            {
                llama::forEachLeaf<RecordDim>(
                    [&](auto coord)
                    {
                        constexpr auto L = std::min(LanesSrc, LanesDst);
                        static_assert(LanesSrc % L == 0);
                        static_assert(LanesDst % L == 0);
                        for (std::size_t j = 0; j < LanesSrc; j += L)
                        {
                            constexpr auto bytes = L * sizeof(llama::GetType<RecordDim, decltype(coord)>);
                            std::memcpy(&dst[map(i + j, coord, LanesDst)], threadSrc, bytes);
                            threadSrc += bytes;
                        }
                    });
            }
        }
    }
    else
    {
        // optimized for linear writing
        const auto elementsPerThread = ((flatSize / LanesDst) / numThreads) * LanesDst;
#pragma omp parallel num_threads(numThreads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto start = id * elementsPerThread;
            const auto stop = id == numThreads - 1 ? flatSize : (id + 1) * elementsPerThread;

            auto* threadDst = dst + map(start, llama::RecordCoord<>{}, LanesDst);

            for (std::size_t i = start; i < stop; i += LanesDst)
            {
                llama::forEachLeaf<RecordDim>(
                    [&](auto coord)
                    {
                        constexpr auto L = std::min(LanesSrc, LanesDst);
                        for (std::size_t j = 0; j < LanesDst; j += L)
                        {
                            constexpr auto bytes = L * sizeof(llama::GetType<RecordDim, decltype(coord)>);
                            std::memcpy(threadDst, &src[map(i + j, coord, LanesSrc)], bytes);
                            threadDst += bytes;
                        }
                    });
            }
        }
    }
}

template <typename Mapping, typename BlobType>
auto hash(const llama::View<Mapping, BlobType>& view)
{
    std::size_t acc = 0;
    for (auto ad : llama::ArrayDimsIndexRange{view.mapping.arrayDims()})
        llama::forEachLeaf<Particle>([&](auto coord) { boost::hash_combine(acc, view(ad)(coord)); });
    return acc;
}
template <typename Mapping>
auto prepareViewAndHash(Mapping mapping)
{
    auto view = llama::allocView(mapping);

    auto value = 0.0f;
    for (auto ad : llama::ArrayDimsIndexRange{mapping.arrayDims()})
    {
        auto p = view(ad);
        p(tag::Pos{}, tag::X{}) = value++;
        p(tag::Pos{}, tag::Y{}) = value++;
        p(tag::Pos{}, tag::Z{}) = value++;
        p(tag::Vel{}, tag::X{}) = value++;
        p(tag::Vel{}, tag::Y{}) = value++;
        p(tag::Vel{}, tag::Z{}) = value++;
        p(tag::Mass{}) = value++;
    }

    const auto checkSum = hash(view);
    return std::tuple{view, checkSum};
}

template <typename Mapping>
inline constexpr auto is_AoSoA = false;

template <typename AD, typename RD, std::size_t L>
inline constexpr auto is_AoSoA<llama::mapping::AoSoA<AD, RD, L>> = true;

auto main() -> int
try
{
    const auto numThreads = static_cast<std::size_t>(omp_get_num_threads());
    std::cout << "Threads: " << numThreads << "\n";

    const auto arrayDims = llama::ArrayDims{1024, 1024, 16};

    std::ofstream plotFile{"viewcopy.tsv"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << "\"\"\t\"naive copy\"\t\"naive copy(p)\"\t\"std::copy\"\t\"memcpy\"\t\"memcpy(p)\"\t\"aosoa "
                "copy(r)\"\t\"aosoa copy(w)"
                "\"\t\"aosoa copy(r,p)\"\t\"aosoa copy(w,p)\"\n";

    auto benchmarkAllCopies = [&](std::string_view srcName, std::string_view dstName, auto srcMapping, auto dstMapping)
    {
        std::cout << srcName << " -> " << dstName << "\n";
        plotFile << "\"" << srcName << " -> " << dstName << "\"\t";

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);

        auto benchmarkCopy = [&, srcView = srcView, srcHash = srcHash](std::string_view name, auto copy)
        {
            auto dstView = llama::allocView(dstMapping);
            Stopwatch watch;
            copy(srcView, dstView);
            const auto seconds = watch.printAndReset(name, '\t');
            const auto dstHash = hash(dstView);
            std::cout << (srcHash == dstHash ? "" : "\thash BAD ") << "\n";
            plotFile << seconds << "\t";
        };

        benchmarkCopy("naive copy", [](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView); });
        benchmarkCopy(
            "naive copy(p)",
            [&](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView, numThreads); });
        benchmarkCopy("std::copy", [](const auto& srcView, auto& dstView) { std_copy(srcView, dstView); });
        benchmarkCopy(
            "memcpy",
            [](const auto& srcView, auto& dstView)
            {
                static_assert(decltype(srcView.storageBlobs)::rank == 1);
                static_assert(decltype(dstView.storageBlobs)::rank == 1);
                std::memcpy(
                    dstView.storageBlobs[0].data(),
                    srcView.storageBlobs[0].data(),
                    dstView.storageBlobs[0].size());
            });
        benchmarkCopy(
            "memcpy(p)",
            [&](const auto& srcView, auto& dstView)
            {
                static_assert(decltype(srcView.storageBlobs)::rank == 1);
                static_assert(decltype(dstView.storageBlobs)::rank == 1);
                parallel_memcpy(
                    dstView.storageBlobs[0].data(),
                    srcView.storageBlobs[0].data(),
                    dstView.storageBlobs[0].size(),
                    numThreads);
            });
        if constexpr (is_AoSoA<decltype(srcMapping)> && is_AoSoA<decltype(dstMapping)>)
        {
            benchmarkCopy(
                "aosoa copy(r)",
                [](const auto& srcView, auto& dstView) { aosoa_copy<true>(srcView, dstView); });
            benchmarkCopy(
                "aosoa copy(w)",
                [](const auto& srcView, auto& dstView) { aosoa_copy<false>(srcView, dstView); });
            benchmarkCopy(
                "aosoa_copy(r,p)",
                [&](const auto& srcView, auto& dstView) { aosoa_copy<true>(srcView, dstView, numThreads); });
            benchmarkCopy(
                "aosoa_copy(w,p)",
                [&](const auto& srcView, auto& dstView) { aosoa_copy<false>(srcView, dstView, numThreads); });
        }
        else
        {
            plotFile << "0\t";
            plotFile << "0\t";
            plotFile << "0\t";
            plotFile << "0\t";
        }
        plotFile << "\n";
    };

    const auto aosMapping = llama::mapping::AoS{arrayDims, Particle{}};
    const auto soaMapping = llama::mapping::SoA{arrayDims, Particle{}};
    const auto aosoa8Mapping = llama::mapping::AoSoA<decltype(arrayDims), Particle, 8>{arrayDims};
    const auto aosoa32Mapping = llama::mapping::AoSoA<decltype(arrayDims), Particle, 32>{arrayDims};
    const auto aosoa64Mapping = llama::mapping::AoSoA<decltype(arrayDims), Particle, 64>{arrayDims};

    benchmarkAllCopies("AoS", "SoA", aosMapping, soaMapping);
    benchmarkAllCopies("SoA", "AoS", soaMapping, aosMapping);
    benchmarkAllCopies("AoSoA8", "AoSoA32", aosoa8Mapping, aosoa32Mapping);
    benchmarkAllCopies("AoSoA8", "AoSoA64", aosoa8Mapping, aosoa64Mapping);
    benchmarkAllCopies("AoSoA32", "AoSoA8", aosoa32Mapping, aosoa8Mapping);
    benchmarkAllCopies("AoSoA64", "AoSoA8", aosoa64Mapping, aosoa8Mapping);

    std::cout << "Plot with: ./viewcopy.sh\n";
    std::ofstream{"viewcopy.sh"} << R"(#!/usr/bin/gnuplot -p
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
plot 'viewcopy.tsv' using 2:xtic(1) ti col, "" using 3 ti col, "" using 4 ti col, "" using 5 ti col, "" using 6 ti col, "" using 7 ti col, "" using 8 ti col, "" using 9 ti col, "" using 10 ti col
)";
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

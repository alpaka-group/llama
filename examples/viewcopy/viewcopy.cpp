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
    void parallelForEachADCoord(ArrayDomain<Dim> adSize, std::size_t numThreads, Func&& func)
    {
#pragma omp parallel for num_threads(numThreads)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(adSize[0]); i++)
        {
            if constexpr (Dim > 1)
                forEachADCoord(internal::popFront(adSize), std::forward<Func>(func), static_cast<std::size_t>(i));
            else
                std::forward<Func>(func)(ArrayDomain<Dim>{static_cast<std::size_t>(i)});
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

    if (srcView.mapping.arrayDomainSize != dstView.mapping.arrayDomainSize)
        throw std::runtime_error{"UserDomain sizes are different"};

    llamaex::parallelForEachADCoord(
        srcView.mapping.arrayDomainSize,
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
    typename ArrayDomain,
    typename RecordDim,
    std::size_t LanesSrc,
    typename BlobType1,
    std::size_t LanesDst,
    typename BlobType2>
void aosoa_copy(
    const llama::View<
        llama::mapping::AoSoA<ArrayDomain, RecordDim, LanesSrc, llama::mapping::LinearizeArrayDomainCpp>,
        BlobType1>& srcView,
    llama::View<
        llama::mapping::AoSoA<ArrayDomain, RecordDim, LanesDst, llama::mapping::LinearizeArrayDomainCpp>,
        BlobType2>& dstView,
    std::size_t numThreads = 1)
{
    static_assert(decltype(srcView.storageBlobs)::rank == 1);
    static_assert(decltype(dstView.storageBlobs)::rank == 1);

    if (srcView.mapping.arrayDomainSize != dstView.mapping.arrayDomainSize)
        throw std::runtime_error{"UserDomain sizes are different"};

    const auto flatSize = std::reduce(
        std::begin(dstView.mapping.arrayDomainSize),
        std::end(dstView.mapping.arrayDomainSize),
        std::size_t{1},
        std::multiplies<>{});

    const std::byte* src = srcView.storageBlobs[0].data();
    std::byte* dst = dstView.storageBlobs[0].data();

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
    for (auto ad : llama::ArrayDomainIndexRange{view.mapping.arrayDomainSize})
        llama::forEachLeaf<Particle>([&](auto coord) { boost::hash_combine(acc, view(ad)(coord)); });
    return acc;
}
template <typename Mapping>
auto prepareViewAndHash(Mapping mapping)
{
    auto view = llama::allocView(mapping);

    auto value = 0.0f;
    for (auto ad : llama::ArrayDomainIndexRange{mapping.arrayDomainSize})
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

template <typename SrcView, typename DstMapping, typename F>
void benchmarkCopy(
    std::string_view name,
    std::ostream& plotFile,
    const SrcView& srcView,
    std::size_t srcHash,
    DstMapping dstMapping,
    F copy)
{
    auto dstView = llama::allocView(dstMapping);
    Stopwatch watch;
    copy(srcView, dstView);
    const auto seconds = watch.printAndReset(name, '\t');
    const auto dstHash = hash(dstView);
    std::cout << (srcHash == dstHash ? "" : "\thash BAD ") << "\n";
    plotFile << seconds << "\t";
}

auto main() -> int
try
{
    const auto numThreads = static_cast<std::size_t>(omp_get_num_threads());
    std::cout << "Threads: " << numThreads << "\n";

    const auto userDomain = llama::ArrayDomain{1024, 1024, 16};

    std::ofstream plotFile{"viewcopy.tsv"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << "\"\"\t\"naive copy\"\t\"naive copy(p)\"\t\"memcpy\"\t\"memcpy(p)\"\t\"aosoa copy(r)\"\t\"aosoa copy(w)"
                "\"\t\"aosoa copy(r,p)\"\t\"aosoa copy(w,p)\"\n";

    {
        std::cout << "AoS -> SoA\n";
        plotFile << "\"AoS -> SoA\"\t";
        const auto srcMapping = llama::mapping::AoS{userDomain, Particle{}};
        const auto dstMapping = llama::mapping::SoA{userDomain, Particle{}};

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);
        benchmarkCopy(
            "naive copy",
            plotFile,
            srcView,
            srcHash,
            dstMapping,
            [](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView); });
        benchmarkCopy(
            "naive copy(p)",
            plotFile,
            srcView,
            srcHash,
            dstMapping,
            [&](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView, numThreads); });
        benchmarkCopy(
            "memcpy",
            plotFile,
            srcView,
            srcHash,
            dstMapping,
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
            plotFile,
            srcView,
            srcHash,
            dstMapping,
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
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "\n";
    }

    {
        std::cout << "SoA -> AoS\n";
        plotFile << "\"SoA -> AoS\"\t";
        const auto srcMapping = llama::mapping::SoA{userDomain, Particle{}};
        const auto dstMapping = llama::mapping::AoS{userDomain, Particle{}};

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);
        benchmarkCopy(
            "naive copy",
            plotFile,
            srcView,
            srcHash,
            dstMapping,
            [](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView); });
        benchmarkCopy(
            "naive copy(p)",
            plotFile,
            srcView,
            srcHash,
            dstMapping,
            [&](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView, numThreads); });
        benchmarkCopy(
            "memcpy",
            plotFile,
            srcView,
            srcHash,
            dstMapping,
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
            plotFile,
            srcView,
            srcHash,
            dstMapping,
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
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "\n";
    }

    using namespace boost::mp11;
    mp_for_each<mp_list<
        mp_list_c<std::size_t, 8, 32>,
        mp_list_c<std::size_t, 8, 64>,
        mp_list_c<std::size_t, 32, 8>,
        mp_list_c<std::size_t, 64, 8>>>(
        [&](auto pair)
        {
            constexpr auto LanesSrc = mp_first<decltype(pair)>::value;
            constexpr auto LanesDst = mp_second<decltype(pair)>::value;

            std::cout << "AoSoA" << LanesSrc << " -> AoSoA" << LanesDst << "\n";
            plotFile << "\"AoSoA" << LanesSrc << " -> AoSoA" << LanesDst << "\"\t";
            const auto srcMapping = llama::mapping::AoSoA<decltype(userDomain), Particle, LanesSrc>{userDomain};
            const auto dstMapping = llama::mapping::AoSoA<decltype(userDomain), Particle, LanesDst>{userDomain};

            auto [srcView, srcHash] = prepareViewAndHash(srcMapping);
            benchmarkCopy(
                "naive copy",
                plotFile,
                srcView,
                srcHash,
                dstMapping,
                [](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView); });
            benchmarkCopy(
                "naive copy(p)",
                plotFile,
                srcView,
                srcHash,
                dstMapping,
                [&](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView, numThreads); });
            benchmarkCopy(
                "memcpy",
                plotFile,
                srcView,
                srcHash,
                dstMapping,
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
                plotFile,
                srcView,
                srcHash,
                dstMapping,
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
            benchmarkCopy(
                "aosoa copy(r)",
                plotFile,
                srcView,
                srcHash,
                dstMapping,
                [](const auto& srcView, auto& dstView) { aosoa_copy<true>(srcView, dstView); });
            benchmarkCopy(
                "aosoa copy(w)",
                plotFile,
                srcView,
                srcHash,
                dstMapping,
                [](const auto& srcView, auto& dstView) { aosoa_copy<false>(srcView, dstView); });
            benchmarkCopy(
                "aosoa_copy(r,p)",
                plotFile,
                srcView,
                srcHash,
                dstMapping,
                [&](const auto& srcView, auto& dstView) { aosoa_copy<true>(srcView, dstView, numThreads); });
            benchmarkCopy(
                "aosoa_copy(w,p)",
                plotFile,
                srcView,
                srcHash,
                dstMapping,
                [&](const auto& srcView, auto& dstView) { aosoa_copy<false>(srcView, dstView, numThreads); });
            plotFile << "\n";
        });

    std::cout << "Plot with: ./viewcopy.sh\n";
    std::ofstream{"viewcopy.sh"} << R"(#!/usr/bin/gnuplot -p
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
plot 'viewcopy.tsv' using 2:xtic(1) ti col, "" using 3 ti col, "" using 4 ti col, "" using 5 ti col, "" using 6 ti col, "" using 7 ti col, "" using 8 ti col, "" using 9 ti col
)";
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"
#include "../common/ttjet_13tev_june2019.hpp"

#include <boost/functional/hash.hpp>
#include <boost/mp11.hpp>
#include <fmt/format.h>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <llama/llama.hpp>
#include <numeric>
#include <omp.h>
#include <string_view>

constexpr auto REPETITIONS = 5;
constexpr auto arrayDims = llama::ArrayDims{512, 512, 16};

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

// using RecordDim = Particle;
using RecordDim = boost::mp11::mp_take_c<Event, 20>;
// using RecordDim = Event; // WARN: expect long compilation time

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

template <typename SrcMapping, typename SrcBlobType, typename DstMapping, typename DstBlobType>
void naive_copy(
    const llama::View<SrcMapping, SrcBlobType>& srcView,
    llama::View<DstMapping, DstBlobType>& dstView,
    std::size_t numThreads = 1)
{
    static_assert(std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>);

    if (srcView.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    llamaex::parallelForEachADCoord(
        srcView.mapping.arrayDims(),
        numThreads,
        [&](auto ad) LLAMA_LAMBDA_INLINE
        {
            llama::forEachLeaf<typename DstMapping::RecordDim>([&](auto coord) LLAMA_LAMBDA_INLINE
                                                               { dstView(ad)(coord) = srcView(ad)(coord); });
        });
}

template <typename SrcMapping, typename SrcBlobType, typename DstMapping, typename DstBlobType>
void std_copy(
    const llama::View<SrcMapping, SrcBlobType>& srcView,
    llama::View<DstMapping, DstBlobType>& dstView,
    std::size_t numThreads = 1)
{
    static_assert(std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>);

    if (srcView.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    std::copy(srcView.begin(), srcView.end(), dstView.begin());
}

// adapted from: https://stackoverflow.com/a/30386256/1034717
void* memcpy_avx2(void* dst, const void* src, size_t n) noexcept
{
#define ALIGN(ptr, align) (((ptr) + (align) -1) & ~((align) -1))

    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);

    // fall back to memcpy() if dst and src are misaligned
    if ((reinterpret_cast<uintptr_t>(d) & 31) != (reinterpret_cast<uintptr_t>(s) & 31))
        return memcpy(d, s, n);

    // align dst/src address multiple of 32
    if (reinterpret_cast<uintptr_t>(d) & 31)
    {
        uintptr_t header_bytes = 32 - (reinterpret_cast<uintptr_t>(d) & 31);
        assert(header_bytes < 32);

        memcpy(d, s, std::min(header_bytes, n));

        d = reinterpret_cast<char*>(ALIGN(reinterpret_cast<uintptr_t>(d), 32));
        s = reinterpret_cast<char*>(ALIGN(reinterpret_cast<uintptr_t>(s), 32));
        n -= std::min(header_bytes, n);
    }

    constexpr auto unrollFactor = 8;
    constexpr auto bytesPerIteration = 32 * unrollFactor;
    while (n >= bytesPerIteration)
    {
#pragma unroll
#pragma GCC unroll unrollFactor
        for (auto i = 0; i < unrollFactor; i++)
            _mm256_stream_si256(
                reinterpret_cast<__m256i*>(d) + i,
                _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(s) + i));
        s += bytesPerIteration;
        d += bytesPerIteration;
        n -= bytesPerIteration;
    }

    if (n > 0)
        memcpy(d, s, n);

    return dst;
#undef ALIGN
}

inline void parallel_memcpy(
    std::byte* dst,
    const std::byte* src,
    std::size_t size,
    decltype(std::memcpy) = std::memcpy,
    std::size_t numThreads = 1)
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
    std::size_t LanesDst,
    bool MBSrc,
    bool MBDst,
    typename SrcView,
    typename DstView>
void aosoa_copy_internal(const SrcView& srcView, DstView& dstView, std::size_t numThreads)
{
    if (srcView.mapping.arrayDims() != dstView.mapping.arrayDims())
        throw std::runtime_error{"Array dimensions sizes are different"};

    constexpr auto srcIsAoSoA = LanesSrc != std::numeric_limits<std::size_t>::max();
    constexpr auto dstIsAoSoA = LanesDst != std::numeric_limits<std::size_t>::max();

    static_assert(!srcIsAoSoA || decltype(srcView.storageBlobs)::rank == 1);
    static_assert(!dstIsAoSoA || decltype(dstView.storageBlobs)::rank == 1);

    const auto arrayDims = dstView.mapping.arrayDims();
    const auto flatSize = std::reduce(std::begin(arrayDims), std::end(arrayDims), std::size_t{1}, std::multiplies<>{});

    // the same as AoSoA::blobNrAndOffset but takes a flat array index
    auto mapAoSoA = [](std::size_t flatArrayIndex, auto coord, std::size_t Lanes) LLAMA_LAMBDA_INLINE
    {
        const auto blockIndex = flatArrayIndex / Lanes;
        const auto laneIndex = flatArrayIndex % Lanes;
        const auto offset = (llama::sizeOf<RecordDim> * Lanes) * blockIndex
            + llama::offsetOf<RecordDim, decltype(coord)> * Lanes
            + sizeof(llama::GetType<RecordDim, decltype(coord)>) * laneIndex;
        return offset;
    };
    // the same as SoA::blobNrAndOffset but takes a flat array index
    auto mapSoA = [&](std::size_t flatArrayIndex, auto coord, bool mb) LLAMA_LAMBDA_INLINE
    {
        const auto blob = mb * llama::flatRecordCoord<RecordDim, decltype(coord)>;
        const auto offset = !mb * llama::offsetOf<RecordDim, decltype(coord)> * flatSize
            + sizeof(llama::GetType<RecordDim, decltype(coord)>) * flatArrayIndex;
        return llama::NrAndOffset{blob, offset};
    };

    auto mapSrc = [&srcView, &mapAoSoA, &mapSoA](std::size_t flatArrayIndex, auto coord) LLAMA_LAMBDA_INLINE
    {
        if constexpr (srcIsAoSoA)
            return &srcView.storageBlobs[0][0] + mapAoSoA(flatArrayIndex, coord, LanesSrc);
        else
        {
            const auto [blob, off] = mapSoA(flatArrayIndex, coord, MBSrc);
            return &srcView.storageBlobs[blob][off];
        }
    };
    auto mapDst = [&dstView, &mapAoSoA, &mapSoA](std::size_t flatArrayIndex, auto coord) LLAMA_LAMBDA_INLINE
    {
        if constexpr (dstIsAoSoA)
            return &dstView.storageBlobs[0][0] + mapAoSoA(flatArrayIndex, coord, LanesDst);
        else
        {
            const auto [blob, off] = mapSoA(flatArrayIndex, coord, MBDst);
            return &dstView.storageBlobs[blob][off];
        }
    };

    constexpr auto L = std::min(LanesSrc, LanesDst);
    static_assert(!srcIsAoSoA || LanesSrc % L == 0);
    static_assert(!dstIsAoSoA || LanesDst % L == 0);
    if constexpr (ReadOpt)
    {
        // optimized for linear reading
        const auto elementsPerThread
            = srcIsAoSoA ? flatSize / LanesSrc / numThreads * LanesSrc : flatSize / L / numThreads * L;
#pragma omp parallel num_threads(numThreads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto start = id * elementsPerThread;
            const auto stop = id == numThreads - 1 ? flatSize : (id + 1) * elementsPerThread;

            auto copyLBlock = [&](const std::byte*& threadSrc, std::size_t dstIndex, auto coord) LLAMA_LAMBDA_INLINE
            {
                constexpr auto bytes = L * sizeof(llama::GetType<RecordDim, decltype(coord)>);
                std::memcpy(mapDst(dstIndex, coord), threadSrc, bytes);
                threadSrc += bytes;
            };
            if constexpr (srcIsAoSoA)
            {
                auto* threadSrc = mapSrc(start, llama::RecordCoord<>{});
                for (std::size_t i = start; i < stop; i += LanesSrc)
                    llama::forEachLeaf<RecordDim>(
                        [&](auto coord) LLAMA_LAMBDA_INLINE
                        {
                            for (std::size_t j = 0; j < LanesSrc; j += L)
                                copyLBlock(threadSrc, i + j, coord);
                        });
            }
            else
            {
                llama::forEachLeaf<RecordDim>(
                    [&](auto coord) LLAMA_LAMBDA_INLINE
                    {
                        auto* threadSrc = mapSrc(start, coord);
                        for (std::size_t i = start; i < stop; i += L)
                            copyLBlock(threadSrc, i, coord);
                    });
            }
        }
    }
    else
    {
        // optimized for linear writing
        const auto elementsPerThread
            = dstIsAoSoA ? ((flatSize / LanesDst) / numThreads) * LanesDst : flatSize / L / numThreads * L;
#pragma omp parallel num_threads(numThreads)
        {
            const auto id = static_cast<std::size_t>(omp_get_thread_num());
            const auto start = id * elementsPerThread;
            const auto stop = id == numThreads - 1 ? flatSize : (id + 1) * elementsPerThread;

            auto copyLBlock = [&](std::byte*& threadDst, std::size_t srcIndex, auto coord) LLAMA_LAMBDA_INLINE
            {
                constexpr auto bytes = L * sizeof(llama::GetType<RecordDim, decltype(coord)>);
                std::memcpy(threadDst, mapSrc(srcIndex, coord), bytes);
                threadDst += bytes;
            };
            if constexpr (dstIsAoSoA)
            {
                auto* threadDst = mapDst(start, llama::RecordCoord<>{});
                for (std::size_t i = start; i < stop; i += LanesDst)
                    llama::forEachLeaf<RecordDim>(
                        [&](auto coord) LLAMA_LAMBDA_INLINE
                        {
                            for (std::size_t j = 0; j < LanesDst; j += L)
                                copyLBlock(threadDst, i + j, coord);
                        });
            }
            else
            {
                llama::forEachLeaf<RecordDim>(
                    [&](auto coord) LLAMA_LAMBDA_INLINE
                    {
                        auto* threadDst = mapDst(start, coord);
                        for (std::size_t i = start; i < stop; i += L)
                            copyLBlock(threadDst, i, coord);
                    });
            }
        }
    }
}

template <
    bool ReadOpt,
    typename ArrayDims,
    typename RecordDim,
    std::size_t LanesSrc,
    typename SrcBlobType,
    std::size_t LanesDst,
    typename DstBlobType>
void aosoa_copy(
    const llama::View<
        llama::mapping::AoSoA<ArrayDims, RecordDim, LanesSrc, llama::mapping::LinearizeArrayDimsCpp>,
        SrcBlobType>& srcView,
    llama::View<
        llama::mapping::AoSoA<ArrayDims, RecordDim, LanesDst, llama::mapping::LinearizeArrayDimsCpp>,
        DstBlobType>& dstView,
    std::size_t numThreads = 1)
{
    aosoa_copy_internal<ReadOpt, ArrayDims, RecordDim, LanesSrc, LanesDst, false, false>(srcView, dstView, numThreads);
}

template <
    bool ReadOpt,
    typename ArrayDims,
    typename RecordDim,
    std::size_t LanesSrc,
    typename SrcBlobType,
    bool DstSeparateBuffers,
    typename DstBlobType>
void aosoa_copy(
    const llama::View<
        llama::mapping::AoSoA<ArrayDims, RecordDim, LanesSrc, llama::mapping::LinearizeArrayDimsCpp>,
        SrcBlobType>& srcView,
    llama::View<
        llama::mapping::SoA<ArrayDims, RecordDim, DstSeparateBuffers, llama::mapping::LinearizeArrayDimsCpp>,
        DstBlobType>& dstView,
    std::size_t numThreads = 1)
{
    aosoa_copy_internal<
        ReadOpt,
        ArrayDims,
        RecordDim,
        LanesSrc,
        std::numeric_limits<std::size_t>::max(),
        false,
        DstSeparateBuffers>(srcView, dstView, numThreads);
}

template <
    bool ReadOpt,
    typename ArrayDims,
    typename RecordDim,
    bool SrcSeparateBuffers,
    typename SrcBlobType,
    std::size_t LanesDst,
    typename DstBlobType>
void aosoa_copy(
    const llama::View<
        llama::mapping::SoA<ArrayDims, RecordDim, SrcSeparateBuffers, llama::mapping::LinearizeArrayDimsCpp>,
        SrcBlobType>& srcView,
    llama::View<
        llama::mapping::AoSoA<ArrayDims, RecordDim, LanesDst, llama::mapping::LinearizeArrayDimsCpp>,
        DstBlobType>& dstView,
    std::size_t numThreads = 1)
{
    aosoa_copy_internal<
        ReadOpt,
        ArrayDims,
        RecordDim,
        std::numeric_limits<std::size_t>::max(),
        LanesDst,
        SrcSeparateBuffers,
        false>(srcView, dstView, numThreads);
}


template <typename Mapping, typename BlobType>
auto hash(const llama::View<Mapping, BlobType>& view)
{
    std::size_t acc = 0;
    for (auto ad : llama::ArrayDimsIndexRange{view.mapping.arrayDims()})
        llama::forEachLeaf<typename Mapping::RecordDim>([&](auto coord) { boost::hash_combine(acc, view(ad)(coord)); });
    return acc;
}
template <typename Mapping>
auto prepareViewAndHash(Mapping mapping)
{
    auto view = llama::allocView(mapping);

    auto value = std::size_t{0};
    for (auto ad : llama::ArrayDimsIndexRange{mapping.arrayDims()})
        llama::forEachLeaf<typename Mapping::RecordDim>([&](auto coord) { view(ad)(coord) = value++; });

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
    const auto dataSize
        = std::reduce(arrayDims.begin(), arrayDims.end(), std::size_t{1}, std::multiplies{}) * llama::sizeOf<RecordDim>;
    const auto numThreads = static_cast<std::size_t>(omp_get_max_threads());
    std::cout << "Data size: " << dataSize / 1024 / 1024 << "MiB\n";
    std::cout << "Threads: " << numThreads << "\n";

    std::ofstream plotFile{"viewcopy.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "viewcopy CPU {}MiB particles on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 4
set ylabel "throughput [GiB/s]"
$data << EOD
)",
        dataSize / 1024 / 1024,
        common::hostname());

    plotFile << "\"\"\t\"memcpy\"\t\"memcpy\\\\\\_avx2\"\t\"memcpy(p)\"\t\"memcpy\\\\\\_avx2(p)\"\t\"naive "
                "copy\"\t\"std::copy\"\t\"aosoa copy(r)\"\t\"aosoa copy(w)\"\t\"naive copy(p)\"\t\"aosoa "
                "copy(r,p)\"\t\"aosoa copy(w,p)\"\n";

    std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> src(dataSize);

    auto benchmarkMemcpy = [&](std::string_view name, auto memcpy)
    {
        std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> dst(dataSize);
        Stopwatch watch;
        for (auto i = 0; i < REPETITIONS; i++)
            memcpy(dst.data(), src.data(), dataSize);
        const auto seconds = watch.printAndReset(name, '\t') / REPETITIONS;
        const auto gbs = (dataSize / seconds) / (1024.0 * 1024.0 * 1024.0);
        std::cout << gbs << "GiB/s\t\n";
        plotFile << gbs << "\t";
    };

    std::cout << "byte[] -> byte[]\n";
    plotFile << "\"byte[] -> byte[]\"\t";
    benchmarkMemcpy("memcpy", std::memcpy);
    benchmarkMemcpy("memcpy_avx2", memcpy_avx2);
    benchmarkMemcpy(
        "memcpy(p)",
        [&](auto* dst, auto* src, auto size) { parallel_memcpy(dst, src, size, std::memcpy, numThreads); });
    benchmarkMemcpy(
        "memcpy_avx2(p)",
        [&](auto* dst, auto* src, auto size) { parallel_memcpy(dst, src, size, memcpy_avx2, numThreads); });
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "0\t";
    plotFile << "\n";

    auto benchmarkAllCopies = [&](std::string_view srcName, std::string_view dstName, auto srcMapping, auto dstMapping)
    {
        std::cout << srcName << " -> " << dstName << "\n";
        plotFile << "\"" << srcName << " -> " << dstName << "\"\t";

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);

        auto benchmarkCopy = [&, srcView = srcView, srcHash = srcHash](std::string_view name, auto copy)
        {
            auto dstView = llama::allocView(dstMapping);
            Stopwatch watch;
            for (auto i = 0; i < REPETITIONS; i++)
                copy(srcView, dstView);
            const auto seconds = watch.printAndReset(name, '\t') / REPETITIONS;
            const auto gbs = (dataSize / seconds) / (1024.0 * 1024.0 * 1024.0);
            const auto dstHash = hash(dstView);
            std::cout << gbs << "GiB/s\t" << (srcHash == dstHash ? "" : "\thash BAD ") << "\n";
            plotFile << gbs << "\t";
        };

        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        plotFile << "0\t";
        benchmarkCopy("naive copy", [](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView); });
        benchmarkCopy("std::copy", [](const auto& srcView, auto& dstView) { std_copy(srcView, dstView); });
        constexpr auto oneIsAoSoA = is_AoSoA<decltype(srcMapping)> || is_AoSoA<decltype(dstMapping)>;
        if constexpr (oneIsAoSoA)
        {
            benchmarkCopy(
                "aosoa copy(r)",
                [](const auto& srcView, auto& dstView) { aosoa_copy<true>(srcView, dstView); });
            benchmarkCopy(
                "aosoa copy(w)",
                [](const auto& srcView, auto& dstView) { aosoa_copy<false>(srcView, dstView); });
        }
        else
        {
            plotFile << "0\t";
            plotFile << "0\t";
        }
        benchmarkCopy(
            "naive copy(p)",
            [&](const auto& srcView, auto& dstView) { naive_copy(srcView, dstView, numThreads); });
        if constexpr (oneIsAoSoA)
        {
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
        }
        plotFile << "\n";
    };

    const auto packedAoSMapping = llama::mapping::PackedAoS<decltype(arrayDims), RecordDim>{arrayDims};
    const auto alignedAoSMapping = llama::mapping::AlignedAoS<decltype(arrayDims), RecordDim>{arrayDims};
    const auto multiBlobSoAMapping = llama::mapping::MultiBlobSoA<decltype(arrayDims), RecordDim>{arrayDims};
    const auto aosoa8Mapping = llama::mapping::AoSoA<decltype(arrayDims), RecordDim, 8>{arrayDims};
    const auto aosoa32Mapping = llama::mapping::AoSoA<decltype(arrayDims), RecordDim, 32>{arrayDims};
    const auto aosoa64Mapping = llama::mapping::AoSoA<decltype(arrayDims), RecordDim, 64>{arrayDims};

    benchmarkAllCopies("P AoS", "A AoS", packedAoSMapping, alignedAoSMapping);
    benchmarkAllCopies("A AoS", "P AoS", alignedAoSMapping, packedAoSMapping);

    benchmarkAllCopies("A AoS", "SoA MB", alignedAoSMapping, multiBlobSoAMapping);
    benchmarkAllCopies("SoA MB", "A AoS", multiBlobSoAMapping, alignedAoSMapping);

    benchmarkAllCopies("SoA MB", "AoSoA32", multiBlobSoAMapping, aosoa32Mapping);
    benchmarkAllCopies("AoSoA32", "SoA MB", aosoa32Mapping, multiBlobSoAMapping);

    benchmarkAllCopies("AoSoA8", "AoSoA32", aosoa8Mapping, aosoa32Mapping);
    benchmarkAllCopies("AoSoA32", "AoSoA8", aosoa32Mapping, aosoa8Mapping);

    benchmarkAllCopies("AoSoA8", "AoSoA64", aosoa8Mapping, aosoa64Mapping);
    benchmarkAllCopies("AoSoA64", "AoSoA8", aosoa64Mapping, aosoa8Mapping);

    plotFile << R"(EOD
plot $data using 2:xtic(1) ti col, "" using 3 ti col, "" using 4 ti col, "" using 5 ti col, "" using 6 ti col, "" using 7 ti col, "" using 8 ti col, "" using 9 ti col, "" using 10 ti col, "" using 11 ti col, "" using 12 ti col
)";
    std::cout << "Plot with: ./viewcopy.sh\n";
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

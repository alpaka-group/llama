// Copyright 2020 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "../common/Stats.hpp"
#include "../common/Stopwatch.hpp"
#include "../common/env.hpp"
#include "../common/ttjet_13tev_june2019.hpp"

#include <boost/functional/hash.hpp>
#include <fmt/core.h>
#include <fstream>
#include <immintrin.h>
#include <llama/llama.hpp>
#include <numeric>
#include <omp.h>
#include <string_view>

constexpr auto repetitions = 20; // excluding 1 warmup run
constexpr auto extents = llama::ArrayExtents{512, 512, 16};
constexpr auto measureMemcpy = false;
constexpr auto runParallelVersions = true;

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

template<typename SrcMapping, typename SrcBlobType, typename DstMapping, typename DstBlobType>
void stdCopy(const llama::View<SrcMapping, SrcBlobType>& srcView, llama::View<DstMapping, DstBlobType>& dstView)
{
    static_assert(std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>);

    if(srcView.extents() != dstView.extents())
        throw std::runtime_error{"Array extents are different"};

    std::copy(srcView.begin(), srcView.end(), dstView.begin());
}

#ifdef __AVX2__
// adapted from: https://stackoverflow.com/a/30386256/1034717
auto memcpyAVX2(void* dst, const void* src, size_t n) noexcept -> void*
{
    auto* d = static_cast<std::byte*>(dst);
    const auto* s = static_cast<const std::byte*>(src);

    // fall back to memcpy() if dst and src are misaligned
    const auto lowerDstBits = reinterpret_cast<uintptr_t>(d) & 31u;
    if(lowerDstBits != (reinterpret_cast<uintptr_t>(s) & 31u))
        return memcpy(d, s, n);

    // align dst/src address multiple of 32
    if(lowerDstBits != 0u)
    {
        const auto headerBytes = std::min(static_cast<size_t>(32 - lowerDstBits), n);
        memcpy(d, s, headerBytes);
        d += headerBytes;
        s += headerBytes;
        n -= headerBytes;
    }

    constexpr auto unrollFactor = 8;
    constexpr auto bytesPerIteration = 32 * unrollFactor;
    while(n >= bytesPerIteration)
    {
        LLAMA_UNROLL(unrollFactor)
        for(auto i = 0; i < unrollFactor; i++)
            _mm256_stream_si256(
                reinterpret_cast<__m256i*>(d) + i,
                _mm256_stream_load_si256(reinterpret_cast<const __m256i*>(s) + i));
        s += bytesPerIteration;
        d += bytesPerIteration;
        n -= bytesPerIteration;
    }

    if(n > 0)
        memcpy(d, s, n);

    return dst;
}
#endif

template<typename Mapping, typename BlobType>
auto hash(const llama::View<Mapping, BlobType>& view)
{
    std::size_t acc = 0;
    for(auto ad : llama::ArrayIndexRange{view.extents()})
        llama::forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc) { boost::hash_combine(acc, view(ad)(rc)); });
    return acc;
}

template<typename Mapping>
auto prepareViewAndHash(Mapping mapping)
{
    auto view = llama::allocViewUninitialized(mapping);

    auto value = std::size_t{0};
    for(auto ad : llama::ArrayIndexRange{mapping.extents()})
        llama::forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc) { view(ad)(rc) = value++; });

    const auto checkSum = hash(view);
    return std::tuple{view, checkSum};
}

void benchmarkMemcopy(std::size_t dataSize, std::ostream& plotFile)
{
    std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> src(dataSize);

    auto run = [&](std::string_view name, auto memcpy)
    {
        std::vector<std::byte, llama::bloballoc::AlignedAllocator<std::byte, 64>> dst(dataSize);
        Stopwatch watch;
        common::Stats stats;
        for(auto i = 0; i < repetitions + 1; i++)
        {
            memcpy(dst.data(), src.data(), dataSize);
            const auto seconds = watch.getAndReset();
            const auto gbs = dataSize / (seconds * 1024.0 * 1024.0 * 1024.0);
            stats(gbs);
        }
        std::cout << name << " " << stats.mean() << "GiB/s\n";
        plotFile << "# " << name << ' ' << stats.mean() << "\t" << stats.sem() << '\n' << std::flush;
    };

    run("memcpy", std::memcpy);
#ifdef __AVX2__
    run("memcpy_avx2", memcpyAVX2);
#endif
    run("memcpy(p)",
        [&](auto* dst, auto* src, auto size)
        {
#pragma omp parallel
            llama::internal::parallelMemcpy(dst, src, size, omp_get_thread_num(), omp_get_num_threads(), std::memcpy);
        });
#ifdef __AVX2__
    run("memcpy_avx2(p)",
        [&](auto* dst, auto* src, auto size)
        {
#    pragma omp parallel
            llama::internal::parallelMemcpy(dst, src, size, omp_get_thread_num(), omp_get_num_threads(), memcpyAVX2);
        });
#endif
}

auto main() -> int
try
{
    const auto env = common::captureEnv();
    const auto dataSize = llama::product(extents) * llama::sizeOf<RecordDim>;
    fmt::print("Data size: {}MiB\n{}\n", dataSize / 1024 / 1024, env);

    std::ofstream plotFile{"viewcopy.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"plot(#!/usr/bin/gnuplot -p
# {}
set title "viewcopy CPU {}MiB particles"
set style data histograms
set style histogram errorbars
set style fill solid border -1
set xtics rotate by 45 right nomirror
set key out top center maxrows 4
set ylabel "Throughput [GiB/s]"

)plot",
        env,
        dataSize / 1024 / 1024);

    // measure memcpy to have a baseline for comparison
    if constexpr(measureMemcpy)
    {
        fmt::print("byte[] -> byte[]\n");
        benchmarkMemcopy(dataSize, plotFile);
    }

    // benchmark structural copies

    plotFile << R"plot(
$data << EOD
"src layout"	"dst layout"	"naive copy"	"naive copy_sem"	"std::copy"	"std::copy_sem"	"aosoa copy(r)"	"aosoa copy(r)_sem"	"aosoa copy(w)"	"aosoa copy(w)_sem"	"LLAMA copy"	"LLAMA copy_sem")plot";
    if constexpr(runParallelVersions)
        plotFile
            << R"plot(	"naive copy(p)"	"naive copy(p)_sem"	"aosoa copy(r,p)"	"aosoa copy(r,p)_sem"	"aosoa copy(w,p)"	"aosoa copy(w,p)_sem"	"LLAMA copy(p)"	"LLAMA copy_sem(p)")plot";
    plotFile << '\n';

    auto benchmarkAllCopies = [&](std::string_view srcName, std::string_view dstName, auto srcMapping, auto dstMapping)
    {
        fmt::print("{} -> {}\n", srcName, dstName);
        plotFile << "\"" << srcName << "\" \"" << dstName << "\"\t";

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);

        auto benchmarkCopy = [&, srcView = srcView, srcHash = srcHash](std::string_view name, auto copy)
        {
            auto dstView = llama::allocViewUninitialized(dstMapping);
            Stopwatch watch;
            common::Stats stats;
            for(auto i = 0; i < repetitions + 1; i++)
            {
                copy(srcView, dstView);
                const auto seconds = watch.getAndReset();
                const auto gbs = (dataSize / seconds) / (1024.0 * 1024.0 * 1024.0);
                stats(gbs);
            }
            const auto dstHash = hash(dstView);
            fmt::print("{} {}GiB/s\t{}\n", name, stats.mean(), srcHash == dstHash ? "" : "\thash BAD ");
            plotFile << stats.mean() << "\t" << stats.sem() << '\t';
        };

        benchmarkCopy(
            "naive copy",
            [](const auto& srcView, auto& dstView) { llama::fieldWiseCopy(srcView, dstView); });
        benchmarkCopy("std::copy", [](const auto& srcView, auto& dstView) { stdCopy(srcView, dstView); });
        using namespace llama::mapping;
        constexpr auto hasCommonBlockCopy = (isAoSoA<decltype(srcMapping)> && isAoSoA<decltype(dstMapping)>)
            || (isAoSoA<decltype(srcMapping)> && isSoA<decltype(dstMapping)>)
            || (isSoA<decltype(srcMapping)> && isAoSoA<decltype(dstMapping)>);
        if constexpr(hasCommonBlockCopy)
        {
            benchmarkCopy(
                "aosoa copy(r)",
                [](const auto& srcView, auto& dstView) { llama::aosoaCommonBlockCopy(srcView, dstView, true); });
            benchmarkCopy(
                "aosoa copy(w)",
                [](const auto& srcView, auto& dstView) { llama::aosoaCommonBlockCopy(srcView, dstView, false); });
        }
        else
        {
            for(int i = 0; i < 2 * 2; i++)
                plotFile << "0\t";
        }
        benchmarkCopy("llama", [&](const auto& srcView, auto& dstView) { llama::copy(srcView, dstView); });

        if constexpr(runParallelVersions)
        {
            benchmarkCopy(
                "naive copy(p)",
                [&](const auto& srcView, auto& dstView)
                {
#pragma omp parallel
                    // NOLINTNEXTLINE(openmp-exception-escape)
                    llama::fieldWiseCopy(srcView, dstView, omp_get_thread_num(), omp_get_num_threads());
                });
            if constexpr(hasCommonBlockCopy)
            {
                benchmarkCopy(
                    "aosoa_copy(r,p)",
                    [&](const auto& srcView, auto& dstView)
                    {
#pragma omp parallel
                        // NOLINTNEXTLINE(openmp-exception-escape)
                        llama::aosoaCommonBlockCopy(
                            srcView,
                            dstView,
                            true,
                            omp_get_thread_num(),
                            omp_get_num_threads());
                    });
                benchmarkCopy(
                    "aosoa_copy(w,p)",
                    [&](const auto& srcView, auto& dstView)
                    {
#pragma omp parallel
                        // NOLINTNEXTLINE(openmp-exception-escape)
                        llama::aosoaCommonBlockCopy(
                            srcView,
                            dstView,
                            false,
                            omp_get_thread_num(),
                            omp_get_num_threads());
                    });
            }
            else
            {
                for(int i = 0; i < 2 * 2; i++)
                    plotFile << "0\t";
            }
            benchmarkCopy(
                "llama(p)",
                [&](const auto& srcView, auto& dstView)
                {
#pragma omp parallel
                    // NOLINTNEXTLINE(openmp-exception-escape)
                    llama::copy(srcView, dstView, omp_get_thread_num(), omp_get_num_threads());
                });
        }
        plotFile << "\n";
    };

    using namespace boost::mp11;
    using ArrayExtents = std::remove_const_t<decltype(extents)>;
    using Mappings = mp_list<
        // llama::mapping::PackedAoS<ArrayExtents, RecordDim>,
        llama::mapping::AlignedAoS<ArrayExtents, RecordDim>,
        llama::mapping::AlignedSingleBlobSoA<ArrayExtents, RecordDim>,
        llama::mapping::MultiBlobSoA<ArrayExtents, RecordDim>,
        llama::mapping::AoSoA<ArrayExtents, RecordDim, 8>,
        llama::mapping::AoSoA<ArrayExtents, RecordDim, 32>>;
    std::string_view mappingNames[] = {/*"AoS P",*/ "AoS A", "SoA SB", "SoA MB", "AoSoA8", "AoSoA32"};
    mp_for_each<mp_iota<mp_size<Mappings>>>(
        [&]<typename I>(I)
        {
            mp_for_each<mp_iota<mp_size<Mappings>>>(
                [&]<typename J>(J)
                {
                    benchmarkAllCopies(
                        mappingNames[I::value],
                        mappingNames[J::value],
                        mp_at<Mappings, I>{extents},
                        mp_at<Mappings, J>{extents});
                });
        });

    plotFile << R"(EOD
plot $data using  3: 4:xtic(sprintf("%s -> %s", stringcolumn(1), stringcolumn(2))) ti col, \
        "" using  5: 6         ti col, \
        "" using  7: 8         ti col, \
        "" using  9:10         ti col, \
        "" using 11:12         ti col)";
    if constexpr(runParallelVersions)
        plotFile << R"(, \
        "" using 13:14         ti col, \
        "" using 15:16         ti col, \
        "" using 17:18         ti col, \
        "" using 19:20         ti col)";
    plotFile << '\n';
    fmt::print("Plot with: ./viewcopy.sh\n");
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

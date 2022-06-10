#include "../../common/Stopwatch.hpp"
#include "../../common/hostname.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fmt/core.h>
#include <fstream>
#include <iomanip>
#include <llama/llama.hpp>
#include <omp.h>
#include <vector>

constexpr auto PROBLEM_SIZE = std::size_t{1024 * 1024 * 128};
constexpr auto BLOCK_SIZE = std::size_t{256};
constexpr auto WARMUP_STEPS = 1;
constexpr auto STEPS = 5;
constexpr auto alpha = 3.14;

static_assert(PROBLEM_SIZE % BLOCK_SIZE == 0);

void daxpy(std::ofstream& plotFile)
{
    const auto* title = "baseline std::vector";
    std::cout << title << "\n";

    Stopwatch watch;
    using Vec = std::vector<double, llama::bloballoc::AlignedAllocator<double, 64>>;
    auto x = Vec(PROBLEM_SIZE);
    auto y = Vec(PROBLEM_SIZE);
    auto z = Vec(PROBLEM_SIZE);
    watch.printAndReset("alloc");

    for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i);
    }
    watch.printAndReset("init");

    double sum = 0;
    for(std::size_t s = 0; s < WARMUP_STEPS + STEPS; ++s)
    {
#pragma omp parallel for
        for(std::ptrdiff_t i = 0; i < PROBLEM_SIZE; i++)
            z[i] = alpha * x[i] + y[i];
        if(s < WARMUP_STEPS)
            watch.printAndReset("daxpy (warmup)");
        else
            sum += watch.printAndReset("daxpy");
    }
    plotFile << std::quoted(title) << "\t" << sum / STEPS << '\n';
}

template<typename Acc>
inline constexpr bool isGPU = false;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template<typename Dim, typename Idx>
inline constexpr bool isGPU<alpaka::AccGpuCudaRt<Dim, Idx>> = true;
#endif

template<typename Mapping>
void daxpy_alpaka_llama(std::string mappingName, std::ofstream& plotFile, Mapping mapping)
{
    std::size_t storageSize = 0;
    for(std::size_t i = 0; i < mapping.blobCount; i++)
        storageSize += mapping.blobSize(i);

    auto title = "alpaka/LLAMA " + std::move(mappingName);
    fmt::print("{0} (blobs size: {1}MiB)\n", title, storageSize / 1024 / 1024);

    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using Dev = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<Dev, alpaka::Blocking>;
    const auto devAcc = alpaka::getDevByIdx<alpaka::Pltf<Dev>>(0u);
    const auto devHost = alpaka::getDevByIdx<alpaka::PltfCpu>(0u);
    auto queue = Queue(devAcc);

    Stopwatch watch;
    auto x = llama::allocViewUninitialized(mapping);
    auto y = llama::allocViewUninitialized(mapping);
    auto z = llama::allocViewUninitialized(mapping);
    watch.printAndReset("alloc host");

    for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i);
    }
    watch.printAndReset("init host");

    auto alpakaBlobAlloc = llama::bloballoc::AlpakaBuf<Size, decltype(devAcc)>{devAcc};
    auto viewX = llama::allocViewUninitialized(mapping, alpakaBlobAlloc);
    auto viewY = llama::allocViewUninitialized(mapping, alpakaBlobAlloc);
    auto viewZ = llama::allocViewUninitialized(mapping, alpakaBlobAlloc);
    watch.printAndReset("alloc device");

    for(std::size_t i = 0; i < Mapping::blobCount; i++)
    {
        auto vx = alpaka::createView(devHost, &x.storageBlobs[0][0], mapping.blobSize(i));
        auto vy = alpaka::createView(devHost, &y.storageBlobs[0][0], mapping.blobSize(i));
        alpaka::memcpy(queue, viewX.storageBlobs[i], vx, mapping.blobSize(i));
        alpaka::memcpy(queue, viewY.storageBlobs[i], vy, mapping.blobSize(i));
    }
    watch.printAndReset("copy H->D");

    constexpr auto blockSize = isGPU<Acc> ? BLOCK_SIZE : 1;
    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>(
        alpaka::Vec<Dim, Size>{PROBLEM_SIZE / blockSize},
        alpaka::Vec<Dim, Size>{blockSize},
        alpaka::Vec<Dim, Size>{Size{1}});
    watch = {};

    double sum = 0;
    for(std::size_t s = 0; s < WARMUP_STEPS + STEPS; ++s)
    {
        auto kernel = [] ALPAKA_FN_ACC(
                          const Acc& acc,
                          decltype(llama::shallowCopy(viewX)) x,
                          decltype(llama::shallowCopy(viewY)) y,
                          double alpha,
                          decltype(llama::shallowCopy(viewZ)) z)
        {
            const auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            z[i] = alpha * x[i] + y[i];
        };
        alpaka::exec<Acc>(
            queue,
            workdiv,
            kernel,
            llama::shallowCopy(viewX),
            llama::shallowCopy(viewY),
            alpha,
            llama::shallowCopy(viewZ));
        if(s < WARMUP_STEPS)
            watch.printAndReset("daxpy (warmup)");
        else
            sum += watch.printAndReset("daxpy");
    }

    for(std::size_t i = 0; i < Mapping::blobCount; i++)
    {
        auto vz = alpaka::createView(devHost, &z.storageBlobs[0][0], mapping.blobSize(i));
        alpaka::memcpy(queue, vz, viewZ.storageBlobs[i], mapping.blobSize(i));
    }
    watch.printAndReset("copy D->H");

    plotFile << std::quoted(title) << "\t" << sum / STEPS << '\n';
}

auto main() -> int
try
{
    const auto numThreads = static_cast<std::size_t>(omp_get_max_threads());
    const char* affinity = std::getenv("GOMP_CPU_AFFINITY"); // NOLINT(concurrency-mt-unsafe)
    affinity = affinity == nullptr ? "NONE - PLEASE PIN YOUR THREADS!" : affinity;

    fmt::print(
        R"({}Mi doubles ({}MiB data)
Threads: {}
Affinity: {}
)",
        PROBLEM_SIZE / 1024 / 1024,
        PROBLEM_SIZE * sizeof(double) / 1024 / 1024,
        numThreads,
        affinity);

    std::ofstream plotFile{"daxpy.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# threads: {} affinity: {}
set title "daxpy CPU {}Mi doubles on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key off
set yrange [0:*]
set ylabel "runtime [s]"
$data << EOD
)",
        numThreads,
        affinity,
        PROBLEM_SIZE / 1024 / 1024,
        common::hostname());

    daxpy(plotFile);

    const auto extents = llama::ArrayExtents{PROBLEM_SIZE};
    daxpy_alpaka_llama("AoS", plotFile, llama::mapping::AoS{extents, double{}});
    daxpy_alpaka_llama(
        "SoA",
        plotFile,
        llama::mapping::SoA<llama::ArrayExtentsDynamic<std::size_t, 1>, double, false>{extents});
    daxpy_alpaka_llama(
        "SoA MB",
        plotFile,
        llama::mapping::SoA<llama::ArrayExtentsDynamic<std::size_t, 1>, double, true>{extents});
    daxpy_alpaka_llama(
        "Bytesplit",
        plotFile,
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<std::size_t, 1>, double, llama::mapping::BindAoS<>::fn>{
            extents});
    daxpy_alpaka_llama(
        "ChangeType D->F",
        plotFile,
        llama::mapping::ChangeType<
            llama::ArrayExtentsDynamic<std::size_t, 1>,
            double,
            llama::mapping::BindAoS<>::fn,
            boost::mp11::mp_list<boost::mp11::mp_list<double, float>>>{extents});
    daxpy_alpaka_llama("Bitpack 52^{11}", plotFile, llama::mapping::BitPackedFloatSoA{extents, 11, 52, double{}});
    daxpy_alpaka_llama(
        "Bitpack 52^{11} CT",
        plotFile,
        llama::mapping::BitPackedFloatSoA<
            llama::ArrayExtentsDynamic<std::size_t, 1>,
            double,
            llama::Constant<11>,
            llama::Constant<52>>{extents});
    daxpy_alpaka_llama("Bitpack 23^{8}", plotFile, llama::mapping::BitPackedFloatSoA{extents, 8, 23, double{}});
    daxpy_alpaka_llama(
        "Bitpack 23^{8} CT",
        plotFile,
        llama::mapping::BitPackedFloatSoA<
            llama::ArrayExtentsDynamic<std::size_t, 1>,
            double,
            llama::Constant<8>,
            llama::Constant<23>>{extents});
    daxpy_alpaka_llama("Bitpack 10^{5}", plotFile, llama::mapping::BitPackedFloatSoA{extents, 5, 10, double{}});
    daxpy_alpaka_llama(
        "Bitpack 10^{5} CT",
        plotFile,
        llama::mapping::BitPackedFloatSoA<
            llama::ArrayExtentsDynamic<std::size_t, 1>,
            double,
            llama::Constant<5>,
            llama::Constant<10>>{extents});

    plotFile << R"(EOD
plot $data using 2:xtic(1)
)";
    std::cout << "Plot with: ./daxpy.sh\n";

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

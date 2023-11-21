// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "../../common/Stopwatch.hpp"
#include "../../common/env.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fmt/core.h>
#include <fstream>
#include <iomanip>
#include <llama/llama.hpp>
#include <omp.h>
#include <vector>

constexpr auto problemSize = std::size_t{1024} * 1024 * 128;
constexpr auto gpuBlockSize = std::size_t{256};
constexpr auto warmupSteps = 1;
constexpr auto steps = 5;
constexpr auto alpha = 3.14;

static_assert(problemSize % gpuBlockSize == 0);

void daxpy(std::ofstream& plotFile)
{
    const auto* title = "baseline std::vector";
    std::cout << title << "\n";

    Stopwatch watch;
    using Vec = std::vector<double, llama::bloballoc::AlignedAllocator<double, 64>>;
    auto x = Vec(problemSize);
    auto y = Vec(problemSize);
    auto z = Vec(problemSize);
    watch.printAndReset("alloc");

    for(std::size_t i = 0; i < problemSize; ++i)
    {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i);
    }
    watch.printAndReset("init");

    double sum = 0;
    for(std::size_t s = 0; s < warmupSteps + steps; ++s)
    {
#pragma omp parallel for
        for(std::ptrdiff_t i = 0; i < problemSize; i++)
            z[i] = alpha * x[i] + y[i];
        if(s < warmupSteps)
            watch.printAndReset("daxpy (warmup)");
        else
            sum += watch.printAndReset("daxpy");
    }
    plotFile << std::quoted(title) << "\t" << sum / steps << '\n';
}

template<typename Acc>
inline constexpr bool isGPU = false;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template<typename Dim, typename Idx>
inline constexpr bool isGPU<alpaka::AccGpuCudaRt<Dim, Idx>> = true;
#endif

template<typename Mapping>
void daxpyAlpakaLlama(std::string mappingName, std::ofstream& plotFile, Mapping mapping)
{
    std::size_t storageSize = 0;
    for(std::size_t i = 0; i < mapping.blobCount; i++)
        storageSize += mapping.blobSize(i);

    auto title = "alpaka/LLAMA " + std::move(mappingName);
    fmt::print("{0} (blobs size: {1}MiB)\n", title, storageSize / 1024 / 1024);

    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    const auto platformAcc = alpaka::Platform<Acc>{};
    const auto platformHost = alpaka::PlatformCpu{};
    const auto devAcc = alpaka::getDevByIdx(platformAcc, 0);
    const auto devHost = alpaka::getDevByIdx(platformHost, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>(devAcc);

    Stopwatch watch;
    auto x = llama::allocViewUninitialized(mapping);
    auto y = llama::allocViewUninitialized(mapping);
    auto z = llama::allocViewUninitialized(mapping);
    watch.printAndReset("alloc host");

    for(std::size_t i = 0; i < problemSize; ++i)
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
        auto vx = alpaka::createView(devHost, &x.blobs()[0][0], mapping.blobSize(i));
        auto vy = alpaka::createView(devHost, &y.blobs()[0][0], mapping.blobSize(i));
        alpaka::memcpy(queue, viewX.blobs()[i], vx);
        alpaka::memcpy(queue, viewY.blobs()[i], vy);
    }
    watch.printAndReset("copy H->D");

    constexpr auto blockSize = isGPU<Acc> ? gpuBlockSize : 1;
    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>(
        alpaka::Vec<Dim, Size>{problemSize / blockSize},
        alpaka::Vec<Dim, Size>{blockSize},
        alpaka::Vec<Dim, Size>{Size{1}});
    watch = {};

    double sum = 0;
    for(std::size_t s = 0; s < warmupSteps + steps; ++s)
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
        if(s < warmupSteps)
            watch.printAndReset("daxpy (warmup)");
        else
            sum += watch.printAndReset("daxpy");
    }

    for(std::size_t i = 0; i < Mapping::blobCount; i++)
    {
        auto vz = alpaka::createView(devHost, &z.blobs()[0][0], mapping.blobSize(i));
        alpaka::memcpy(queue, vz, viewZ.blobs()[i]);
    }
    watch.printAndReset("copy D->H");

    plotFile << std::quoted(title) << "\t" << sum / steps << '\n';
}

auto main() -> int
try
{
    const auto env = common::captureEnv();

    fmt::print(
        "{}Mi doubles ({}MiB data)\n{}\n",
        problemSize / 1024 / 1024,
        problemSize * sizeof(double) / 1024 / 1024,
        env);

    std::ofstream plotFile{"daxpy.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# {}
set title "daxpy CPU {}Mi doubles"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key off
set yrange [0:*]
set ylabel "runtime [s]"
$data << EOD
)",
        env,
        problemSize / 1024 / 1024);

    daxpy(plotFile);

    const auto extents = llama::ArrayExtentsDynamic<std::size_t, 1>{problemSize};
    daxpyAlpakaLlama(
        "AoS",
        plotFile,
        llama::mapping::AoS<llama::ArrayExtentsDynamic<std::size_t, 1>, double>{extents});
    daxpyAlpakaLlama(
        "SoA",
        plotFile,
        llama::mapping::SoA<llama::ArrayExtentsDynamic<std::size_t, 1>, double, llama::mapping::Blobs::Single>{
            extents});
    daxpyAlpakaLlama(
        "SoA MB",
        plotFile,
        llama::mapping::SoA<llama::ArrayExtentsDynamic<std::size_t, 1>, double, llama::mapping::Blobs::OnePerField>{
            extents});
    daxpyAlpakaLlama(
        "Bytesplit",
        plotFile,
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<std::size_t, 1>, double, llama::mapping::BindAoS<>::fn>{
            extents});
    daxpyAlpakaLlama(
        "ChangeType D->F",
        plotFile,
        llama::mapping::ChangeType<
            llama::ArrayExtentsDynamic<std::size_t, 1>,
            double,
            llama::mapping::BindAoS<>::fn,
            boost::mp11::mp_list<boost::mp11::mp_list<double, float>>>{extents});
    daxpyAlpakaLlama(
        "Bitpack 52^{11}",
        plotFile,
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, double, unsigned, unsigned>{
            extents,
            11,
            52,
            double{}});
    daxpyAlpakaLlama(
        "Bitpack 52^{11} CT",
        plotFile,
        llama::mapping::BitPackedFloatSoA<
            llama::ArrayExtentsDynamic<std::size_t, 1>,
            double,
            llama::Constant<11>,
            llama::Constant<52>>{extents});
    daxpyAlpakaLlama(
        "Bitpack 23^{8}",
        plotFile,
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, double, unsigned, unsigned>{
            extents,
            8,
            23,
            double{}});
    daxpyAlpakaLlama(
        "Bitpack 23^{8} CT",
        plotFile,
        llama::mapping::BitPackedFloatSoA<
            llama::ArrayExtentsDynamic<std::size_t, 1>,
            double,
            llama::Constant<8>,
            llama::Constant<23>>{extents});
    daxpyAlpakaLlama(
        "Bitpack 10^{5}",
        plotFile,
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, double, unsigned, unsigned>{
            extents,
            5,
            10,
            double{}});
    daxpyAlpakaLlama(
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

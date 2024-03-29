// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: CC0-1.0

#include "../../common/Stats.hpp"
#include "../../common/Stopwatch.hpp"
#include "../../common/env.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

using Size = std::size_t;

constexpr auto problemSize = Size{64 * 1024 * 1024 + 3}; ///< problem size
constexpr auto steps = 20; // excluding 1 warmup run
constexpr auto aosoaLanes = 32;
constexpr auto elements = Size{32};

// use different types for various struct members to alignment/padding plays a role
using X_t = float;
using Y_t = double;
using Z_t = std::uint16_t;

// clang-format off
namespace tag
{
    struct X{} x;
    struct Y{} y;
    struct Z{} z;
} // namespace tag

using Vector = llama::Record<
    llama::Field<tag::X, X_t>,
    llama::Field<tag::Y, Y_t>,
    llama::Field<tag::Z, Z_t>>;
// clang-format on

template<typename Acc>
inline constexpr bool isGpu = false;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template<typename Dim, typename Size>
inline constexpr bool isGpu<alpaka::AccGpuCudaRt<Dim, Size>> = true;
#endif

template<std::size_t Elems>
struct Kernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View a, View b) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto [n] = a.extents();
        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, n);

        LLAMA_INDEPENDENT_DATA
        for(auto i = start; i < end; ++i)
        {
            a(i)(tag::X{}) += b(i)(tag::X{});
            a(i)(tag::Y{}) -= b(i)(tag::Y{});
            a(i)(tag::Z{}) *= b(i)(tag::Z{});
        }
    }
};

template<int MappingIndex>
void run(std::ofstream& plotFile)
try
{
    const auto mappingname = [&]
    {
        if constexpr(MappingIndex == 0)
            return "AoS";
        if constexpr(MappingIndex == 1)
            return "SoA SB packed";
        if constexpr(MappingIndex == 2)
            return "SoA SB aligned";
        if constexpr(MappingIndex == 3)
            return "SoA SB aligned CT size";
        if constexpr(MappingIndex == 4)
            return "SoA MB";
        if constexpr(MappingIndex == 5)
            return "AoSoA" + std::to_string(aosoaLanes);
    }();

    std::cout << "\nLLAMA " << mappingname << "\n";

    // alpaka
    using Dim = alpaka::DimInt<1>;

    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    const auto platformAcc = alpaka::Platform<Acc>{};
    const auto platformHost = alpaka::PlatformCpu{};
    const auto devAcc = alpaka::getDevByIdx(platformAcc, 0);
    const auto devHost = alpaka::getDevByIdx(platformHost, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{devAcc};

    // LLAMA
    const auto mapping = [&]
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<Size, 1>;
        const auto extents = ArrayExtents{problemSize};
        if constexpr(MappingIndex == 0)
            return llama::mapping::AoS{extents, Vector{}};
        if constexpr(MappingIndex == 1)
        {
            if constexpr(isGpu<Acc>)
                throw std::runtime_error{"Misaligned memory access not supported"};
            return llama::mapping::
                SoA<ArrayExtents, Vector, llama::mapping::Blobs::Single, llama::mapping::SubArrayAlignment::Pack>{
                    extents};
        }
        if constexpr(MappingIndex == 2)
            return llama::mapping::
                SoA<ArrayExtents, Vector, llama::mapping::Blobs::Single, llama::mapping::SubArrayAlignment::Align>{
                    extents};
        if constexpr(MappingIndex == 3)
            return llama::mapping::SoA<
                llama::ArrayExtents<Size, problemSize>,
                Vector,
                llama::mapping::Blobs::Single,
                llama::mapping::SubArrayAlignment::Align>{};
        if constexpr(MappingIndex == 4)
            return llama::mapping::SoA<ArrayExtents, Vector, llama::mapping::Blobs::OnePerField>{extents};
        if constexpr(MappingIndex == 5)
            return llama::mapping::AoSoA<ArrayExtents, Vector, aosoaLanes>{extents};
    }();

    std::cout << problemSize / 1000 / 1000 << " million vectors\n"
              << problemSize * llama::sizeOf<Vector> * 2 / 1000 / 1000 << " MB on device\n";

    Stopwatch chrono;

    // allocate LLAMA views
    auto hostA = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devHost)>{devHost});
    auto hostB = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devHost)>{devHost});
    auto devA = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devAcc)>{devAcc});
    auto devB = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devAcc)>{devAcc});

    chrono.printAndReset("Alloc views");

    LLAMA_INDEPENDENT_DATA
    for(std::size_t i = 0; i < problemSize; ++i)
    {
        hostA(i) = i;
        hostB(i) = i;
    }
    chrono.printAndReset("Init");

    auto copyBlobAlpaka
        = [&](const auto& srcBlob, auto& dstBlob, std::size_t size) { alpaka::memcpy(queue, dstBlob, srcBlob, size); };
    llama::copyBlobs(hostA, devA, copyBlobAlpaka);
    llama::copyBlobs(hostB, devB, copyBlobAlpaka);
    chrono.printAndReset("Copy H->D");

    const auto workdiv = alpaka::getValidWorkDiv<Acc>(devAcc, problemSize, elements, false);
    std::cout << "Workdiv: " << workdiv << "\n";

    common::Stats stats;
    for(std::size_t s = 0; s < steps + 1; ++s)
    {
        alpaka::exec<Acc>(queue, workdiv, Kernel<elements>{}, llama::shallowCopy(devA), llama::shallowCopy(devB));
        stats(chrono.printAndReset("vectoradd"));
    }
    plotFile << "\"LLAMA " << mappingname << "\"\t" << stats.mean() << "\t" << stats.sem() << '\n';

    llama::copyBlobs(devA, hostA, copyBlobAlpaka);
    llama::copyBlobs(devB, hostB, copyBlobAlpaka);
    chrono.printAndReset("Copy D->H");
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

auto main() -> int
{
    using Acc = alpaka::ExampleDefaultAcc<alpaka::DimInt<1>, Size>;
    const auto env = common::captureEnv<Acc>();
    std::cout << problemSize / 1000 / 1000 << "M values "
              << "(" << problemSize * sizeof(float) / 1024 << "kiB)\n"
              << env << '\n';

    std::ofstream plotFile{"vectoradd_alpaka.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# {}
set title "vectoradd alpaka {}Mi elements on {}"
set style data histograms
set style histogram errorbars
set style fill solid border -1
set xtics rotate by 45 right nomirror
set key off
set yrange [0:*]
set ylabel "runtime [s]"
$data << EOD
""	"runtime"	"runtime_sem"
)",
        env,
        problemSize / 1024 / 1024,
        alpaka::getAccName<Acc>());

    boost::mp11::mp_for_each<boost::mp11::mp_iota_c<6>>([&](auto ic) { run<decltype(ic)::value>(plotFile); });

    plotFile << R"(EOD
plot $data using 2:3:xtic(1) ti col
)";
    std::cout << "Plot with: ./vectoradd_alpaka.sh\n";
    return 0;
}

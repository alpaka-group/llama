#include "../../common/Stopwatch.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <string>
#include <utility>

using FP = float;

constexpr auto PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto STEPS = 5; ///< number of steps to calculate
constexpr auto TIMESTEP = FP{0.0001};
constexpr auto ALLOW_RSQRT = true; // rsqrt can be way faster, but less accurate

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        error Cannot enable CUDA together with other backends, because nvcc cannot parse the Vc header, sorry :/
#    endif
// nvcc fails to compile Vc headers even if nothing is used from there, so we need to conditionally include it
#    include <Vc/Vc>
constexpr auto DESIRED_ELEMENTS_PER_THREAD = Vc::float_v::size();
constexpr auto THREADS_PER_BLOCK = 1;
constexpr auto AOSOA_LANES = Vc::float_v::size(); // vectors
#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
constexpr auto DESIRED_ELEMENTS_PER_THREAD = 1;
constexpr auto THREADS_PER_BLOCK = 256;
constexpr auto AOSOA_LANES = 32; // coalesced memory access
#else
#    error "Unsupported backend"
#endif

// makes our life easier for now
static_assert(PROBLEM_SIZE % (DESIRED_ELEMENTS_PER_THREAD * THREADS_PER_BLOCK) == 0);

constexpr FP EPS2 = 0.01;

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
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>>,
    llama::Field<tag::Vel, llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

enum Mapping
{
    AoS,
    SoA,
    AoSoA
};

namespace stdext
{
    LLAMA_FN_HOST_ACC_INLINE auto rsqrt(FP f) -> FP
    {
        return 1.0f / std::sqrt(f);
    }
} // namespace stdext

// FIXME: this makes assumptions that there are always float_v::size() many values blocked in the LLAMA view
template<typename Vec>
LLAMA_FN_HOST_ACC_INLINE auto load(const FP& src)
{
    if constexpr(std::is_same_v<Vec, FP>)
        return src;
    else
        return Vec(&src);
}

template<typename Vec>
LLAMA_FN_HOST_ACC_INLINE auto broadcast(const FP& src)
{
    return Vec(src);
}

template<typename Vec>
LLAMA_FN_HOST_ACC_INLINE auto store(FP& dst, Vec v)
{
    if constexpr(std::is_same_v<Vec, FP>)
        dst = v;
    else
        v.store(&dst);
}

template<std::size_t Elems>
struct VecType
{
    // TODO(bgruber): we need a vector type that also works on GPUs
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    using type = Vc::SimdArray<FP, Elems>;
#endif
};
template<>
struct VecType<1>
{
    using type = FP;
};

template<std::size_t Elems, typename ViewParticleI, typename VirtualParticleJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(ViewParticleI pi, VirtualParticleJ pj)
{
    using Vec = typename VecType<Elems>::type;

    using std::sqrt;
    using stdext::rsqrt;
#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    using Vc::rsqrt;
    using Vc::sqrt;
#endif

    const Vec xdistance = load<Vec>(pi(tag::Pos{}, tag::X{})) - broadcast<Vec>(pj(tag::Pos{}, tag::X{}));
    const Vec ydistance = load<Vec>(pi(tag::Pos{}, tag::Y{})) - broadcast<Vec>(pj(tag::Pos{}, tag::Y{}));
    const Vec zdistance = load<Vec>(pi(tag::Pos{}, tag::Z{})) - broadcast<Vec>(pj(tag::Pos{}, tag::Z{}));
    const Vec xdistanceSqr = xdistance * xdistance;
    const Vec ydistanceSqr = ydistance * ydistance;
    const Vec zdistanceSqr = zdistance * zdistance;
    const Vec distSqr = +EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
    const Vec distSixth = distSqr * distSqr * distSqr;
    const Vec invDistCube = ALLOW_RSQRT ? rsqrt(distSixth) : (1.0f / sqrt(distSixth));
    const Vec sts = broadcast<Vec>(pj(tag::Mass())) * invDistCube * TIMESTEP;
    store<Vec>(pi(tag::Vel{}, tag::X{}), xdistanceSqr * sts + load<Vec>(pi(tag::Vel{}, tag::X{})));
    store<Vec>(pi(tag::Vel{}, tag::Y{}), ydistanceSqr * sts + load<Vec>(pi(tag::Vel{}, tag::Y{})));
    store<Vec>(pi(tag::Vel{}, tag::Z{}), zdistanceSqr * sts + load<Vec>(pi(tag::Vel{}, tag::Z{})));
}

template<std::size_t ProblemSize, std::size_t Elems, std::size_t BlockSize, Mapping MappingSM>
struct UpdateKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles) const
    {
        auto sharedView = [&]
        {
            // if there is only 1 thread per block, use stack instead of shared memory
            if constexpr(BlockSize == 1)
                return llama::allocViewStack<View::ArrayDims::rank, typename View::RecordDim>();
            else
            {
                constexpr auto sharedMapping = []
                {
                    constexpr auto arrayDims = llama::ArrayDims<1>{BlockSize};
                    if constexpr(MappingSM == AoS)
                        return llama::mapping::AoS{arrayDims, Particle{}};
                    if constexpr(MappingSM == SoA)
                        return llama::mapping::SoA<decltype(arrayDims), Particle, false>{arrayDims};
                    if constexpr(MappingSM == AoSoA)
                        return llama::mapping::AoSoA<decltype(arrayDims), Particle, AOSOA_LANES>{arrayDims};
                }();
                static_assert(decltype(sharedMapping)::blobCount == 1);

                constexpr auto sharedMemSize = llama::sizeOf<typename View::RecordDim> * BlockSize;
                auto& sharedMem = alpaka::declareSharedVar<std::byte[sharedMemSize], __COUNTER__>(acc);
                return llama::View{sharedMapping, llama::Array<std::byte*, 1>{&sharedMem[0]}};
            }
        }();

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        // TODO(bgruber): we could optimize here, because only velocity is ever updated
        auto pi = [&]
        {
            constexpr auto arrayDims = llama::ArrayDims<1>{Elems};
            constexpr auto mapping
                = llama::mapping::SoA<typename View::ArrayDims, typename View::RecordDim, false>{arrayDims};
            constexpr auto blobAlloc = llama::bloballoc::Stack<llama::sizeOf<typename View::RecordDim> * Elems>{};
            return llama::allocViewUninitialized(mapping, blobAlloc);
        }();
        // TODO(bgruber): vector load
        LLAMA_INDEPENDENT_DATA
        for(auto e = 0u; e < Elems; e++)
            pi(e) = particles(ti * Elems + e);

        LLAMA_INDEPENDENT_DATA
        for(std::size_t blockOffset = 0; blockOffset < ProblemSize; blockOffset += BlockSize)
        {
            LLAMA_INDEPENDENT_DATA
            for(auto j = tbi; j < BlockSize; j += THREADS_PER_BLOCK)
                sharedView(j) = particles(blockOffset + j);
            alpaka::syncBlockThreads(acc);

            LLAMA_INDEPENDENT_DATA
            for(auto j = std::size_t{0}; j < BlockSize; ++j)
                pPInteraction<Elems>(pi(0u), sharedView(j));
            alpaka::syncBlockThreads(acc);
        }
        // TODO(bgruber): vector store
        LLAMA_INDEPENDENT_DATA
        for(auto e = 0u; e < Elems; e++)
            particles(ti * Elems + e) = pi(e);
    }
};

template<std::size_t ProblemSize, std::size_t Elems>
struct MoveKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto i = ti * Elems;

        using Vec = typename VecType<Elems>::type;
        store<Vec>(
            particles(i)(tag::Pos{}, tag::X{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::X{})) + load<Vec>(particles(i)(tag::Vel{}, tag::X{})) * TIMESTEP);
        store<Vec>(
            particles(i)(tag::Pos{}, tag::Y{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::Y{})) + load<Vec>(particles(i)(tag::Vel{}, tag::Y{})) * TIMESTEP);
        store<Vec>(
            particles(i)(tag::Pos{}, tag::Z{}),
            load<Vec>(particles(i)(tag::Pos{}, tag::Z{})) + load<Vec>(particles(i)(tag::Vel{}, tag::Z{})) * TIMESTEP);
    }
};

template<template<typename, typename> typename AccTemplate, Mapping MappingGM, Mapping MappingSM>
void run(std::ostream& plotFile)
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = AccTemplate<Dim, Size>;
    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto mappingName = [](int m) -> std::string
    {
        if(m == 0)
            return "AoS";
        if(m == 1)
            return "SoA";
        if(m == 2)
            return "AoSoA" + std::to_string(AOSOA_LANES);
        std::abort();
    };
    const auto title = "GM " + mappingName(MappingGM) + " SM " + mappingName(MappingSM);
    std::cout << '\n' << title << '\n';

    const DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    auto mapping = []
    {
        const auto arrayDims = llama::ArrayDims{PROBLEM_SIZE};
        if constexpr(MappingGM == AoS)
            return llama::mapping::AoS<decltype(arrayDims), Particle>{arrayDims};
        if constexpr(MappingGM == SoA)
            return llama::mapping::SoA<decltype(arrayDims), Particle, false>{arrayDims};
        // if constexpr (MappingGM == 2)
        //    return llama::mapping::SoA<decltype(arrayDims), Particle, true>{arrayDims};
        if constexpr(MappingGM == AoSoA)
            return llama::mapping::AoSoA<decltype(arrayDims), Particle, AOSOA_LANES>{arrayDims};
    }();

    Stopwatch watch;

    const auto bufferSize = Size(mapping.blobSize(0));

    auto hostBuffer = alpaka::allocBuf<std::byte, Size>(devHost, bufferSize);
    auto accBuffer = alpaka::allocBuf<std::byte, Size>(devAcc, bufferSize);

    watch.printAndReset("alloc");

    auto hostView = llama::View(mapping, llama::Array{alpaka::getPtrNative(hostBuffer)});
    auto accView = llama::View(mapping, llama::Array{alpaka::getPtrNative(accBuffer)});

    watch.printAndReset("views");

    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(generator);
        p(tag::Pos(), tag::Y()) = distribution(generator);
        p(tag::Pos(), tag::Z()) = distribution(generator);
        p(tag::Vel(), tag::X()) = distribution(generator) / FP(10);
        p(tag::Vel(), tag::Y()) = distribution(generator) / FP(10);
        p(tag::Vel(), tag::Z()) = distribution(generator) / FP(10);
        p(tag::Mass()) = distribution(generator) / FP(100);
        hostView(i) = p;
    }

    watch.printAndReset("init");

    alpaka::memcpy(queue, accBuffer, hostBuffer, bufferSize);
    watch.printAndReset("copy H->D");

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{static_cast<Size>(PROBLEM_SIZE / (THREADS_PER_BLOCK * DESIRED_ELEMENTS_PER_THREAD))},
        alpaka::Vec<Dim, Size>{static_cast<Size>(THREADS_PER_BLOCK)},
        alpaka::Vec<Dim, Size>{static_cast<Size>(DESIRED_ELEMENTS_PER_THREAD)}};

    double sumUpdate = 0;
    double sumMove = 0;
    for(std::size_t s = 0; s < STEPS; ++s)
    {
        auto updateKernel = UpdateKernel<PROBLEM_SIZE, DESIRED_ELEMENTS_PER_THREAD, THREADS_PER_BLOCK, MappingSM>{};
        alpaka::exec<Acc>(queue, workdiv, updateKernel, accView);
        sumUpdate += watch.printAndReset("update", '\t');

        auto moveKernel = MoveKernel<PROBLEM_SIZE, DESIRED_ELEMENTS_PER_THREAD>{};
        alpaka::exec<Acc>(queue, workdiv, moveKernel, accView);
        sumMove += watch.printAndReset("move");
    }
    plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

    alpaka::memcpy(queue, hostBuffer, accBuffer, bufferSize);
    watch.printAndReset("copy D->H");
}

auto main() -> int
try
{
    std::cout << PROBLEM_SIZE / 1000 << "k particles (" << PROBLEM_SIZE * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << THREADS_PER_BLOCK << " particles ("
              << THREADS_PER_BLOCK * llama::sizeOf<Particle> / 1024 << " kiB) in shared memory\n"
              << "Reducing on " << DESIRED_ELEMENTS_PER_THREAD << " particles per thread\n"
              << "Using " << THREADS_PER_BLOCK << " threads per block\n";
    std::cout << std::fixed;

    std::ofstream plotFile{"nbody.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    std::ofstream{"nbody.sh"} << R"(#!/usr/bin/gnuplot -p
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
$data << EOD
)";
    plotFile << "\"\"\t\"update\"\t\"move\"\n";

    // using Acc = alpaka::ExampleDefaultAcc;
    // using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::AccCpuSerial<Dim, Size>;
    // using Acc = alpaka::AccCpuOmp2Blocks<Dim, Size>;

    run<alpaka::ExampleDefaultAcc, AoS, AoS>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoS, SoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoS, AoSoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, SoA, AoS>(plotFile);
    run<alpaka::ExampleDefaultAcc, SoA, SoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, SoA, AoSoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoSoA, AoS>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoSoA, SoA>(plotFile);
    run<alpaka::ExampleDefaultAcc, AoSoA, AoSoA>(plotFile);

    plotFile << R"(EOD
plot $data using 2:xtic(1) ti col
)";
    std::cout << "Plot with: ./nbody.sh\n";

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

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

#if __has_include(<xsimd/xsimd.hpp>)
#    include <xsimd/xsimd.hpp>
#    define HAVE_XSIMD
#endif

using FP = float;

constexpr auto problemSize = 16 * 1024; ///< total number of particles
constexpr auto steps = 5; ///< number of steps to calculate
constexpr auto timestep = FP{0.0001};
constexpr auto allowRsqrt = true; // rsqrt can be way faster, but less accurate

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#    if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#        error Cannot enable CUDA together with other backends
#    endif
constexpr auto desiredElementsPerThread = xsimd::batch<float>::size;
constexpr auto threadsPerBlock = 1;
constexpr auto aosoaLanes = xsimd::batch<float>::size; // vectors
#elif defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
constexpr auto desiredElementsPerThread = 1;
constexpr auto threadsPerBlock = 256;
constexpr auto aosoaLanes = 32; // coalesced memory access
#else
#    error "Unsupported backend"
#endif

// makes our life easier for now
static_assert(problemSize % (desiredElementsPerThread * threadsPerBlock) == 0);

constexpr FP epS2 = 0.01;

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

// FIXME: this makes assumptions that there are always Simd::size many values blocked in the LLAMA view
template<typename Simd>
LLAMA_FN_HOST_ACC_INLINE auto load(const FP& src)
{
    if constexpr(std::is_same_v<Simd, FP>)
        return src;
    else
        return Simd::load_unaligned(&src);
}

template<typename Simd>
LLAMA_FN_HOST_ACC_INLINE auto broadcast(const FP& src)
{
    return Simd(src);
}

template<typename Vec>
LLAMA_FN_HOST_ACC_INLINE auto store(FP& dst, Vec v)
{
    if constexpr(std::is_same_v<Vec, FP>)
        dst = v;
    else
        v.store_unaligned(&dst);
}

template<int Elems>
struct SimdType
{
    // TODO(bgruber): we need a vector type that also works on GPUs
    using type = xsimd::make_sized_batch_t<FP, Elems>;
    static_assert(!std::is_void_v<type>, "xsimd does not have a SIMD type for this element count");
};

template<>
struct SimdType<1>
{
    using type = FP;
};

template<int Elems, typename ViewParticleI, typename ParticleRefJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(ViewParticleI pi, ParticleRefJ pj)
{
    using Simd = typename SimdType<Elems>::type;

    using std::sqrt;
    using stdext::rsqrt;
    using xsimd::rsqrt;
    using xsimd::sqrt;

    const Simd xdistance = load<Simd>(pi(tag::Pos{}, tag::X{})) - broadcast<Simd>(pj(tag::Pos{}, tag::X{}));
    const Simd ydistance = load<Simd>(pi(tag::Pos{}, tag::Y{})) - broadcast<Simd>(pj(tag::Pos{}, tag::Y{}));
    const Simd zdistance = load<Simd>(pi(tag::Pos{}, tag::Z{})) - broadcast<Simd>(pj(tag::Pos{}, tag::Z{}));
    const Simd xdistanceSqr = xdistance * xdistance;
    const Simd ydistanceSqr = ydistance * ydistance;
    const Simd zdistanceSqr = zdistance * zdistance;
    const Simd distSqr = +epS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
    const Simd distSixth = distSqr * distSqr * distSqr;
    const Simd invDistCube = allowRsqrt ? rsqrt(distSixth) : (1.0f / sqrt(distSixth));
    const Simd sts = broadcast<Simd>(pj(tag::Mass())) * invDistCube * timestep;
    store<Simd>(pi(tag::Vel{}, tag::X{}), xdistanceSqr * sts + load<Simd>(pi(tag::Vel{}, tag::X{})));
    store<Simd>(pi(tag::Vel{}, tag::Y{}), ydistanceSqr * sts + load<Simd>(pi(tag::Vel{}, tag::Y{})));
    store<Simd>(pi(tag::Vel{}, tag::Z{}), zdistanceSqr * sts + load<Simd>(pi(tag::Vel{}, tag::Z{})));
}

template<int ProblemSize, int Elems, int BlockSize, Mapping MappingSM>
struct UpdateKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles) const
    {
        auto sharedView = [&]
        {
            // if there is only 1 thread per block, use stack instead of shared memory
            if constexpr(BlockSize == 1)
                return llama::allocViewStack<View::ArrayExtents::rank, typename View::RecordDim>();
            else
            {
                constexpr auto sharedMapping = []
                {
                    using ArrayExtents = llama::ArrayExtents<int, BlockSize>;
                    if constexpr(MappingSM == AoS)
                        return llama::mapping::AoS<ArrayExtents, Particle>{};
                    if constexpr(MappingSM == SoA)
                        return llama::mapping::SoA<ArrayExtents, Particle, false>{};
                    if constexpr(MappingSM == AoSoA)
                        return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{};
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
            constexpr auto mapping
                = llama::mapping::SoA<llama::ArrayExtents<int, Elems>, typename View::RecordDim, false>{};
            return llama::allocViewUninitialized(mapping, llama::bloballoc::Stack<mapping.blobSize(0)>{});
        }();
        // TODO(bgruber): vector load
        LLAMA_INDEPENDENT_DATA
        for(auto e = 0u; e < Elems; e++)
            pi(e) = particles(ti * Elems + e);

        LLAMA_INDEPENDENT_DATA
        for(int blockOffset = 0; blockOffset < ProblemSize; blockOffset += BlockSize)
        {
            LLAMA_INDEPENDENT_DATA
            for(int j = tbi; j < BlockSize; j += threadsPerBlock)
                sharedView(j) = particles(blockOffset + j);
            alpaka::syncBlockThreads(acc);

            LLAMA_INDEPENDENT_DATA
            for(int j = 0; j < BlockSize; ++j)
                pPInteraction<Elems>(pi(0u), sharedView(j));
            alpaka::syncBlockThreads(acc);
        }
        // TODO(bgruber): vector store
        LLAMA_INDEPENDENT_DATA
        for(int e = 0u; e < Elems; e++)
            particles(ti * Elems + e) = pi(e);
    }
};

template<int ProblemSize, int Elems>
struct MoveKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto i = ti * Elems;

        using Simd = typename SimdType<Elems>::type;
        store<Simd>(
            particles(i)(tag::Pos{}, tag::X{}),
            load<Simd>(particles(i)(tag::Pos{}, tag::X{}))
                + load<Simd>(particles(i)(tag::Vel{}, tag::X{})) * timestep);
        store<Simd>(
            particles(i)(tag::Pos{}, tag::Y{}),
            load<Simd>(particles(i)(tag::Pos{}, tag::Y{}))
                + load<Simd>(particles(i)(tag::Vel{}, tag::Y{})) * timestep);
        store<Simd>(
            particles(i)(tag::Pos{}, tag::Z{}),
            load<Simd>(particles(i)(tag::Pos{}, tag::Z{}))
                + load<Simd>(particles(i)(tag::Vel{}, tag::Z{})) * timestep);
    }
};

template<template<typename, typename> typename AccTemplate, Mapping MappingGM, Mapping MappingSM>
void run(std::ostream& plotFile)
{
    using Dim = alpaka::DimInt<1>;
    using Size = int;
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
            return "AoSoA" + std::to_string(aosoaLanes);
        std::abort();
    };
    const auto title = "GM " + mappingName(MappingGM) + " SM " + mappingName(MappingSM);
    std::cout << '\n' << title << '\n';

    const DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    auto mapping = []
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
        const auto extents = ArrayExtents{problemSize};
        if constexpr(MappingGM == AoS)
            return llama::mapping::AoS<ArrayExtents, Particle>{extents};
        if constexpr(MappingGM == SoA)
            return llama::mapping::SoA<ArrayExtents, Particle, false>{extents};
        // if constexpr (MappingGM == 2)
        //    return llama::mapping::SoA<ArrayExtents, Particle, true>{extents};
        if constexpr(MappingGM == AoSoA)
            return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{extents};
    }();

    Stopwatch watch;

    auto hostView
        = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devHost)>{devHost});
    auto accView = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devAcc)>{devAcc});
    watch.printAndReset("alloc views");

    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP{0}, FP{1});
    for(int i = 0; i < problemSize; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(generator);
        p(tag::Pos(), tag::Y()) = distribution(generator);
        p(tag::Pos(), tag::Z()) = distribution(generator);
        p(tag::Vel(), tag::X()) = distribution(generator) / FP{10};
        p(tag::Vel(), tag::Y()) = distribution(generator) / FP{10};
        p(tag::Vel(), tag::Z()) = distribution(generator) / FP{10};
        p(tag::Mass()) = distribution(generator) / FP{100};
        hostView(i) = p;
    }
    watch.printAndReset("init");

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, accView.storageBlobs[i], hostView.storageBlobs[i]);
    watch.printAndReset("copy H->D");

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{static_cast<Size>(problemSize / (threadsPerBlock * desiredElementsPerThread))},
        alpaka::Vec<Dim, Size>{static_cast<Size>(threadsPerBlock)},
        alpaka::Vec<Dim, Size>{static_cast<Size>(desiredElementsPerThread)}};

    double sumUpdate = 0;
    double sumMove = 0;
    for(int s = 0; s < steps; ++s)
    {
        auto updateKernel = UpdateKernel<problemSize, desiredElementsPerThread, threadsPerBlock, MappingSM>{};
        alpaka::exec<Acc>(queue, workdiv, updateKernel, llama::shallowCopy(accView));
        sumUpdate += watch.printAndReset("update", '\t');

        auto moveKernel = MoveKernel<problemSize, desiredElementsPerThread>{};
        alpaka::exec<Acc>(queue, workdiv, moveKernel, llama::shallowCopy(accView));
        sumMove += watch.printAndReset("move");
    }
    plotFile << std::quoted(title) << "\t" << sumUpdate / steps << '\t' << sumMove / steps << '\n';

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, hostView.storageBlobs[i], accView.storageBlobs[i]);
    watch.printAndReset("copy D->H");
}

auto main() -> int
try
{
    std::cout << problemSize / 1000 << "k particles (" << problemSize * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << threadsPerBlock << " particles (" << threadsPerBlock * llama::sizeOf<Particle> / 1024
              << " kiB) in shared memory\n"
              << "Reducing on " << desiredElementsPerThread << " particles per thread\n"
              << "Using " << threadsPerBlock << " threads per block\n";
    std::cout << std::fixed;

    std::ofstream plotFile{"nbody.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << R"(#!/usr/bin/gnuplot -p
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

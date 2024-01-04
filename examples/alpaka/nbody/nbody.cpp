// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "../../common/Stats.hpp"
#include "../../common/Stopwatch.hpp"
#include "../../common/env.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <fmt/format.h>
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

constexpr auto problemSize = 64 * 1024; ///< total number of particles
constexpr auto steps = 20; ///< number of steps to calculate, excluding 1 warmup run
constexpr auto allowRsqrt = false; // rsqrt can be way faster, but less accurate (some compilers may insert an
                                   // additional newton raphson refinement)
constexpr auto runUpdate = true; // run update step. Useful to disable for benchmarking the move step.

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)                       \
    || (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_CPU))
#    define ANY_CPU_ENABLED 1
#endif
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)                                       \
    || (defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_ONEAPI_GPU))
#    define ANY_GPU_ENABLED 1
#endif

#if ANY_CPU_ENABLED && ANY_GPU_ENABLED
#    error Cannot enable CPU and GPU backends at the same time
#endif

#if ANY_CPU_ENABLED
constexpr auto threadsPerBlock = 1;
constexpr auto sharedElementsPerBlock = 1;
constexpr auto elementsPerThread = xsimd::batch<float>::size;
constexpr auto aosoaLanes = elementsPerThread;
#elif ANY_GPU_ENABLED
constexpr auto threadsPerBlock = 256;
constexpr auto sharedElementsPerBlock = 512;
constexpr auto elementsPerThread = 1;
constexpr auto aosoaLanes = 32; // coalesced memory access
#else
#    error "Unsupported backend"
#endif

// makes our life easier for now
static_assert(problemSize % sharedElementsPerBlock == 0);
static_assert(sharedElementsPerBlock % threadsPerBlock == 0);

constexpr auto timestep = FP{0.0001};
constexpr auto eps2 = FP{0.01};

constexpr auto rngSeed = 42;
constexpr auto referenceParticleIndex = 1338;

#ifdef HAVE_XSIMD
template<typename Batch>
struct llama::SimdTraits<Batch, std::enable_if_t<xsimd::is_batch<Batch>::value>>
{
    using value_type = typename Batch::value_type;

    inline static constexpr std::size_t lanes = Batch::size;

    static LLAMA_FORCE_INLINE auto loadUnaligned(const value_type* mem) -> Batch
    {
        return Batch::load_unaligned(mem);
    }

    static LLAMA_FORCE_INLINE void storeUnaligned(Batch batch, value_type* mem)
    {
        batch.store_unaligned(mem);
    }

    template<std::size_t... Is>
    static LLAMA_FORCE_INLINE auto indicesToReg(std::array<int, lanes> indices, std::index_sequence<Is...>)
    {
        return xsimd::batch<int, typename Batch::arch_type>(indices[Is]...);
    }

    static LLAMA_FORCE_INLINE auto gather(const value_type* mem, std::array<int, lanes> indices) -> Batch
    {
        return Batch::gather(mem, indicesToReg(indices, std::make_index_sequence<lanes>{}));
    }

    static LLAMA_FORCE_INLINE void scatter(Batch batch, value_type* mem, std::array<int, lanes> indices)
    {
        batch.scatter(mem, indicesToReg(indices, std::make_index_sequence<lanes>{}));
    }
};

template<typename T, std::size_t N>
struct MakeSizedBatchImpl
{
    using type = xsimd::make_sized_batch_t<T, N>;
    static_assert(!std::is_void_v<type>);
};

template<typename T, std::size_t N>
using MakeSizedBatch = typename MakeSizedBatchImpl<T, N>::type;
#endif

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
    struct Padding{};
} // namespace tag

using Vec3 = llama::Record<
    llama::Field<tag::X, FP>,
    llama::Field<tag::Y, FP>,
    llama::Field<tag::Z, FP>>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Vel, Vec3>,
    llama::Field<tag::Mass, FP>>;

using ParticleJ = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

// using SharedMemoryParticle = Particle;
using SharedMemoryParticle = ParticleJ;

enum Mapping
{
    AoS,
    SoA_SB,
    SoA_MB,
    AoSoA,
    SplitGpuGems
};

template<typename Acc, typename ParticleRefI, typename ParticleRefJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(const Acc& acc, ParticleRefI& pis, ParticleRefJ pj)
{
    auto dist = pis(tag::Pos{}) - pj(tag::Pos{});
    dist *= dist;
    const auto distSqr = +eps2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const auto distSixth = distSqr * distSqr * distSqr;
    const auto invDistCube
        = allowRsqrt ? alpaka::math::rsqrt(acc, distSixth) : (FP{1} / alpaka::math::sqrt(acc, distSixth));
    const auto sts = (pj(tag::Mass{}) * timestep) * invDistCube;
    pis(tag::Vel{}) += dist * sts;
}

template<int ThreadsPerBlock, int SharedElementsPerBlock, int ElementsPerThread, typename QuotedSMMapping>
struct UpdateKernel
{
    // TODO(bgruber): make this an IILE in C++20
    template<typename Mapping, typename Acc, std::size_t... Is>
    ALPAKA_FN_HOST_ACC auto makeSharedViewHelper(const Acc& acc, std::index_sequence<Is...>) const
    {
        return llama::View{
            Mapping{},
            llama::Array<std::byte*, sizeof...(Is)>{
                alpaka::declareSharedVar<std::byte[Mapping{}.blobSize(Is)], Is>(acc)...}};
    }

    template<typename Acc, typename View>
    ALPAKA_FN_HOST_ACC void operator()(const Acc& acc, View particles) const
    {
        auto sharedView = [&]
        {
            // if there is only 1 shared element per block, use just a variable (in registers) instead of shared memory
            if constexpr(SharedElementsPerBlock == 1)
            {
                using Mapping = llama::mapping::MinAlignedOne<llama::ArrayExtents<int, 1>, SharedMemoryParticle>;
                return allocViewUninitialized(Mapping{}, llama::bloballoc::Array<Mapping{}.blobSize(0)>{});
            }
            else
            {
                using Mapping = typename QuotedSMMapping::
                    template fn<llama::ArrayExtents<int, SharedElementsPerBlock>, SharedMemoryParticle>;
                return makeSharedViewHelper<Mapping>(acc, std::make_index_sequence<Mapping::blobCount>{});
            }
        }();

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        auto pis = llama::SimdN<Particle, ElementsPerThread, MakeSizedBatch>{};
        llama::loadSimd(particles(ti * ElementsPerThread), pis);

        for(int blockOffset = 0; blockOffset < problemSize; blockOffset += SharedElementsPerBlock)
        {
            for(int j = 0; j < SharedElementsPerBlock; j += ThreadsPerBlock)
                sharedView(j) = particles(blockOffset + tbi + j);
            alpaka::syncBlockThreads(acc);
            for(int j = 0; j < SharedElementsPerBlock; ++j)
                pPInteraction(acc, pis, sharedView(j));
            alpaka::syncBlockThreads(acc);
        }
        llama::storeSimd(pis(tag::Vel{}), particles(ti * ElementsPerThread)(tag::Vel{}));
    }
};

template<int ElementsPerThread>
struct MoveKernel
{
    template<typename Acc, typename View>
    ALPAKA_FN_HOST_ACC void operator()(const Acc& acc, View particles) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        const auto i = ti * ElementsPerThread;
        llama::SimdN<Vec3, ElementsPerThread, MakeSizedBatch> pos;
        llama::SimdN<Vec3, ElementsPerThread, MakeSizedBatch> vel;
        llama::loadSimd(particles(i)(tag::Pos{}), pos);
        llama::loadSimd(particles(i)(tag::Vel{}), vel);
        llama::storeSimd(pos + vel * +timestep, particles(i)(tag::Pos{}));
    }
};

template<typename Acc>
constexpr auto hasSharedMem
    = alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt, alpaka::TagGpuHipRt, alpaka::TagGpuSyclIntel>;

template<typename Acc, Mapping MappingGM, Mapping MappingSM>
void run(std::ostream& plotFile)
{
    using Dim = alpaka::Dim<Acc>;
    using Size = alpaka::Idx<Acc>;

    auto mappingName = [](int m) -> std::string
    {
        if(m == 0)
            return "AoS";
        if(m == 1)
            return "SoA SB";
        if(m == 2)
            return "SoA MB";
        if(m == 3)
            return "AoSoA" + std::to_string(aosoaLanes);
        if(m == 4)
            return "SplitGpuGems";
        std::abort();
    };
    const auto title = "GM " + mappingName(MappingGM) + (hasSharedMem<Acc> ? " SM " + mappingName(MappingSM) : "");
    std::cout << '\n' << title << '\n';

    const auto platformAcc = alpaka::Platform<Acc>{};
    const auto platformHost = alpaka::PlatformCpu{};
    const auto devAcc = alpaka::getDevByIdx(platformAcc, 0);
    const auto devHost = alpaka::getDevByIdx(platformHost, 0);
    auto queue = alpaka::Queue<Acc, alpaka::Blocking>{devAcc};

    auto mapping = []
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
        const auto extents = ArrayExtents{problemSize};
        if constexpr(MappingGM == AoS)
            return llama::mapping::AoS<ArrayExtents, Particle>{extents};
        if constexpr(MappingGM == SoA_SB)
            return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::Single>{extents};
        if constexpr(MappingGM == SoA_MB)
            return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::OnePerField>{extents};
        if constexpr(MappingGM == AoSoA)
            return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{extents};
        using boost::mp11::mp_list;
        if constexpr(MappingGM == SplitGpuGems)
        {
            using Vec4 = llama::Record<
                llama::Field<tag::X, FP>,
                llama::Field<tag::Y, FP>,
                llama::Field<tag::Z, FP>,
                llama::Field<tag::Padding, FP>>; // dummy
            using ParticlePadded = llama::
                Record<llama::Field<tag::Pos, Vec3>, llama::Field<tag::Vel, Vec4>, llama::Field<tag::Mass, FP>>;
            return llama::mapping::Split<
                ArrayExtents,
                ParticlePadded,
                mp_list<
                    mp_list<tag::Pos, tag::X>,
                    mp_list<tag::Pos, tag::Y>,
                    mp_list<tag::Pos, tag::Z>,
                    mp_list<tag::Mass>>,
                llama::mapping::BindAoS<>::fn,
                llama::mapping::BindAoS<>::fn,
                true>{extents};
        }
    }();

    [[maybe_unused]] auto selectedSMMapping = []
    {
        if constexpr(MappingSM == AoS)
            return llama::mapping::BindAoS{};
        if constexpr(MappingSM == SoA_SB)
            return llama::mapping::BindSoA<llama::mapping::Blobs::Single>{};
        if constexpr(MappingSM == SoA_MB)
            return llama::mapping::BindSoA<llama::mapping::Blobs::OnePerField>{};
        if constexpr(MappingSM == AoSoA)
            return llama::mapping::BindAoSoA<aosoaLanes>{};
    }();
    using QuotedMappingSM = decltype(selectedSMMapping);

    std::ofstream{"nbody_alpaka_mapping_" + mappingName(MappingGM) + ".svg"}
        << llama::toSvg(decltype(mapping){llama::ArrayExtentsDynamic<int, 1>{32}});

    Stopwatch watch;

    auto hostView
        = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devHost)>{devHost});
    auto accView = llama::allocViewUninitialized(mapping, llama::bloballoc::AlpakaBuf<Size, decltype(devAcc)>{devAcc});
    watch.printAndReset("alloc views");

    std::mt19937_64 engine{rngSeed};
    std::normal_distribution distribution(FP{0}, FP{1});
    for(int i = 0; i < problemSize; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(engine);
        p(tag::Pos(), tag::Y()) = distribution(engine);
        p(tag::Pos(), tag::Z()) = distribution(engine);
        p(tag::Vel(), tag::X()) = distribution(engine) / FP{10};
        p(tag::Vel(), tag::Y()) = distribution(engine) / FP{10};
        p(tag::Vel(), tag::Z()) = distribution(engine) / FP{10};
        p(tag::Mass()) = distribution(engine) / FP{100};
        hostView(i) = p;
    }
    watch.printAndReset("init");

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, accView.blobs()[i], hostView.blobs()[i]);
    watch.printAndReset("copy H->D");

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{static_cast<Size>(problemSize / (threadsPerBlock * elementsPerThread))},
        alpaka::Vec<Dim, Size>{static_cast<Size>(threadsPerBlock)},
        alpaka::Vec<Dim, Size>{static_cast<Size>(elementsPerThread)}};

    common::Stats statsUpdate;
    common::Stats statsMove;
    for(int s = 0; s < steps + 1; ++s)
    {
        if constexpr(runUpdate)
        {
            auto updateKernel
                = UpdateKernel<threadsPerBlock, sharedElementsPerBlock, elementsPerThread, QuotedMappingSM>{};
            alpaka::exec<Acc>(queue, workdiv, updateKernel, llama::shallowCopy(accView));
            statsUpdate(watch.printAndReset("update", '\t'));
        }

        auto moveKernel = MoveKernel<elementsPerThread>{};
        alpaka::exec<Acc>(queue, workdiv, moveKernel, llama::shallowCopy(accView));
        statsMove(watch.printAndReset("move"));
    }
    plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
             << statsMove.mean() << "\t" << statsMove.sem() << '\n';

    for(std::size_t i = 0; i < mapping.blobCount; i++)
        alpaka::memcpy(queue, hostView.blobs()[i], accView.blobs()[i]);
    watch.printAndReset("copy D->H");

    const auto [x, y, z] = hostView(referenceParticleIndex)(tag::Pos{});
    fmt::print("reference pos: {{{} {} {}}}\n", x, y, z);
}

auto main() -> int
try
{
    using Dim = alpaka::DimInt<1>;
    using Size = int;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;

    const auto env = common::captureEnv<Acc>();
    std::cout << problemSize / 1000 << "k particles (" << problemSize * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << threadsPerBlock << " particles (" << threadsPerBlock * llama::sizeOf<Particle> / 1024
              << " kiB) in shared memory\n"
              << "Reducing on " << elementsPerThread << " particles per thread\n"
              << "Using " << threadsPerBlock << " threads per block\n"
              << env << '\n';
    std::cout << std::fixed;

    std::ofstream plotFile{"nbody_alpaka.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# {}
set title "nbody alpaka {}ki particles on {}"
set style data histograms
set style histogram errorbars
set style fill solid border -1
set xtics rotate by 45 right nomirror
set key out top center maxrows 3
set yrange [0:*]
set y2range [0:*]
set ylabel "update runtime [s]"
set y2label "move runtime [s]"
set y2tics auto
$data << EOD
""	"update"	"update_sem"	"move"	"move_sem"
)",
        env,
        problemSize / 1024,
        alpaka::getAccName<Acc>());

    constexpr auto runSMVariations = hasSharedMem<Acc> && runUpdate;

    run<Acc, AoS, AoS>(plotFile);
    if constexpr(runSMVariations)
        run<Acc, AoS, SoA_SB>(plotFile);
    if constexpr(runSMVariations)
        run<Acc, AoS, AoSoA>(plotFile);
    run<Acc, SoA_MB, AoS>(plotFile);
    if constexpr(runSMVariations)
        run<Acc, SoA_MB, SoA_SB>(plotFile);
    if constexpr(runSMVariations)
        run<Acc, SoA_MB, AoSoA>(plotFile);
    run<Acc, AoSoA, AoS>(plotFile);
    if constexpr(runSMVariations)
        run<Acc, AoSoA, SoA_SB>(plotFile);
    if constexpr(runSMVariations)
        run<Acc, AoSoA, AoSoA>(plotFile);
    run<Acc, SplitGpuGems, AoS>(plotFile);

    plotFile << R"(EOD
plot $data using 2:3:xtic(1) ti col axis x1y1, "" using 4:5 ti col axis x1y2
)";
    std::cout << "Plot with: ./nbody_alpaka.sh\n";

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

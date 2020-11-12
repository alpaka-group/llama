/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../../common/Stopwatch.hpp"
#include "../../common/alpakaHelpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto MAPPING = 0; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blobs), 3 tree AoS, 4 tree SoA
constexpr auto USE_SHARED = true; ///< defines whether shared memory shall be used
constexpr auto USE_SHARED_TREE = true; ///< defines whether the shared memory shall use tree mapping or
                                       ///< native mapping

constexpr auto PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto BLOCK_SIZE = 256; ///< number of elements per block
constexpr auto STEPS = 5; ///< number of steps to calculate

using FP = float;
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
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>>>,
    llama::DE<tag::Vel, llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>>>,
    llama::DE<tag::Mass, FP>>;
// clang-format on

template <typename VirtualParticleI, typename VirtualParticleJ>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(VirtualParticleI pi, VirtualParticleJ pj, FP ts)
{
    auto dist = pi(tag::Pos()) - pj(tag::Pos());
    dist *= dist;
    const FP distSqr = EPS2 + dist(tag::X()) + dist(tag::Y()) + dist(tag::Z());
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP sts = pj(tag::Mass()) * invDistCube * ts;
    pi(tag::Vel()) += dist * sts;
}

template <std::size_t ProblemSize, std::size_t Elems, std::size_t BlockSize>
struct UpdateKernel
{
    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        [[maybe_unused]] auto sharedView = [&] {
            if constexpr (USE_SHARED)
            {
                const auto sharedMapping = [&] {
                    if constexpr (USE_SHARED_TREE)
                        return llama::mapping::tree::Mapping{
                            typename View::ArrayDomain{BlockSize},
                            llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                            typename View::DatumDomain{}};
                    else
                        return llama::mapping::SoA{typename View::ArrayDomain{BlockSize}, typename View::DatumDomain{}};
                }();

                // if there is only 1 thread per block, avoid using shared
                // memory
                if constexpr (BlockSize / Elems == 1)
                    return llama::allocViewStack<View::ArrayDomain::rank, typename View::DatumDomain>();
                else
                {
                    constexpr auto sharedMemSize = llama::sizeOf<typename View::DatumDomain> * BlockSize;
                    auto& sharedMem = alpaka::allocVar<std::byte[sharedMemSize], __COUNTER__>(acc);
                    return llama::View{sharedMapping, llama::Array{&sharedMem[0]}};
                }
            }
            else
                return int{}; // dummy
        }();

        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        const auto tbi = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);
        LLAMA_INDEPENDENT_DATA
        for (std::size_t b = 0; b < (ProblemSize + BlockSize - 1u) / BlockSize; ++b)
        {
            const auto start2 = b * BlockSize;
            const auto end2 = alpaka::math::min(acc, start2 + BlockSize, ProblemSize) - start2;
            if constexpr (USE_SHARED)
            {
                LLAMA_INDEPENDENT_DATA
                for (auto pos2 = decltype(end2)(0); pos2 + ti < end2; pos2 += BlockSize / Elems)
                    sharedView(pos2 + tbi) = particles(start2 + pos2 + tbi);
                alpaka::syncBlockThreads(acc);
            }
            LLAMA_INDEPENDENT_DATA
            for (auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2)
                LLAMA_INDEPENDENT_DATA
            for (auto i = start; i < end; ++i)
                if constexpr (USE_SHARED)
                    pPInteraction(particles(i), sharedView(pos2), ts);
                else
                    pPInteraction(particles(i), particles(start2 + pos2), ts);
            if constexpr (USE_SHARED)
                alpaka::syncBlockThreads(acc);
        }
    }
};

template <std::size_t ProblemSize, std::size_t Elems>
struct MoveKernel
{
    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View particles, FP ts) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);

        LLAMA_INDEPENDENT_DATA
        for (auto i = start; i < end; ++i)
            particles(i)(tag::Pos()) += particles(i)(tag::Vel()) * ts;
    }
};

int main()
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;

    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    // using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::AccCpuSerial<Dim, Size>;

    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;
    const DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    // NBODY
    constexpr std::size_t hardwareThreads = 2; // relevant for OpenMP2Threads
    using Distribution = common::ThreadsElemsDistribution<Acc, BLOCK_SIZE, hardwareThreads>;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    constexpr FP ts = 0.0001;

    // LLAMA
    const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};

    const auto mapping = [&] {
        if constexpr (MAPPING == 0)
            return llama::mapping::AoS{arrayDomain, Particle{}};
        if constexpr (MAPPING == 1)
            return llama::mapping::SoA{arrayDomain, Particle{}};
        if constexpr (MAPPING == 2)
            return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
        if constexpr (MAPPING == 3)
            return llama::mapping::tree::Mapping{arrayDomain, llama::Tuple{}, Particle{}};
        if constexpr (MAPPING == 4)
            return llama::mapping::tree::Mapping{
                arrayDomain,
                llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                Particle{}};
    }();

    std::cout << PROBLEM_SIZE / 1000 << " thousand particles\n"
              << PROBLEM_SIZE * llama::sizeOf<Particle> / 1000 / 1000 << "MB \n";

    Stopwatch chrono;

    const auto bufferSize = Size(mapping.getBlobSize(0));

    auto hostBuffer = alpaka::allocBuf<std::byte, Size>(devHost, bufferSize);
    auto accBuffer = alpaka::allocBuf<std::byte, Size>(devAcc, bufferSize);

    chrono.printAndReset("Alloc");

    auto hostView = llama::View{mapping, llama::Array{alpaka::getPtrNative(hostBuffer)}};
    auto accView = llama::View{mapping, llama::Array{alpaka::getPtrNative(accBuffer)}};

    chrono.printAndReset("Views");

    /// Random initialization of the particles
    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        auto temp = llama::allocVirtualDatumStack<Particle>();
        temp(tag::Pos(), tag::X()) = distribution(generator);
        temp(tag::Pos(), tag::Y()) = distribution(generator);
        temp(tag::Pos(), tag::Z()) = distribution(generator);
        temp(tag::Vel(), tag::X()) = distribution(generator) / FP(10);
        temp(tag::Vel(), tag::Y()) = distribution(generator) / FP(10);
        temp(tag::Vel(), tag::Z()) = distribution(generator) / FP(10);
        temp(tag::Mass()) = distribution(generator) / FP(100);
        hostView(i) = temp;
    }

    chrono.printAndReset("Init");

    alpaka::memcpy(queue, accBuffer, hostBuffer, bufferSize);
    chrono.printAndReset("Copy H->D");

    const alpaka::Vec<Dim, Size> Elems(static_cast<Size>(elemCount));
    const alpaka::Vec<Dim, Size> threads(static_cast<Size>(threadCount));
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::Vec<Dim, Size> blocks(static_cast<Size>((PROBLEM_SIZE + innerCount - 1u) / innerCount));

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{blocks, threads, Elems};

    for (std::size_t s = 0; s < STEPS; ++s)
    {
        UpdateKernel<PROBLEM_SIZE, elemCount, BLOCK_SIZE> updateKernel;
        alpaka::exec<Acc>(queue, workdiv, updateKernel, accView, ts);

        chrono.printAndReset("Update kernel");

        MoveKernel<PROBLEM_SIZE, elemCount> moveKernel;
        alpaka::exec<Acc>(queue, workdiv, moveKernel, accView, ts);
        chrono.printAndReset("Move kernel");
    }

    alpaka::memcpy(queue, hostBuffer, accBuffer, bufferSize);
    chrono.printAndReset("Copy D->H");

    return 0;
}

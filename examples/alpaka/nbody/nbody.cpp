/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../../common/Chrono.hpp"
#include "../../common/alpakaHelpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto MAPPING
    = 0; /// 0 native AoS, 1 native SoA, 2 tree AoS, 3 tree SoA
constexpr auto USE_SHARED
    = true; ///< defines whether shared memory shall be used
constexpr auto USE_SHARED_TREE
    = true; ///< defines whether the shared memory shall use tree mapping or
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

/// Helper function for particle particle interaction. Gets two virtual
/// datums like they are real particle objects
template<typename T_VirtualDatum1, typename T_VirtualDatum2>
LLAMA_FN_HOST_ACC_INLINE void
pPInteraction(T_VirtualDatum1 && p1, T_VirtualDatum2 && p2, const FP & ts)
{
    // Creating tempory virtual datum object for distance on stack:
    auto distance = p1(tag::Pos()) + p2(tag::Pos());
    distance *= distance; // square for each element
    const FP distSqr
        = EPS2 + distance(tag::X()) + distance(tag::Y()) + distance(tag::Z());
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP s = p2(tag::Mass()) * invDistCube;
    distance *= s * ts;
    p1(tag::Vel()) += distance;
}

/// Alpaka kernel for updating the speed of every particle based on the
/// distance and mass to each other particle. Has complexity O(N²).
template<std::size_t ProblemSize, std::size_t Elems, std::size_t BlockSize>
struct UpdateKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void
    operator()(const Acc & acc, View particles, FP ts) const
    {
        [[maybe_unused]] auto sharedView = [&] {
            if constexpr(USE_SHARED)
            {
                const auto sharedMapping = [&] {
                    if constexpr(USE_SHARED_TREE)
                    {
                        auto treeOperationList = llama::Tuple{
                            llama::mapping::tree::functor::LeafOnlyRT()};
                        using SharedMapping = llama::mapping::tree::Mapping<
                            typename View::Mapping::UserDomain,
                            typename View::Mapping::DatumDomain,
                            decltype(treeOperationList)>;
                        return SharedMapping({BlockSize}, treeOperationList);
                    }
                    else
                    {
                        using SharedMapping = llama::mapping::SoA<
                            typename View::Mapping::UserDomain,
                            typename View::Mapping::DatumDomain>;
                        return SharedMapping({BlockSize});
                    }
                }();
                using SharedMapping = decltype(sharedMapping);

                // if there is only 1 thread per block, avoid using shared
                // memory
                if constexpr(BlockSize / Elems == 1)
                    return llama::allocViewStack<
                        View::Mapping::UserDomain::count,
                        typename View::Mapping::DatumDomain>();
                else
                {
                    constexpr auto sharedMemSize
                        = llama::SizeOf<
                              typename View::Mapping::DatumDomain>::value
                        * BlockSize;
                    auto & sharedMem = alpaka::block::shared::st::
                        allocVar<std::byte[sharedMemSize], __COUNTER__>(acc);
                    return llama::View<SharedMapping, std::byte *>{
                        sharedMapping, {&sharedMem[0]}};
                }
            }
            else
                return int{}; // dummy
        }();

        const auto ti
            = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
        const auto tbi
            = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);
        LLAMA_INDEPENDENT_DATA
        for(std::size_t b = 0; b < (ProblemSize + BlockSize - 1u) / BlockSize;
            ++b)
        {
            const auto start2 = b * BlockSize;
            const auto end2
                = alpaka::math::min(acc, start2 + BlockSize, ProblemSize)
                - start2;
            if constexpr(USE_SHARED)
            {
                LLAMA_INDEPENDENT_DATA
                for(auto pos2 = decltype(end2)(0); pos2 + ti < end2;
                    pos2 += BlockSize / Elems)
                    sharedView(pos2 + tbi) = particles(start2 + pos2 + tbi);
                alpaka::block::sync::syncBlockThreads(acc);
            }
            LLAMA_INDEPENDENT_DATA
            for(auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2)
                LLAMA_INDEPENDENT_DATA
            for(auto i = start; i < end; ++i)
                if constexpr(USE_SHARED)
                    pPInteraction(particles(i), sharedView(pos2), ts);
                else
                    pPInteraction(particles(i), particles(start2 + pos2), ts);
            if constexpr(USE_SHARED)
                alpaka::block::sync::syncBlockThreads(acc);
        }
    }
};

/// Alpaka kernel for moving each particle with its speed. Has complexity
/// O(N).
template<std::size_t ProblemSize, std::size_t Elems>
struct MoveKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void
    operator()(const Acc & acc, View particles, FP ts) const
    {
        const auto ti
            = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);

        LLAMA_INDEPENDENT_DATA
        for(auto i = start; i < end; ++i)
            particles(i)(tag::Pos()) += particles(i)(tag::Vel()) * ts;
    }
};

int main(int argc, char ** argv)
{
    using Dim = alpaka::dim::DimInt<1>;
    using Size = std::size_t;

    using Acc = alpaka::example::ExampleDefaultAcc<Dim, Size>;
    // using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;

    using DevHost = alpaka::dev::DevCpu;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::Queue<DevAcc, alpaka::queue::Blocking>;
    const DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    // NBODY
    constexpr std::size_t hardwareThreads = 2; // relevant for OpenMP2Threads
    using Distribution
        = common::ThreadsElemsDistribution<Acc, BLOCK_SIZE, hardwareThreads>;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    constexpr FP ts = 0.0001;

    // LLAMA
    using UserDomain = llama::UserDomain<1>;
    const UserDomain userDomain{PROBLEM_SIZE};

    const auto mapping = [&] {
        if constexpr(MAPPING == 0)
        {
            using Mapping = llama::mapping::AoS<UserDomain, Particle>;
            return Mapping(userDomain);
        }
        if constexpr(MAPPING == 1)
        {
            using Mapping = llama::mapping::SoA<UserDomain, Particle>;
            return Mapping(userDomain);
        }
        if constexpr(MAPPING == 2)
        {
            auto treeOperationList = llama::Tuple{};
            using Mapping = llama::mapping::tree::
                Mapping<UserDomain, Particle, decltype(treeOperationList)>;
            return Mapping(userDomain, treeOperationList);
        }
        if constexpr(MAPPING == 3)
        {
            auto treeOperationList
                = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
            using Mapping = llama::mapping::tree::
                Mapping<UserDomain, Particle, decltype(treeOperationList)>;
            return Mapping(userDomain, treeOperationList);
        }
    }();
    using Mapping = decltype(mapping);

    std::cout << PROBLEM_SIZE / 1000 << " thousand particles\n"
              << PROBLEM_SIZE * llama::SizeOf<Particle>::value / 1000 / 1000
              << "MB \n";

    Chrono chrono;

    const auto bufferSize = Size(mapping.getBlobSize(0));

    auto hostBuffer
        = alpaka::mem::buf::alloc<std::byte, Size>(devHost, bufferSize);
    auto accBuffer
        = alpaka::mem::buf::alloc<std::byte, Size>(devAcc, bufferSize);

    chrono.printAndReset("Alloc");

    auto hostView = llama::View<Mapping, std::byte *>{
        mapping, {alpaka::mem::view::getPtrNative(hostBuffer)}};
    auto accView = llama::View<Mapping, std::byte *>{
        mapping, {alpaka::mem::view::getPtrNative(accBuffer)}};

    chrono.printAndReset("Views");

    /// Random initialization of the particles
    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    LLAMA_INDEPENDENT_DATA
    for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        auto temp = llama::stackVirtualDatumAlloc<Particle>();
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

    alpaka::mem::view::copy(queue, accBuffer, hostBuffer, bufferSize);
    chrono.printAndReset("Copy H->D");

    const alpaka::vec::Vec<Dim, Size> Elems(static_cast<Size>(elemCount));
    const alpaka::vec::Vec<Dim, Size> threads(static_cast<Size>(threadCount));
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::vec::Vec<Dim, Size> blocks(
        static_cast<Size>((PROBLEM_SIZE + innerCount - 1u) / innerCount));

    const auto workdiv
        = alpaka::workdiv::WorkDivMembers<Dim, Size>{blocks, threads, Elems};

    for(std::size_t s = 0; s < STEPS; ++s)
    {
        UpdateKernel<PROBLEM_SIZE, elemCount, BLOCK_SIZE> updateKernel;
        alpaka::kernel::exec<Acc>(queue, workdiv, updateKernel, accView, ts);

        chrono.printAndReset("Update kernel");

        MoveKernel<PROBLEM_SIZE, elemCount> moveKernel;
        alpaka::kernel::exec<Acc>(queue, workdiv, moveKernel, accView, ts);
        chrono.printAndReset("Move kernel");
    }

    alpaka::mem::view::copy(queue, hostBuffer, accBuffer, bufferSize);
    chrono.printAndReset("Copy D->H");

    return 0;
}

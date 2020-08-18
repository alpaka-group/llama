/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file nbody.cpp
 *  \brief Realistic nbody example for using LLAMA and ALPAKA together.
 */

#ifdef __CUDACC__
#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#endif
#include "../../common/AlpakaAllocator.hpp"
#include "../../common/AlpakaMemCopy.hpp"
#include "../../common/AlpakaThreadElemsDistribution.hpp"
#include "../../common/Chrono.hpp"
#include "../../common/Dummy.hpp"

#include <alpaka/alpaka.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto NBODY_USE_TREE
    = true; ///< defines whether tree mapping or native mapping shall be used
constexpr auto NBODY_USE_SHARED
    = true; ///< defines whether shared memory shall be used
constexpr auto NBODY_USE_SHARED_TREE
    = true; ///< defines whether the shared memory shall use tree mapping or
            ///< native mapping

constexpr auto NBODY_PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto NBODY_BLOCK_SIZE = 256; ///< number of elements per block
constexpr auto NBODY_STEPS = 5; ///< number of steps to calculate

namespace nbody
{
    using Element = float;
    constexpr Element EPS2 = 0.01;

    // clang-format off
    namespace dd
    {
        struct Pos{};
        struct Vel{};
        struct X{};
        struct Y{};
        struct Z{};
        struct Mass{};
    }

    // clang-format off
    using Particle = llama::DS<
        llama::DE<dd::Pos, llama::DS<
            llama::DE<dd::X, Element>,
            llama::DE<dd::Y, Element>,
            llama::DE<dd::Z, Element>>>,
        llama::DE<dd::Vel, llama::DS<
            llama::DE<dd::X, Element>,
            llama::DE<dd::Y, Element>,
            llama::DE<dd::Z, Element>>>,
        llama::DE<dd::Mass, Element>>;
    // clang-format on

    /// Helper function for particle particle interaction. Gets two virtual
    /// datums like they are real particle objects
    template<typename T_VirtualDatum1, typename T_VirtualDatum2>
    LLAMA_FN_HOST_ACC_INLINE void pPInteraction(
        T_VirtualDatum1 && p1,
        T_VirtualDatum2 && p2,
        const Element & ts)
    {
        // Creating tempory virtual datum object for distance on stack:
        auto distance = p1(dd::Pos()) + p2(dd::Pos());
        distance *= distance; // square for each element
        const Element distSqr
            = EPS2 + distance(dd::X()) + distance(dd::Y()) + distance(dd::Z());
        const Element distSixth = distSqr * distSqr * distSqr;
        const Element invDistCube = 1.0f / std::sqrt(distSixth);
        const Element s = p2(dd::Mass()) * invDistCube;
        distance *= s * ts;
        p1(dd::Vel()) += distance;
    }

    /// Implements an allocator for LLAMA using the ALPAKA shared memory or just
    /// local stack memory depending on the number of threads per block. If only
    /// one thread exists per block, the expensive share memory allocation can
    /// be avoided
    template<
        typename T_Acc,
        std::size_t T_size,
        std::size_t T_counter,
        std::size_t threads>
    struct BlockSharedMemoryAllocator
    {
        using type = common::allocator::AlpakaShared<T_Acc, T_size, T_counter>;

        template<typename T_Factory, typename T_Mapping>
        LLAMA_FN_HOST_ACC_INLINE static auto
        allocView(const T_Mapping mapping, const T_Acc & acc)
            -> decltype(T_Factory::allocView(mapping, acc))
        {
            return T_Factory::allocView(mapping, acc);
        }
    };

    template<typename T_Acc, std::size_t T_size, std::size_t T_counter>
    struct BlockSharedMemoryAllocator<T_Acc, T_size, T_counter, 1>
    {
        using type = llama::allocator::Stack<T_size>;

        template<typename T_Factory, typename T_Mapping>
        LLAMA_FN_HOST_ACC_INLINE static auto
        allocView(const T_Mapping mapping, const T_Acc & acc)
            -> decltype(T_Factory::allocView(mapping))
        {
            return T_Factory::allocView(mapping);
        }
    };

    /// Alpaka kernel for updating the speed of every particle based on the
    /// distance and mass to each other particle. Has complexity O(N²).
    template<std::size_t problemSize, std::size_t elems, std::size_t blockSize>
    struct UpdateKernel
    {
        template<typename T_Acc, typename T_View>
        LLAMA_FN_HOST_ACC_INLINE void
        operator()(const T_Acc & acc, T_View particles, Element ts) const
        {
            constexpr std::size_t threads = blockSize / elems;
            const auto threadBlockIndex
                = alpaka::idx::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
            [[maybe_unused]] auto temp = [&] {
                if constexpr(NBODY_USE_SHARED)
                {
                    const auto sharedMapping = [&] {
                        if constexpr(NBODY_USE_SHARED_TREE)
                        {
                            auto treeOperationList = llama::Tuple{
                                llama::mapping::tree::functor::LeafOnlyRT()};
                            using SharedMapping = llama::mapping::tree::Mapping<
                                typename decltype(
                                    particles)::Mapping::UserDomain,
                                typename decltype(
                                    particles)::Mapping::DatumDomain,
                                decltype(treeOperationList)>;
                            return SharedMapping(
                                {blockSize}, treeOperationList);
                        }
                        else
                        {
                            using SharedMapping = llama::mapping::SoA<
                                typename decltype(
                                    particles)::Mapping::UserDomain,
                                typename decltype(
                                    particles)::Mapping::DatumDomain>;
                            return SharedMapping({blockSize});
                        }
                    }();
                    using SharedMapping = decltype(sharedMapping);

                    using SharedAllocator = BlockSharedMemoryAllocator<
                        T_Acc,
                        llama::SizeOf<typename decltype(
                            particles)::Mapping::DatumDomain>::value
                            * blockSize,
                        __COUNTER__,
                        threads>;
                    using SharedFactory = llama::
                        Factory<SharedMapping, typename SharedAllocator::type>;

                    return SharedAllocator::
                        template allocView<SharedFactory, SharedMapping>(
                            sharedMapping, acc);
                }
                else
                    return int{}; // dummy
            }();

            auto threadIndex
                = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

            const auto start = threadIndex * elems;
            const auto end = alpaka::math::min(acc, start + elems, problemSize);
            LLAMA_INDEPENDENT_DATA
            for(std::size_t b = 0;
                b < (problemSize + blockSize - 1u) / blockSize;
                ++b)
            {
                const auto start2 = b * blockSize;
                const auto end2
                    = alpaka::math::min(acc, start2 + blockSize, problemSize)
                    - start2;
                if constexpr(NBODY_USE_SHARED)
                {
                    LLAMA_INDEPENDENT_DATA
                    for(auto pos2 = decltype(end2)(0);
                        pos2 + threadIndex < end2;
                        pos2 += threads)
                        temp(pos2 + threadBlockIndex)
                            = particles(start2 + pos2 + threadBlockIndex);
                    alpaka::block::sync::syncBlockThreads(acc);
                }
                LLAMA_INDEPENDENT_DATA
                for(auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2)
                    LLAMA_INDEPENDENT_DATA
                for(auto pos = start; pos < end; ++pos)
                    if constexpr(NBODY_USE_SHARED)
                        pPInteraction(particles(pos), temp(pos2), ts);
                    else
                        pPInteraction(
                            particles(pos), particles(start2 + pos2), ts);
                if constexpr(NBODY_USE_SHARED)
                    alpaka::block::sync::syncBlockThreads(acc);
            }
        }
    };

    /// Alpaka kernel for moving each particle with its speed. Has complexity
    /// O(N).
    template<std::size_t problemSize, std::size_t elems>
    struct MoveKernel
    {
        template<typename T_Acc, typename T_View>
        LLAMA_FN_HOST_ACC_INLINE void
        operator()(const T_Acc & acc, T_View particles, Element ts) const
        {
            auto threadIndex
                = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

            const auto start = threadIndex * elems;
            const auto end = alpaka::math::min(
                acc, (threadIndex + 1) * elems, problemSize);

            LLAMA_INDEPENDENT_DATA
            for(auto pos = start; pos < end; ++pos)
                particles(pos)(dd::Pos()) += particles(pos)(dd::Vel()) * ts;
        }
    };

    int main(int argc, char ** argv)
    {
        using Dim = alpaka::dim::DimInt<1>;
        using Size = std::size_t;
        using Host = alpaka::acc::AccCpuSerial<Dim, Size>;

        // using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
        using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
        // using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
        // using Acc = alpaka::acc::AccCpuOmp2Threads< Dim, Size >;
        // using Acc = alpaka::acc::AccCpuOmp4< Dim, Size >;

        using DevHost = alpaka::dev::Dev<Host>;
        using DevAcc = alpaka::dev::Dev<Acc>;
        using PltfHost = alpaka::pltf::Pltf<DevHost>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
        using Queue = alpaka::queue::Queue<DevAcc, alpaka::queue::Blocking>;
        const DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
        const DevHost devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
        Queue queue(devAcc);

        // NBODY
        constexpr std::size_t problemSize = NBODY_PROBLEM_SIZE;
        constexpr std::size_t blockSize = NBODY_BLOCK_SIZE;
        constexpr std::size_t hardwareThreads
            = 2; // relevant for OpenMP2Threads
        using Distribution
            = common::ThreadsElemsDistribution<Acc, blockSize, hardwareThreads>;
        constexpr std::size_t elemCount = Distribution::elemCount;
        constexpr std::size_t threadCount = Distribution::threadCount;
        constexpr Element ts = 0.0001;
        constexpr std::size_t steps = NBODY_STEPS;

        // LLAMA
        using UserDomain = llama::UserDomain<1>;
        const UserDomain userDomainSize{problemSize};

        const auto mapping = [&] {
            if constexpr(NBODY_USE_TREE)
            {
                auto treeOperationList
                    = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
                using Mapping = llama::mapping::tree::
                    Mapping<UserDomain, Particle, decltype(treeOperationList)>;
                return Mapping(userDomainSize, treeOperationList);
            }
            else
            {
                using Mapping = llama::mapping::SoA<UserDomain, Particle>;
                return Mapping(userDomainSize);
            }
        }();
        using Mapping = decltype(mapping);
        using DevFactory
            = llama::Factory<Mapping, common::allocator::Alpaka<DevAcc, Size>>;
        using MirrorFactory = llama::Factory<
            Mapping,
            common::allocator::AlpakaMirror<DevAcc, Size, Mapping>>;
        using HostFactory
            = llama::Factory<Mapping, common::allocator::Alpaka<DevHost, Size>>;

        std::cout << problemSize / 1000 << " thousand particles\n"
                  << problemSize * llama::SizeOf<Particle>::value / 1000 / 1000
                  << "MB \n";

        Chrono chrono;

        auto hostView = HostFactory::allocView(mapping, devHost);
        auto accView = DevFactory::allocView(mapping, devAcc);
        auto mirrorView = MirrorFactory::allocView(mapping, accView);

        chrono.printAndReset("Alloc");

        /// Random initialization of the particles
        std::mt19937_64 generator;
        std::normal_distribution<Element> distribution(Element(0), Element(1));
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            auto temp = llama::stackVirtualDatumAlloc<Particle>();
            temp(dd::Pos(), dd::X()) = distribution(generator);
            temp(dd::Pos(), dd::Y()) = distribution(generator);
            temp(dd::Pos(), dd::Z()) = distribution(generator);
            temp(dd::Vel(), dd::X()) = distribution(generator) / Element(10);
            temp(dd::Vel(), dd::Y()) = distribution(generator) / Element(10);
            temp(dd::Vel(), dd::Z()) = distribution(generator) / Element(10);
            temp(dd::Mass()) = distribution(generator) / Element(100);
            hostView(i) = temp;
        }

        chrono.printAndReset("Init");

        alpakaMemCopy(accView, hostView, userDomainSize, queue);
        chrono.printAndReset("Copy H->D");

        const alpaka::vec::Vec<Dim, Size> elems(static_cast<Size>(elemCount));
        const alpaka::vec::Vec<Dim, Size> threads(
            static_cast<Size>(threadCount));
        constexpr auto innerCount = elemCount * threadCount;
        const alpaka::vec::Vec<Dim, Size> blocks(
            static_cast<Size>((problemSize + innerCount - 1u) / innerCount));

        const auto workdiv = alpaka::workdiv::WorkDivMembers<Dim, Size>{
            blocks, threads, elems};

        for(std::size_t s = 0; s < steps; ++s)
        {
            UpdateKernel<problemSize, elemCount, blockSize> updateKernel;
            alpaka::kernel::exec<Acc>(
                queue, workdiv, updateKernel, mirrorView, ts);

            chrono.printAndReset("Update kernel");

            MoveKernel<problemSize, elemCount> moveKernel;
            alpaka::kernel::exec<Acc>(
                queue, workdiv, moveKernel, mirrorView, ts);
            chrono.printAndReset("Move kernel");
            dummy(static_cast<void *>(mirrorView.blob[0]));
        }

        alpakaMemCopy(hostView, accView, userDomainSize, queue);
        chrono.printAndReset("Copy D->H");

        return 0;
    }
}

int main(int argc, char ** argv)
{
    return nbody::main(argc, argv);
}

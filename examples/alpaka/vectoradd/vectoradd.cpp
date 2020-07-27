/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file vectoradd.cpp
 *  \brief Vector add example for LLAMA using the ALPAKA library
 */

#include <iostream>
#include <utility>

/// Defines whether CUDA should be used in this example or not
#define VECTORADD_CUDA 0
/// -1 native AoS, 0 native SoA, 1 tree AoS, 2 tree SoA
#define VECTORADD_USE_TREE 2
/// For testing purposes, llama can be bypassed
#define VECTORADD_BYPASS_LLAMA 0
/// Which data layout for bypassing: 0 SoA, 1 AoS
#define VECTORADD_BYPASS_USE_SOA 1

/// problem size
#define VECTORADD_PROBLEM_SIZE 64 * 1024 * 1024
/// total number of elements per block
#define VECTORADD_BLOCK_SIZE 256
/// number of vector adds to perform
#define VECTORADD_STEPS 10

#include <alpaka/alpaka.hpp>
#ifdef __CUDACC__
/// if cuda is used, the function headers need some annotations
#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#endif
#include "../../common/AlpakaAllocator.hpp"
#include "../../common/AlpakaMemCopy.hpp"
#include "../../common/AlpakaThreadElemsDistribution.hpp"
#include "../../common/Chrono.hpp"
#include "../../common/Dummy.hpp"

#include <llama/llama.hpp>
#include <random>

namespace vectoradd
{
    using Element = float;

    namespace dd
    {
        struct X
        {};
        struct Y
        {};
        struct Z
        {};
    }

    using Vector = llama::DS<
        llama::DE<dd::X, Element>,
        llama::DE<dd::Y, Element>,
        llama::DE<dd::Z, Element>>;

    template<std::size_t problemSize, std::size_t elems>
    struct AddKernel
    {
        template<typename T_Acc, typename T_View>
        LLAMA_FN_HOST_ACC_INLINE void operator()(
            T_Acc const & acc,
            T_View a,
            T_View b
#if VECTORADD_BYPASS_LLAMA == 1
            ,
            std::size_t const aSizeAsRuntimeParameter,
            std::size_t const bSizeAsRuntimeParameter
#endif
        ) const
        {
            auto threadIndex
                = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];

            auto const start = threadIndex * elems;
            auto const end = alpaka::math::min(
                acc, (threadIndex + 1) * elems, problemSize);

            LLAMA_INDEPENDENT_DATA
            for(auto pos = start; pos < end; ++pos)
#if VECTORADD_BYPASS_LLAMA == 1
                for(auto dd = 0; dd < 3; ++dd)
#if VECTORADD_BYPASS_USE_SOA == 1
                    a[pos + dd * aSizeAsRuntimeParameter]
                        += b[pos + dd * bSizeAsRuntimeParameter];
#else
                    a[pos * 3 + dd] += b[pos * 3 + dd];
#endif // VECTORADD_BYPASS_USE_SOA
#else
                a(pos) += b(pos);
#endif // VECTORADD_BYPASS_LLAMA
        }
    };

    int main(int argc, char ** argv)
    {
        // ALPAKA
        using Dim = alpaka::dim::DimInt<1>;
        using Size = std::size_t;
        using Host = alpaka::acc::AccCpuSerial<Dim, Size>;

#if VECTORADD_CUDA == 1
        using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
#else
        //~ using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
        using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
        //~ using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
        //~ using Acc = alpaka::acc::AccCpuOmp4<Dim, Size>;
#endif // VECTORADD_CUDA
        using DevHost = alpaka::dev::Dev<Host>;
        using DevAcc = alpaka::dev::Dev<Acc>;
        using PltfHost = alpaka::pltf::Pltf<DevHost>;
        using PltfAcc = alpaka::pltf::Pltf<DevAcc>;

        using QueueProperty = alpaka::queue::Blocking;
        using Queue = alpaka::queue::Queue<Acc, QueueProperty>;

        DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
        DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
        Queue queue(devAcc);

        // VECTORADD
        constexpr std::size_t problemSize = VECTORADD_PROBLEM_SIZE;
        constexpr std::size_t blockSize = VECTORADD_BLOCK_SIZE;
        constexpr std::size_t hardwareThreads = 2; // relevant for
                                                   // OpenMP2Threads
        using Distribution
            = common::ThreadsElemsDistribution<Acc, blockSize, hardwareThreads>;
        constexpr std::size_t elemCount = Distribution::elemCount;
        constexpr std::size_t threadCount = Distribution::threadCount;
        constexpr std::size_t steps = VECTORADD_STEPS;

        // LLAMA
        using UserDomain = llama::UserDomain<1>;
        const UserDomain userDomainSize{problemSize};

#if VECTORADD_USE_TREE >= 1
#if VECTORADD_USE_TREE == 1
        auto treeOperationList
            = llama::makeTuple(llama::mapping::tree::functor::Idem());
#elif VECTORADD_USE_TREE == 2
        auto treeOperationList
            = llama::makeTuple(llama::mapping::tree::functor::LeafOnlyRT());
#endif
        using Mapping = llama::mapping::tree::
            Mapping<UserDomain, Vector, decltype(treeOperationList)>;
        const Mapping mapping(userDomainSize, treeOperationList);
#else // VECTORADD_USE_TREE
#if VECTORADD_USE_TREE == -1
        using Mapping = llama::mapping::AoS<UserDomain, Vector>;
#elif VECTORADD_USE_TREE == 0
        using Mapping = llama::mapping::SoA<UserDomain, Vector>;
#endif
        Mapping const mapping(userDomainSize);
#endif // VECTORADD_USE_TREE

        using DevFactory
            = llama::Factory<Mapping, common::allocator::Alpaka<DevAcc, Size>>;
        using MirrorFactory = llama::Factory<
            Mapping,
            common::allocator::AlpakaMirror<DevAcc, Size, Mapping>>;
        using HostFactory
            = llama::Factory<Mapping, common::allocator::Alpaka<DevHost, Size>>;

        std::cout << problemSize / 1000 / 1000 << " million vectors\n";
        std::cout << problemSize * llama::SizeOf<Vector>::value * 2 / 1000
                / 1000
                  << " MB on device\n";

        Chrono chrono;

        auto hostA = HostFactory::allocView(mapping, devHost);
        auto hostB = HostFactory::allocView(mapping, devHost);
        auto devA = DevFactory::allocView(mapping, devAcc);
        auto devB = DevFactory::allocView(mapping, devAcc);
        auto mirrorA = MirrorFactory::allocView(mapping, devA);
        auto mirrorB = MirrorFactory::allocView(mapping, devB);

        chrono.printAndReset("Alloc");

        std::default_random_engine generator;
        std::normal_distribution<Element> distribution(
            Element(0), // mean
            Element(1) // stddev
        );
        auto seed = distribution(generator);
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            hostA(i) = seed + i;
            hostB(i) = seed - i;
        }
        chrono.printAndReset("Init");

        alpakaMemCopy(devA, hostA, userDomainSize, queue);
        alpakaMemCopy(devB, hostB, userDomainSize, queue);

        chrono.printAndReset("Copy H->D");

        const alpaka::vec::Vec<Dim, Size> elems(static_cast<Size>(elemCount));
        const alpaka::vec::Vec<Dim, Size> threads(
            static_cast<Size>(threadCount));
        constexpr auto innerCount = elemCount * threadCount;
        const alpaka::vec::Vec<Dim, Size> blocks(
            static_cast<Size>((problemSize + innerCount - 1) / innerCount));

        auto const workdiv = alpaka::workdiv::WorkDivMembers<Dim, Size>{
            blocks, threads, elems};

        for(std::size_t s = 0; s < steps; ++s)
        {
            AddKernel<problemSize, elemCount> addKernel;
            alpaka::kernel::exec<Acc>(
                queue,
                workdiv,
                addKernel,
#if VECTORADD_BYPASS_LLAMA == 1
                reinterpret_cast<Element *>(mirrorA.blob[0]),
                reinterpret_cast<Element *>(mirrorB.blob[0]),
                problemSize,
                problemSize
#else
                mirrorA,
                mirrorB
#endif // VECTORADD_BYPASS_LLAMA
            );
            chrono.printAndReset("Add kernel");
            dummy(static_cast<void *>(mirrorA.blob[0]));
            dummy(static_cast<void *>(mirrorB.blob[0]));
        }

        alpakaMemCopy(hostA, devA, userDomainSize, queue);
        alpakaMemCopy(hostB, devB, userDomainSize, queue);

        chrono.printAndReset("Copy D->H");

        return 0;
    }

} // namespace vectoradd

int main(int argc, char ** argv)
{
    return vectoradd::main(argc, argv);
}

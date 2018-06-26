#include <iostream>
#include <utility>

#define VECTORADD_CUDA 0
// -1 native AoS, 0 native SoA, 1 tree AoS, 2 tree SoA
#define VECTORADD_USE_TREE 2
#define VECTORADD_BYPASS_LLAMA 0
// 0 SoA, 1 AoS
#define VECTORADD_BYPASS_SOA 1

#define VECTORADD_PROBLEM_SIZE 64*1024*1024
#define VECTORADD_BLOCK_SIZE 256
#define VECTORADD_MIN_ELEM 2
#define VECTORADD_STEPS 10

#include <alpaka/alpaka.hpp>
#ifdef __CUDACC__
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
#else
	#define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC inline
#endif
#include <llama/llama.hpp>
#include <random>

#include "../common/AlpakaAllocator.hpp"
#include "../common/Chrono.hpp"
#include "../common/Dummy.hpp"

using Element = float;

namespace dd
{
    struct X {};
    struct Y {};
    struct Z {};
}

using Vector = llama::DS<
    llama::DE< dd::X, Element >,
    llama::DE< dd::Y, Element >,
    llama::DE< dd::Z, Element >
>;

template<
    std::size_t problemSize,
    std::size_t elems
>
struct AddKernel
{
    template<
        typename T_Acc,
        typename T_View
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const &acc,
        T_View a,
        T_View b
#if VECTORADD_BYPASS_LLAMA == 1
        ,std::size_t const aSizeAsRuntimeParameter
        ,std::size_t const bSizeAsRuntimeParameter
#endif
    ) const
    {
        auto threadIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            (threadIndex + 1) * elems,
            problemSize
        );

        LLAMA_INDEPENDENT_DATA
        for ( auto pos = start; pos < end; ++pos )
#if VECTORADD_BYPASS_LLAMA == 1
            for ( auto dd = 0; dd < 3; ++dd )
#if VECTORADD_BYPASS_SOA == 1
                a[ pos + dd * aSizeAsRuntimeParameter ] +=
                    b[ pos + dd * bSizeAsRuntimeParameter ];
#else
                a[ pos * 3 + dd ] += b[ pos * 3 + dd ];
#endif // VECTORADD_BYPASS_SOA
#else
            a( pos ) += b( pos );
#endif // VECTORADD_BYPASS_LLAMA
    }
};

template<
    typename T_Acc,
    std::size_t blockSize,
    std::size_t hardwareThreads
>
struct ThreadsElemsDistribution
{
    static constexpr std::size_t elemCount = blockSize;
    static constexpr std::size_t threadCount = 1u;
};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size
    >
    struct ThreadsElemsDistribution<
        alpaka::acc::AccGpuCudaRt<T_Dim, T_Size>,
        blockSize,
        hardwareThreads
    >
    {
        static constexpr std::size_t elemCount = VECTORADD_MIN_ELEM;
        static constexpr std::size_t threadCount = blockSize / VECTORADD_MIN_ELEM;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size
    >
    struct ThreadsElemsDistribution<
        alpaka::acc::AccCpuOmp2Threads<T_Dim, T_Size>,
        blockSize,
        hardwareThreads
    >
    {
        static constexpr std::size_t elemCount =
            ( blockSize + hardwareThreads - 1u ) / hardwareThreads;
        static constexpr std::size_t threadCount = hardwareThreads;
    };
#endif


int main(int argc,char * * argv)
{
    // ALPAKA
    using Dim = alpaka::dim::DimInt< 1 >;
    using Size = std::size_t;
    using Extents = Size;
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
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
#if VECTORADD_CUDA == 1
    using Queue = alpaka::queue::QueueCudaRtSync;
#else
    using Queue = alpaka::queue::QueueCpuSync;
#endif // VECTORADD_CUDA
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) );
    DevHost const devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );
    Queue queue( devAcc ) ;

    // VECTORADD
    constexpr std::size_t problemSize = VECTORADD_PROBLEM_SIZE;
    constexpr std::size_t blockSize = VECTORADD_BLOCK_SIZE;
    constexpr std::size_t hardwareThreads = 2; //relevant for OpenMP2Threads
    using Distribution = ThreadsElemsDistribution<
        Acc,
        blockSize,
        hardwareThreads
    >;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    constexpr std::size_t steps = VECTORADD_STEPS;

    // LLAMA
    using UserDomain = llama::UserDomain< 1 >;
    const UserDomain userDomainSize{ problemSize };

#if VECTORADD_USE_TREE >= 1
#if VECTORADD_USE_TREE == 1
    auto treeOperationList = llama::makeTuple(
        llama::mapping::tree::functor::Idem( )
    );
#elif VECTORADD_USE_TREE == 2
    auto treeOperationList = llama::makeTuple(
        llama::mapping::tree::functor::LeaveOnlyRT( )
    );
#endif
    using Mapping = llama::mapping::tree::Mapping<
        UserDomain,
        Vector,
        decltype( treeOperationList )
    >;
    const Mapping mapping(
        userDomainSize,
        treeOperationList
    );
#else // VECTORADD_USE_TREE
#if VECTORADD_USE_TREE == -1
    using Mapping = llama::mapping::AoS<
        UserDomain,
        Vector
    >;
#elif VECTORADD_USE_TREE == 0
    using Mapping = llama::mapping::SoA<
        UserDomain,
        Vector
    >;
#endif
    Mapping const mapping( userDomainSize );
#endif // VECTORADD_USE_TREE

    using DevFactory = llama::Factory<
        Mapping,
        nbody::allocator::Alpaka<
            DevAcc,
            Dim,
            Size
        >
    >;
    using MirrorFactory = llama::Factory<
        Mapping,
        nbody::allocator::AlpakaMirror<
            DevAcc,
            Dim,
            Size,
            Mapping
        >
    >;
    using HostFactory = llama::Factory<
        Mapping,
        nbody::allocator::Alpaka<
            DevHost,
            Dim,
            Size
        >
    >;

    std::cout << problemSize / 1000 / 1000 << " million vectors\n";
    std::cout
        << problemSize * llama::SizeOf<Vector>::value * 2 / 1000 / 1000
        << " MB on device\n";

    Chrono chrono;

    auto   hostA =   HostFactory::allocView( mapping, devHost );
    auto   hostB =   HostFactory::allocView( mapping, devHost );
    auto    devA =    DevFactory::allocView( mapping,  devAcc );
    auto    devB =    DevFactory::allocView( mapping,  devAcc );
    auto mirrorA = MirrorFactory::allocView( mapping,    devA );
    auto mirrorB = MirrorFactory::allocView( mapping,    devB );

    chrono.printAndReset("Alloc");

    std::default_random_engine generator;
    std::normal_distribution< Element > distribution(
        Element( 0 ), // mean
        Element( 1 )  // stddev
    );
    auto seed = distribution(generator);
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < problemSize; ++i)
    {
        hostA(i) = seed + i;
        hostB(i) = seed - i;
    }
    chrono.printAndReset("Init");

    const alpaka::vec::Vec< Dim, Size > elems (
        static_cast< Size >( elemCount )
    );
    const alpaka::vec::Vec< Dim, Size > threads (
        static_cast< Size >( threadCount )
    );
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::vec::Vec< Dim, Size > blocks (
        static_cast< Size >( ( problemSize + innerCount - 1 ) / innerCount )
    );

    auto const workdiv = alpaka::workdiv::WorkDivMembers<
        Dim,
        Size
    > {
        blocks,
        threads,
        elems
    };

    for ( std::size_t s = 0; s < steps; ++s)
    {
        AddKernel<
            problemSize,
            elemCount
        > addKernel;
        alpaka::kernel::exec<Acc>(
            queue,
            workdiv,
            addKernel,
#if VECTORADD_BYPASS_LLAMA == 1
            reinterpret_cast<Element*>(mirrorA.blob[0]),
            reinterpret_cast<Element*>(mirrorB.blob[0]),
            problemSize,
            problemSize
#else
            mirrorA,
            mirrorB
#endif // VECTORADD_BYPASS_LLAMA
        );
        chrono.printAndReset("Add kernel");
        dummy( static_cast<void*>( mirrorA.blob[0] ) );
        dummy( static_cast<void*>( mirrorB.blob[0] ) );
    }


    return 0;
}

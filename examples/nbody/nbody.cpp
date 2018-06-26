#include <iostream>
#include <utility>

#define NBODY_CUDA 0
#define NBODY_USE_TREE 1
#define NBODY_USE_SHARED 1
#define NBODY_USE_SHARED_TREE 1

#define NBODY_PROBLEM_SIZE 16*1024
#define NBODY_BLOCK_SIZE 256
#define NBODY_STEPS 5


#if BOOST_VERSION < 106700 && (__CUDACC__ || __IBMCPP__)
    #ifdef BOOST_PP_VARIADICS
        #undef BOOST_PP_VARIADICS
    #endif
    #define BOOST_PP_VARIADICS 1
#endif

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
constexpr Element EPS2 = 0.01;

namespace dd
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Mass {};
}

using Particle = llama::DS<
    llama::DE< dd::Pos, llama::DS<
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
        llama::DE< dd::Z, Element >
    > >,
    llama::DE< dd::Vel,llama::DS<
        llama::DE< dd::X, Element >,
        llama::DE< dd::Y, Element >,
        llama::DE< dd::Z, Element >
    > >,
    llama::DE< dd::Mass, Element >
>;

template<
    typename T_VirtualDatum1,
    typename T_VirtualDatum2
>
LLAMA_FN_HOST_ACC_INLINE
auto
pPInteraction(
    T_VirtualDatum1&& p1,
    T_VirtualDatum2&& p2,
    Element const & ts
)
-> void
{
    Element const d[3] = {
        p1( dd::Pos(), dd::X() ) -
        p2( dd::Pos(), dd::X() ),
        p1( dd::Pos(), dd::Y() ) -
        p2( dd::Pos(), dd::Y() ),
        p1( dd::Pos(), dd::Z() ) -
        p2( dd::Pos(), dd::Z() )
    };
    Element distSqr = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + EPS2;
    Element distSixth = distSqr * distSqr * distSqr;
    Element invDistCube = 1.0f / sqrtf(distSixth);
    Element s = p1( dd::Mass() ) * invDistCube;
    Element const v_d[3] = {
        d[0] * s * ts,
        d[1] * s * ts,
        d[2] * s * ts
    };
    p1( dd::Vel(), dd::X() ) += v_d[0];
    p1( dd::Vel(), dd::Y() ) += v_d[1];
    p1( dd::Vel(), dd::Z() ) += v_d[2];
}

template<
    typename T_Acc,
    std::size_t T_size,
    std::size_t T_counter,
    std::size_t threads
>
struct BlockSharedMemoryAllocator
{
    using type = nbody::allocator::AlpakaShared<
        T_Acc,
        T_size,
        T_counter
    >;

    template <
        typename T_Factory,
        typename T_Mapping
    >
    LLAMA_FN_HOST_ACC_INLINE
    static
    auto
    allocView(
        T_Mapping const mapping,
        T_Acc const & acc
    )
    -> decltype( T_Factory::allocView( mapping, acc ) )
    {
        return T_Factory::allocView( mapping, acc );
    }
};

template<
    typename T_Acc,
    std::size_t T_size,
    std::size_t T_counter
>
struct BlockSharedMemoryAllocator<
    T_Acc,
    T_size,
    T_counter,
    1
>
{
    using type = llama::allocator::Stack<
        T_size
    >;

    template <
        typename T_Factory,
        typename T_Mapping
    >
    LLAMA_FN_HOST_ACC_INLINE
    static
    auto
    allocView(
        T_Mapping const mapping,
        T_Acc const & acc
    )
    -> decltype( T_Factory::allocView( mapping ) )
    {
        return T_Factory::allocView( mapping );
    }
};

template<
    std::size_t problemSize,
    std::size_t elems,
    std::size_t blockSize
>
struct UpdateKernel
{
    template<
        typename T_Acc,
        typename T_View
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const &acc,
        T_View particles,
        Element ts
    ) const
    {

#if NBODY_USE_SHARED == 1
        constexpr std::size_t threads = blockSize / elems;
        using SharedAllocator = BlockSharedMemoryAllocator<
            T_Acc,
            llama::SizeOf< typename decltype(particles)::Mapping::DatumDomain >::value
            * blockSize,
            __COUNTER__,
            threads
        >;


#if NBODY_USE_SHARED_TREE == 1
        auto treeOperationList = llama::makeTuple(
            llama::mapping::tree::functor::LeaveOnlyRT( )
        );
        using SharedMapping = llama::mapping::tree::Mapping<
            typename decltype(particles)::Mapping::UserDomain,
            typename decltype(particles)::Mapping::DatumDomain,
            decltype( treeOperationList )
        >;
        SharedMapping const sharedMapping(
            { blockSize },
            treeOperationList
        );
#else
        using SharedMapping = llama::mapping::SoA<
            typename decltype(particles)::Mapping::UserDomain,
            typename decltype(particles)::Mapping::DatumDomain
        >;
        SharedMapping const sharedMapping( { blockSize } );
#endif // NBODY_USE_SHARED_TREE

        using SharedFactory = llama::Factory<
            SharedMapping,
            typename SharedAllocator::type
        >;

        auto temp = SharedAllocator::template allocView<
            SharedFactory,
            SharedMapping
        >( sharedMapping, acc );
#endif // NBODY_USE_SHARED

        auto threadIndex  = alpaka::idx::getIdx<
            alpaka::Grid,
            alpaka::Threads
        >( acc )[ 0u ];

        auto const start = threadIndex * elems;
        auto const   end = alpaka::math::min(
            acc,
            start + elems,
            problemSize
        );
        LLAMA_INDEPENDENT_DATA
        for ( std::size_t b = 0; b < problemSize / blockSize; ++b )
        {
            auto const start2 = b * blockSize;
            auto const   end2 = alpaka::math::min(
                acc,
                start2 + blockSize,
                problemSize
            ) - start2;
#if NBODY_USE_SHARED == 1
            LLAMA_INDEPENDENT_DATA
            for (
                auto pos2 = decltype(end2)(0);
                pos2 + threadIndex < end2;
                pos2 += threads
            )
                temp(pos2 + threadIndex) = particles( start2 + pos2 + threadIndex );
#endif // NBODY_USE_SHARED
            LLAMA_INDEPENDENT_DATA
            for ( auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2 )
                LLAMA_INDEPENDENT_DATA
                for ( auto pos = start; pos < end; ++pos )
                    pPInteraction(
                        particles( pos ),
#if NBODY_USE_SHARED == 1
                        temp( pos2 ),
#else
                        particles( start2 + pos2 ),
#endif // NBODY_USE_SHARED
                        ts
                    );
        };
    }
};


template<
    std::size_t problemSize,
    std::size_t elems
>
struct MoveKernel
{
    template<
        typename T_Acc,
        typename T_View
    >
    LLAMA_FN_HOST_ACC_INLINE
    void operator()(
        T_Acc const &acc,
        T_View particles,
        Element ts
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
        {
            particles( pos )( dd::Pos(), dd::X() ) +=
                particles( pos )( dd::Vel(), dd::X() ) * ts;
            particles( pos )( dd::Pos(), dd::Y() ) +=
                particles( pos )( dd::Vel(), dd::Y() ) * ts;
            particles( pos )( dd::Pos(), dd::Z() ) +=
                particles( pos )( dd::Vel(), dd::Z() ) * ts;
        };
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
        static constexpr std::size_t elemCount = 1u;
        static constexpr std::size_t threadCount = blockSize;
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

#if NBODY_CUDA == 1
    using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
#else
    //~ using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    //~ using Acc = alpaka::acc::AccCpuOmp2Threads<Dim, Size>;
    //~ using Acc = alpaka::acc::AccCpuOmp4<Dim, Size>;
#endif // NBODY_CUDA
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
#if NBODY_CUDA == 1
    using Queue = alpaka::queue::QueueCudaRtSync;
#else
    using Queue = alpaka::queue::QueueCpuSync;
#endif // NBODY_CUDA
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) );
    DevHost const devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );
    Queue queue( devAcc ) ;

    // NBODY
    constexpr std::size_t problemSize = NBODY_PROBLEM_SIZE;
    constexpr std::size_t blockSize = NBODY_BLOCK_SIZE;
    constexpr std::size_t hardwareThreads = 2; //relevant for OpenMP2Threads
    using Distribution = ThreadsElemsDistribution<
        Acc,
        blockSize,
        hardwareThreads
    >;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    constexpr Element ts = 0.0001;
    constexpr std::size_t steps = NBODY_STEPS;

    // LLAMA
    using UserDomain = llama::UserDomain< 1 >;
    const UserDomain userDomainSize{ problemSize };

#if NBODY_USE_TREE == 1
    auto treeOperationList = llama::makeTuple(
        llama::mapping::tree::functor::LeaveOnlyRT( )
    );
    using Mapping = llama::mapping::tree::Mapping<
        UserDomain,
        Particle,
        decltype( treeOperationList )
    >;
    const Mapping mapping(
        userDomainSize,
        treeOperationList
    );
#else
    using Mapping = llama::mapping::SoA<
        UserDomain,
        Particle
    >;
    Mapping const mapping( userDomainSize );
#endif // NBODY_USE_TREE

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

    std::cout << problemSize / 1000 / 1000 << " million particles\n";
    std::cout
        << problemSize * llama::SizeOf<Particle>::value / 1000 / 1000
        << "MB \n";

    Chrono chrono;

    auto   hostView =   HostFactory::allocView( mapping, devHost );
    auto    devView =    DevFactory::allocView( mapping,  devAcc );
    auto mirrowView = MirrorFactory::allocView( mapping, devView );

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
        //~ auto temp = llama::tempAlloc< 1, Particle >();
        //~ temp(dd::Pos(), dd::X()) = distribution(generator);
        //~ temp(dd::Pos(), dd::Y()) = distribution(generator);
        //~ temp(dd::Pos(), dd::Z()) = distribution(generator);
        //~ temp(dd::Vel(), dd::X()) = distribution(generator)/Element(10);
        //~ temp(dd::Vel(), dd::Y()) = distribution(generator)/Element(10);
        //~ temp(dd::Vel(), dd::Z()) = distribution(generator)/Element(10);
        hostView(i) = seed;
        //~ hostView(dd::Pos(), dd::X()) = seed;
        //~ hostView(dd::Pos(), dd::Y()) = seed;
        //~ hostView(dd::Pos(), dd::Z()) = seed;
        //~ hostView(dd::Vel(), dd::X()) = seed;
        //~ hostView(dd::Vel(), dd::Y()) = seed;
        //~ hostView(dd::Vel(), dd::Z()) = seed;
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
        UpdateKernel<
            problemSize,
            elemCount,
            blockSize
        > updateKernel;
        alpaka::kernel::exec< Acc > (
            queue,
            workdiv,
            updateKernel,
            mirrowView,
            ts
        );

        chrono.printAndReset("Update kernel");

        MoveKernel<
            problemSize,
            elemCount
        > moveKernel;
        alpaka::kernel::exec<Acc>(
            queue,
            workdiv,
            moveKernel,
            mirrowView,
            ts
        );
        chrono.printAndReset("Move kernel");
        dummy( static_cast<void*>( mirrowView.blob[0] ) );
    }


    return 0;
}

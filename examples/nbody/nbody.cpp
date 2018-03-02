#include <iostream>
#include <utility>

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

#include "AlpakaAllocator.hpp"
#include "Chrono.hpp"
#include "Dummy.hpp"

using Element = float;
constexpr Element EPS2 = 0.01;

LLAMA_DEFINE_DATEDOMAIN(
    Particle, (
        ( Pos, LLAMA_DATESTRUCT, (
            ( X, LLAMA_ATOMTYPE, Element ),
            ( Y, LLAMA_ATOMTYPE, Element ),
            ( Z, LLAMA_ATOMTYPE, Element )
        ) ),
        ( Vel, LLAMA_DATESTRUCT, (
            ( X, LLAMA_ATOMTYPE, Element ),
            ( Y, LLAMA_ATOMTYPE, Element ),
            ( Z, LLAMA_ATOMTYPE, Element )
        ) ),
        ( Mass, LLAMA_ATOMTYPE, Element )
    )
)

template<
    typename T_VirtualDate1,
    typename T_VirtualDate2
>
LLAMA_FN_HOST_ACC_INLINE
auto
pPInteraction(
    T_VirtualDate1&& p1,
    T_VirtualDate2&& p2,
    Element const & ts
)
-> void
{
    Element const d[3] = {
        p1( Particle::Pos::X() ) -
        p2( Particle::Pos::X() ),
        p1( Particle::Pos::Y() ) -
        p2( Particle::Pos::Y() ),
        p1( Particle::Pos::Z() ) -
        p2( Particle::Pos::Z() )
    };
    Element distSqr = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + EPS2;
    Element distSixth = distSqr * distSqr * distSqr;
    Element invDistCube = 1.0f / sqrtf(distSixth);
    Element s = p1( Particle::Mass() ) * invDistCube;
    Element const v_d[3] = {
        d[0] * s * ts,
        d[1] * s * ts,
        d[2] * s * ts
    };
    p1( Particle::Vel::X() ) += v_d[0];
    p1( Particle::Vel::Y() ) += v_d[1];
    p1( Particle::Vel::Z() ) += v_d[2];
}

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
        constexpr std::size_t threads = blockSize / elems;

        using SharedMapping = llama::mapping::SoA<
            typename decltype(particles)::Mapping::UserDomain,
            typename decltype(particles)::Mapping::DateDomain
        >;
        using SharedFactory = llama::Factory<
            SharedMapping,
            nbody::allocator::AlpakaShared<
                T_Acc,
                decltype(particles)::Mapping::DateDomain::size
                * blockSize,
                __COUNTER__
            >
        >;
        SharedMapping sharedMapping( { blockSize } );
        auto temp = SharedFactory::allocView( sharedMapping, acc );

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

            LLAMA_INDEPENDENT_DATA
            for (
                auto pos2 = decltype(end2)(0);
                pos2 + threadIndex < end2;
                pos2 += threads
            )
                temp(pos2 + threadIndex) = particles( start2 + pos2 + threadIndex );

            LLAMA_INDEPENDENT_DATA
            for ( auto pos2 = decltype(end2)(0); pos2 < end2; ++pos2 )
                LLAMA_INDEPENDENT_DATA
                for ( auto pos = start; pos < end; ++pos )
                    pPInteraction(
                        particles( pos ),
                        temp( pos2 ),
                        //~ particles( start2 + pos2 ),
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
            particles( pos )( Particle::Pos::X() ) +=
                particles( pos )( Particle::Vel::X() ) * ts;
            particles( pos )( Particle::Pos::Y() ) +=
                particles( pos )( Particle::Vel::Y() ) * ts;
            particles( pos )( Particle::Pos::Z() ) +=
                particles( pos )( Particle::Vel::Z() ) * ts;
        };
    }
};

template<
    typename T_Acc,
    std::size_t blockSize
>
struct ThreadsElemsDistribution
{
    static constexpr std::size_t elemCount = blockSize;
    static constexpr std::size_t threadCount = 1;
};

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<
        std::size_t blockSize,
        typename T_Dim,
        typename T_Size
    >
    struct ThreadsElemsDistribution<
        alpaka::acc::AccGpuCudaRt<T_Dim, T_Size>,
        blockSize
    >
    {
        static constexpr std::size_t elemCount = 1;
        static constexpr std::size_t threadCount = blockSize;
    };
#endif


int main(int argc,char * * argv)
{

    // ALPAKA
    using Dim = alpaka::dim::DimInt< 1 >;
    using Size = std::size_t;
    using Extents = Size;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    //~ using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuOmp2Blocks<Dim, Size>;
    //~ using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using Stream = alpaka::stream::StreamCpuSync;
    //~ using Stream = alpaka::stream::StreamCudaRtSync;
    DevAcc const devAcc( alpaka::pltf::getDevByIdx< PltfAcc >( 0u ) );
    DevHost const devHost( alpaka::pltf::getDevByIdx< PltfHost >( 0u ) );
    Stream stream( devAcc ) ;

    // NBODY
    constexpr std::size_t problemSize = 64*1024;
    constexpr std::size_t blockSize = 256;
    using Distribution = ThreadsElemsDistribution<
        Acc,
        blockSize
    >;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    constexpr Element ts = 0.0001;
    constexpr std::size_t steps = 5;

    // LLAMA
    using UserDomain = llama::UserDomain< 1 >;
    const UserDomain userDomainSize{ problemSize };
    using DateDomain = Particle::Type;
    using Mapping = llama::mapping::SoA<
        UserDomain,
        DateDomain
    >;
    std::cout << elemCount << '\n';
    Mapping mapping( userDomainSize );

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
    std::cout << problemSize * DateDomain::size / 1000 / 1000 << "MB \n";

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
        //~ auto temp = llama::tempAlloc< 1, DateDomain >();
        //~ temp(Particle::Pos::X()) = distribution(generator);
        //~ temp(Particle::Pos::Y()) = distribution(generator);
        //~ temp(Particle::Pos::Z()) = distribution(generator);
        //~ temp(Particle::Vel::X()) = distribution(generator)/Element(10);
        //~ temp(Particle::Vel::Y()) = distribution(generator)/Element(10);
        //~ temp(Particle::Vel::Z()) = distribution(generator)/Element(10);
        hostView(i) = seed;
        //~ hostView(Particle::Pos::X()) = seed;
        //~ hostView(Particle::Pos::Y()) = seed;
        //~ hostView(Particle::Pos::Z()) = seed;
        //~ hostView(Particle::Vel::X()) = seed;
        //~ hostView(Particle::Vel::Y()) = seed;
        //~ hostView(Particle::Vel::Z()) = seed;
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
        alpaka::stream::enqueue(
            stream,
            alpaka::exec::create< Acc > (
                workdiv,
                updateKernel,
                mirrowView,
                ts
            )
        );

        chrono.printAndReset("Update kernel");

        MoveKernel<
            problemSize,
            elemCount
        > moveKernel;
        alpaka::stream::enqueue(
            stream,
            alpaka::exec::create< Acc > (
                workdiv,
                moveKernel,
                mirrowView,
                ts
            )
        );
        chrono.printAndReset("Move kernel");
        dummy( static_cast<void*>( mirrowView.blob[0] ) );
    }


    return 0;
}

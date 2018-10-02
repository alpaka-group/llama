#include <iostream>
#include <utility>
#include <llama/llama.hpp>

#include "../common/demangle.hpp"

namespace st
{
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Options {};
}

using Name = llama::DS<
    llama::DE< st::Pos, llama::DS<
        llama::DE< st::X, float >,
        llama::DE< st::Y, float >,
        llama::DE< st::Z, float >
    > >,
    llama::DE< st::Momentum,llama::DS<
        llama::DE< st::Z, double >,
        llama::DE< st::X, double >
    > >,
    llama::DE< st::Weight, int >,
    llama::DE< st::Options, llama::DA< bool, 4 > >
>;

template< std::size_t... T_coords >
void printCoords( llama::DatumCoord< T_coords... > dc )
{
    #if __cplusplus >= 201703L
        (std::cout << ... << T_coords);
    #else
        std::cout << type( dc );
    #endif
}

template<
    typename T_VirtualDatum
>
struct SetZeroFunctor
{
    template<
        typename T_OuterCoord,
        typename T_InnerCoord
    >
    auto
    operator()(
        T_OuterCoord,
        T_InnerCoord
    )
    -> void
    {
        vd( typename T_OuterCoord::template Cat< T_InnerCoord >() ) = 0;
        //~ printCoords( typename T_OuterCoord::template Cat< T_InnerCoord >() );
        //~ std::cout << " ";
        //~ printCoords( T_OuterCoord() );
        //~ std::cout << " ";
        //~ printCoords( T_InnerCoord() );
        //~ std::cout << std::endl;
    }
    T_VirtualDatum vd;
};

int main(int argc,char * * argv)
{
    using UD = llama::UserDomain< 2 >;
    UD udSize{ 8192, 8192 };
    std::cout
        << "Datum Domain is "
        << addLineBreaks( type( Name() ) )
        << std::endl;
    std::cout
        << "AoS address of (0,100) <0,1>: "
        << llama::mapping::AoS< UD, Name >( udSize )
            .getBlobByte< 0, 1 >( { 0, 100 } )
        << std::endl;
    std::cout
        << "SoA address of (0,100) <0,1>: "
        << llama::mapping::SoA< UD, Name >( udSize )
            .getBlobByte< 0, 1 >( { 0, 100 } )
        << std::endl;
    std::cout
        << "SizeOf DatumDomain: "
        << llama::SizeOf< Name >::value
        << std::endl;

    std::cout << type( llama::GetCoordFromUID< Name, st::Pos, st::X >() ) << '\n';

    using Mapping = llama::mapping::SoA<
        UD,
        Name,
        llama::LinearizeUserDomainAdress< UD::count >
    >;

    Mapping mapping( udSize );
    using Factory = llama::Factory<
        Mapping,
        llama::allocator::SharedPtr< 256 >
    >;
    auto view = Factory::allocView( mapping );
    const UD pos{ 0, 0 };
    float& position_x = view.accessor< 0, 0 >( pos );
    double& momentum_z = view.accessor< st::Momentum, st::Z >( pos );
    int& weight = view.accessor< 2 >( pos );
    bool& options_2 = view.accessor< 3, 2 >( pos );
    std::cout
        << &position_x
        << std::endl;
    std::cout
        << &momentum_z
        << " "
        << (size_t)&momentum_z - (size_t)&position_x
        << std::endl;
    std::cout
        << &weight
        << " "
        << (size_t)&weight - (size_t)&momentum_z
        << std::endl;
    std::cout
        << &options_2
        << " "
        << (size_t)&options_2 - (size_t)&weight
        << std::endl;

    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            SetZeroFunctor< decltype( view( x, y ) ) > szf{ view( x, y ) };
            llama::ForEach< Name, llama::DatumCoord<0,0> >::apply( szf );
            llama::ForEach< Name, st::Momentum >::apply( szf );
            view.accessor< 1, 0 >( { x, y } ) =
                double( x + y ) / double( udSize[0] + udSize[1] );
        }
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            auto datum = view( x, y );
            datum.access< st::Pos, st::X >() += datum.access< llama::DatumCoord< 1, 0 > >();
            datum.access( st::Pos(), st::Y() ) += datum.access( llama::DatumCoord< 1, 1 >() );
            datum( st::Pos(), st::Z() ) += datum( llama::DatumCoord< 2 >() );
            llama::AdditionFunctor<
                decltype(datum),
                decltype(datum),
                st::Pos
            > as{ datum, datum };
            llama::ForEach<
                Name,
                st::Momentum
            >::apply( as );
        }
    double sum = 0.0;
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
            sum += view( x, y ).access< 1, 0 >(  );
    std::cout
        << "Sum: "
        << sum
        << std::endl;

    return 0;
}

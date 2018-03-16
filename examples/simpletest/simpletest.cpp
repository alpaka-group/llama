#include <iostream>
#include <utility>
#include <llama/llama.hpp>

LLAMA_DEFINE_DATUMDOMAIN(
    Name, (
        ( Pos, LLAMA_DATUMSTRUCT, (
            ( X, LLAMA_ATOMTYPE, float ),
            ( Y, LLAMA_ATOMTYPE, float ),
            ( Z, LLAMA_ATOMTYPE, float )
        ) ),
        ( Momentum, LLAMA_DATUMSTRUCT, (
            ( A, LLAMA_ATOMTYPE, double ),
            ( B, LLAMA_ATOMTYPE, double )
        ) ),
        ( Weight, LLAMA_ATOMTYPE, int ),
        ( Options, LLAMA_DATUMARRAY, (4, LLAMA_ATOMTYPE, bool ) )
    )
)

/* The macro above defines the struct below
 *
 * struct Name : llama::DatumCoord< >
 * {
 *     struct Pos : llama::DatumCoord<0>
 *     {
 *         using X = llama::DatumCoord<0,0>;
 *         using Y = llama::DatumCoord<0,1>;
 *         using Z = llama::DatumCoord<0,2>;
 *     };
 *     struct Momentum : llama::DatumCoord<1>
 *     {
 *         using A = llama::DatumCoord<1,0>;
 *         using B = llama::DatumCoord<1,1>;
 *     };
 *     using Weight = llama::DatumCoord<2>;
 *     struct Options : llama::DatumCoord<3>
 *     {
 *     };
 *
 *     using TypeTree = llama::DatumStruct
 *     <
 *         llama::DatumStruct< float, float, float >,
 *         llama::DatumStruct< double, double >,
 *         int,
 *         llama::DatumArray< bool, 4 >
 *     >;
 * };
 */

template< std::size_t... T_coords >
void printCoords( llama::DatumCoord< T_coords... > )
{
    #if __cplusplus >= 201703L
        (std::cout << ... << T_coords);
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
        << "AoS Adresse: "
        << llama::mapping::AoS< UD, Name >( udSize )
            .getBlobByte< 0, 1 >( { 0, 100 } )
        << std::endl;
    std::cout
        << "SoA Adresse: "
        << llama::mapping::SoA< UD, Name >( udSize )
            .getBlobByte< 0, 1 >( { 0, 100 } )
        << std::endl;

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
    double& momentum_y = view.accessor< 1, 1 >( pos );
    int& weight = view.accessor< 2 >( pos );
    bool& options_2 = view.accessor< 3, 2 >( pos );
    std::cout
        << &position_x
        << std::endl;
    std::cout
        << &momentum_y
        << " "
        << (size_t)&momentum_y - (size_t)&position_x
        << std::endl;
    std::cout
        << &weight
        << " "
        << (size_t)&weight - (size_t)&momentum_y
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
            llama::forEach< Name, Name::Pos >( szf );
            llama::forEach< Name, Name::Momentum >( szf );
            view.accessor< 1, 0 >( { x, y } ) =
                double( x + y ) / double( udSize[0] + udSize[1] );
        }
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            auto datum = view( x, y );
            //~ datum( Name::Momentum::A() ) += datum( llama::DatumCoord< 0, 0 >() );
            //~ datum( Name::Momentum::B() ) += datum( llama::DatumCoord< 0, 1 >() );
            llama::AdditionFunctor<
                decltype(datum),
                decltype(datum),
                Name::Pos
            > as{ datum, datum };
            llama::forEach<
                Name,
                Name::Momentum
            >( as );
        }
    double sum = 0.0;
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
            sum += view.accessor< 1, 0 >( { x, y } );
    std::cout
        << "Sum: "
        << sum
        << std::endl;

    return 0;
}

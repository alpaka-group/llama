#include <iostream>
#include <utility>
#include <llama/llama.hpp>

LLAMA_DEFINE_DATEDOMAIN(
    Name, (
        ( Pos, LLAMA_DATESTRUCT, (
            ( X, LLAMA_ATOMTYPE, float ),
            ( Y, LLAMA_ATOMTYPE, float ),
            ( Z, LLAMA_ATOMTYPE, float )
        ) ),
        ( Momentum, LLAMA_DATESTRUCT, (
            ( A, LLAMA_ATOMTYPE, double ),
            ( B, LLAMA_ATOMTYPE, double )
        ) ),
        ( Weight, LLAMA_ATOMTYPE, int ),
        ( Options, LLAMA_DATEARRAY, (4, LLAMA_ATOMTYPE, bool ) )
    )
)

/* The macro above defines the struct below
 *
 * struct Name : llama::DateCoord< >
 * {
 *     struct Pos : llama::DateCoord<0>
 *     {
 *         using X = llama::DateCoord<0,0>;
 *         using Y = llama::DateCoord<0,1>;
 *         using Z = llama::DateCoord<0,2>;
 *     };
 *     struct Momentum : llama::DateCoord<1>
 *     {
 *         using A = llama::DateCoord<1,0>;
 *         using B = llama::DateCoord<1,1>;
 *     };
 *     using Weight = llama::DateCoord<2>;
 *     struct Options : llama::DateCoord<3>
 *     {
 *     };
 *
 *     using Type = llama::DateStruct
 *     <
 *         llama::DateStruct< float, float, float >,
 *         llama::DateStruct< double, double >,
 *         int,
 *         llama::DateArray< bool, 4 >
 *     >;
 * };
 */

template< std::size_t... T_coords >
void printCoords( llama::DateCoord< T_coords... > )
{
    #if __cplusplus >= 201703L
        (std::cout << ... << T_coords);
    #endif
}

template<
    typename T_VirtualDate
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
    T_VirtualDate vd;
};

int main(int argc,char * * argv)
{
    using UD = llama::UserDomain< 2 >;
    UD udSize{ 8192, 8192 };
    using DD = Name::Type;
    std::cout
        << "AoS Adresse: "
        << llama::mapping::AoS< UD, DD >( udSize )
            .getBlobByte< 0, 1 >( { 0, 100 } )
        << std::endl;
    std::cout
        << "SoA Adresse: "
        << llama::mapping::SoA< UD, DD >( udSize )
            .getBlobByte< 0, 1 >( { 0, 100 } )
        << std::endl;

    using Mapping = llama::mapping::SoA<
        UD,
        DD,
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
            llama::forEach< DD, Name::Pos >( szf );
            llama::forEach< DD, Name::Momentum >( szf );
            view.accessor< 1, 0 >( { x, y } ) =
                double( x + y ) / double( udSize[0] + udSize[1] );
        }
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            auto date = view( x, y );
            //~ date( Name::Momentum::A() ) += date( llama::DateCoord< 0, 0 >() );
            //~ date( Name::Momentum::B() ) += date( llama::DateCoord< 0, 1 >() );
            llama::AdditionFunctor<
                decltype(date),
                decltype(date),
                Name::Pos
            > as{ date, date };
            llama::forEach<
                DD,
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

#include <iostream>
#include <utility>
#include <llama.hpp>

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
 * struct Name
 * {
 *     struct Pos
 *     {
 *         using X = llama::DateCoord<0,0>;
 *         using Y = llama::DateCoord<0,1>;
 *         using Z = llama::DateCoord<0,2>;
 *     };
 *     struct Momentum
 *     {
 *         using A = llama::DateCoord<1,0>;
 *         using B = llama::DateCoord<1,1>;
 *     };
 *     using Weight = llama::DateCoord<2>;
 *     struct Options
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
    auto view = Factory::allowView( mapping );
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

    auto virtualDate = view( pos );

    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
            view.accessor< 1, 0 >( { x, y } ) =
                double( x + y ) / double( udSize[0] + udSize[1] );
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            auto date = view( x, y );
            date( Name::Momentum::A() ) += date( llama::DateCoord< 1, 1 >() );
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

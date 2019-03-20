/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file simpletest.cpp
 *  \brief Simple example for LLAMA showing how to define a user and datum
 *  domain, to create a view and to access the data
 */

#include <iostream>
#include <utility>
#include <vector>
#include <llama/llama.hpp>

#include "../common/demangle.hpp"

namespace simpletest
{

/** LLAMA uses class names for accessing data in the datum domain. It makes sense
 *  to encapsulate these in own namespaces, especially if you are using similar
 *  namings in differnt datum domains, e.g. X, Y and Z in different kinds of
 *  spaces.
 */
namespace st
{
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Options {};
} // namespace st

/** A datum domain in LLAMA is a type, probably always a
 *  \ref llama::DatumStruct, which can be shortend to llama::DS. This takes a
 *  template list of all members of this struct-like domain. Every member needs
 *  to be a \ref llama::DatumElement, which can itself be shortend to llama::DE.
 *  A DatumElement is a list of two elements itself, first the name of the
 *  element as type and secondly the element type itself, which may be a
 *  nested DatumStruct. A handy shortcut is \ref llama::DatumArray (llama::DA),
 *  which defines multuple datum elements with an anonymous name but of the same
 *  type.
 */
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

/** Prints the coordinates of a given \ref llama::DatumCoord for debugging and
 *  testing purposes
 */
template< std::size_t... T_coords >
void printCoords( llama::DatumCoord< T_coords... > dc )
{
    #if __cplusplus >= 201703L
        (std::cout << ... << T_coords);
    #else
        std::cout << type( dc );
    #endif
}

/** Example functor for \ref llama::ForEach which can also be used to print the
 *  coordinates inside of a datum domain when called.
 */
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
    // Defining a two-dimensional user domain
    using UD = llama::UserDomain< 2 >;
    // Setting the run time size of the user domain to 8192 * 8192
    UD udSize{ 8192, 8192 };

    // Printing the domain informations at runtime
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
    using NameStub = llama::StubType< Name >;
    static_assert( std::is_same<Name, NameStub::type >::value,
                   "Type from StubType does not match original type" );
    std::cout
        << "sizeof( llama::StubType< DatumDomain > ): "
        << sizeof( NameStub )
        << std::endl;

    std::vector<NameStub> v;
    static_assert( std::is_same<Name, decltype(v)::value_type::type >::value,
                   "Type from StubType does not match original type" );

    std::cout << type( llama::GetCoordFromUID< Name, st::Pos, st::X >() ) << '\n';

    // chosing a native struct of array mapping for this simple test example
    using Mapping = llama::mapping::SoA<
        UD,
        Name,
        llama::LinearizeUserDomainAdress< UD::count >
    >;

    // Instantiating the mapping with the user domain size
    Mapping mapping( udSize );
    // Defining the factory type based on the mapping and the chosen allocator
    using Factory = llama::Factory<
        Mapping,
        llama::allocator::SharedPtr< 256 >
    >;
    // getting a view wiht allocated memory from the free Factory allocView
    // function
    auto view = Factory::allocView( mapping );

    // defining a position in the user domain
    const UD pos{ 0, 0 };

    // using the position in the user domain and a tree coord or a uid in the
    // datum domain to get the reference to an element in the view
    float& position_x = view( pos ).access< 0, 0 >();
    double& momentum_z = view[ pos ].access< st::Momentum, st::Z >();
    int& weight = view[ {0,0} ]( llama::DatumCoord< 2 >() );
    bool& options_2 = view[ 0 ]( st::Options() )( llama::DatumCoord< 2 >() );
    // printing the address and distances of the element in the memory. This
    // will change based on the chosen mapping. When array of struct is chosen
    // instead the elements will be much closer than with struct of array.
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

    // iterating over the user domain at run time to do some stuff with the
    // allocated data
    for (size_t x = 0; x < udSize[0]; ++x)
        // telling the compiler that all data in the following loop is
        // independent to each other and thus can be vectorized
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            // Defining a functor for a given virtual datum
            SetZeroFunctor< decltype( view( x, y ) ) > szf{ view( x, y ) };
            // Applying the functor for the sub tree 0,0 (pos.x), so basically
            // only for this element
            llama::ForEach< Name, llama::DatumCoord<0,0> >::apply( szf );
            // Applying the functor for the sub tree momentum (0), so basically
            // for momentum.z, and momentum.x
            llama::ForEach< Name, st::Momentum >::apply( szf );
            // the user domain address can be given as multiple comma separated
            // arguments or as one parameter of type user domain
            view( { x, y } ) =
                double( x + y ) / double( udSize[0] + udSize[1] );
        }
    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            // Showing different options of access data with llama. Internally
            // all do the same data- and mappingwise
            auto datum = view( x, y );
            datum.access< st::Pos, st::X >() +=
                datum.access< llama::DatumCoord< 1, 0 > >();
            datum.access( st::Pos(), st::Y() ) +=
                datum.access( llama::DatumCoord< 1, 1 >() );
            datum( st::Pos(), st::Z() ) += datum( llama::DatumCoord< 2 >() );

            // It is also possible to work only on a part of data. The statement
            // below does the same as the commented out forEach call shown
            // afterwards.
            datum( st::Pos() ) += datum( st::Momentum() );
            /* The line above does the same as:
                llama::AdditionFunctor<
                    decltype(datum),
                    decltype(datum),
                    st::Pos
                > as{ datum, datum };
                llama::ForEach<
                    Name,
                    st::Momentum
                >::apply( as );
            */
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

} // namespace simpletest

int main(int argc,char * * argv)
{
    return simpletest::main( argc, argv );
}

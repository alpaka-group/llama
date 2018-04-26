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
}

#include "toString.hpp"

using Name = llama::DS<
    llama::DE< st::Pos, llama::DS<
        llama::DE< st::X, double >,
        llama::DE< st::Y, double >,
        llama::DE< st::Z, double >
    > >,
    llama::DE< st::Weight, float >,
    llama::DE< st::Momentum,llama::DS<
        llama::DE< st::Z, double >,
        llama::DE< st::Y, double >,
        llama::DE< st::X, double >
    > >
>;

int main(int argc,char * * argv)
{
    std::cout
        << "Datum Domain is\n"
        << addLineBreaks( type( Name() ) )
        << std::endl;

    using UD = llama::UserDomain< 2 >;
    UD const udSize{ 8192, 8192 };

    auto treeOperationList = llama::makeTuple(
        llama::mapping::tree::functor::Idem(),

        //~ llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< >
        //~ >( 8192 ),
        //~ llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< 0 >
        //~ >( 8192 * 8192 ),
        //~ llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< 0, 0 >
        //~ >( 8192 * 8192 ),
        //~ llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< 0, 2 >
        //~ >( 8192 * 8192 ),

        //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< >,
            //~ 8192
        //~ >( ),
        //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< 0 >,
            //~ 8192 * 8192
        //~ >( ),
        //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< 0, 0 >,
            //~ 8192 * 8192
        //~ >( ),
        //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< 0, 2 >,
            //~ 8192 * 8192
        //~ >( ),

        llama::mapping::tree::functor::LeaveOnlyRT( ),

        llama::mapping::tree::functor::Idem()
    );

    using Mapping = llama::mapping::tree::Mapping<
        UD,
        Name,
        decltype( treeOperationList )
    >;
    const Mapping mapping(
        udSize,
        treeOperationList
    );

    std::cout
        << "Basic mapping tree type ("
        << sizeof( Mapping::BasicTree)
        << " bytes) is\n"
        << addLineBreaks( type( mapping.basicTree ) )
        << std::endl;

    std::cout
        << "Result mapping tree type ("
        << sizeof( Mapping::ResultTree)
        << " bytes) is\n"
        << addLineBreaks( type( mapping.resultTree ) )
        << std::endl;

    std::cout
        << "Basic mapping tree value is\n"
        << llama::mapping::tree::toString( mapping.basicTree )
        << std::endl;
    std::cout
        << "Result mapping tree value is\n"
        << llama::mapping::tree::toString( mapping.resultTree )
        << std::endl;

    //~ using Mapping = llama::mapping::SoA<
        //~ UD,
        //~ Name
    //~ >;
    //~ Mapping mapping(
        //~ udSize
    //~ );

    std::cout
        << "BlobSize: "
        << mapping.getBlobSize( 0 )
        << std::endl;

    std::cout
        << "BlobByte(50,100,Mom,Y): "
        << mapping.getBlobByte<2,1>( {50, 100} )
        << std::endl;
    std::cout
        << "BlobByte(50,101,Mom,Y): "
        << mapping.getBlobByte<2,1>( {50, 101} )
        << std::endl;

    using Factory = llama::Factory<
        Mapping,
        llama::allocator::SharedPtr< 256 >
    >;
    auto view = Factory::allocView( mapping );

    for (size_t x = 0; x < udSize[0]; ++x)
        LLAMA_INDEPENDENT_DATA
        for (size_t y = 0; y < udSize[1]; ++y)
        {
            auto datum = view( x, y );
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
            sum += view.accessor< 0, 1 >( { x, y } );
    std::cout
        << "Sum: "
        << sum
        << std::endl;

    return 0;
}

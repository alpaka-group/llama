/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../common/demangle.hpp"

#include <iostream>
#include <llama/llama.hpp>
#include <utility>

/** \file treemaptest.cpp
 *  \brief Example to show how to work with the tree mapping of LLAMA.
 */

namespace treemaptest
{
    namespace st
    {
        // clang-format off
        struct Pos{};
        struct X{};
        struct Y{};
        struct Z{};
        struct Momentum{};
        struct Weight{};
        // clang-format on

        auto toString(Pos)
        {
            return "Pos";
        }
        auto toString(X)
        {
            return "X";
        }
        auto toString(Y)
        {
            return "Y";
        }
        auto toString(Z)
        {
            return "Z";
        }
        auto toString(Momentum)
        {
            return "Momentum";
        }
        auto toString(Weight)
        {
            return "Weight";
        }
    }

    // clang-format off
    using Name = llama::DS<
        llama::DE<st::Pos, llama::DS<
            llama::DE<st::X, double>,
            llama::DE<st::Y, double>,
            llama::DE<st::Z, double>>>,
        llama::DE<st::Weight, float>,
        llama::DE<st::Momentum, llama::DS<
            llama::DE<st::Z, double>,
            llama::DE<st::Y, double>,
            llama::DE<st::X, double>>>>;
    // clang-format on

    int main(int argc, char ** argv)
    {
        std::cout << "Datum Domain is\n"
                  << addLineBreaks(type(Name())) << std::endl;

        constexpr std::size_t userDomainSize = 1024 * 12;

        using UD = llama::UserDomain<2>;
        UD const udSize{userDomainSize, userDomainSize};

        auto treeOperationList = llama::Tuple{
            llama::mapping::tree::functor::Idem(),
            //~ llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< >
            //~ >( userDomainSize )
            //~ ,llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< 0 >
            //~ >( userDomainSize * userDomainSize )
            //~ ,llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< 0, 0 >
            //~ >( userDomainSize * userDomainSize )
            //~ ,llama::mapping::tree::functor::MoveRTDown<
            //~ llama::mapping::tree::TreeCoord< 0, 2 >
            //~ >( userDomainSize * userDomainSize )

            //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< >,
            //~ userDomainSize
            //~ >( ),
            //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< 0 >,
            //~ userDomainSize * userDomainSize
            //~ >( ),
            //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< 0, 0 >,
            //~ userDomainSize * userDomainSize
            //~ >( ),
            //~ llama::mapping::tree::functor::MoveRTDownFixed<
            //~ llama::mapping::tree::TreeCoord< 0, 2 >,
            //~ userDomainSize * userDomainSize
            //~ >( )

            llama::mapping::tree::functor::LeafOnlyRT(),
            llama::mapping::tree::functor::Idem()};

        using Mapping = llama::mapping::tree::
            Mapping<UD, Name, decltype(treeOperationList)>;
        const Mapping mapping(udSize, treeOperationList);

        std::cout << "Basic mapping tree type (" << sizeof(Mapping::BasicTree)
                  << " bytes) is\n"
                  << addLineBreaks(type(mapping.basicTree)) << std::endl;

        std::cout << "Result mapping tree type (" << sizeof(Mapping::ResultTree)
                  << " bytes) is\n"
                  << addLineBreaks(type(mapping.resultTree)) << std::endl;

        std::cout << "Basic mapping tree value is\n"
                  << llama::mapping::tree::toString(mapping.basicTree)
                  << std::endl;
        std::cout << "Result mapping tree value is\n"
                  << llama::mapping::tree::toString(mapping.resultTree)
                  << std::endl;

        //~ using Mapping = llama::mapping::SoA<
        //~ UD,
        //~ Name
        //~ >;
        //~ Mapping mapping(
        //~ udSize
        //~ );

        std::cout << "BlobSize: " << mapping.getBlobSize(0) << std::endl;

        std::cout << "BlobByte(50,100,Mom,Y): "
                  << mapping.getBlobByte<2, 1>({50, 100}) << std::endl;
        std::cout << "BlobByte(50,101,Mom,Y): "
                  << mapping.getBlobByte<2, 1>({50, 101}) << std::endl;

        using Factory
            = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>;
        auto view = Factory::allocView(mapping);

        for(size_t x = 0; x < udSize[0]; ++x) LLAMA_INDEPENDENT_DATA
        for(size_t y = 0; y < udSize[1]; ++y)
        {
            auto datum = view(x, y);
            llama::AdditionFunctor<decltype(datum), decltype(datum), st::Pos>
                as{datum, datum};
            llama::ForEach<Name, st::Momentum>::apply(as);
            //~ auto datum2 = view( x+1, y );
            //~ datum( st::Pos(), st::Y() ) += datum2( st::Pos(), st::Y() );
        }
        double sum = 0.0;
        for(size_t x = 0; x < udSize[0]; ++x) LLAMA_INDEPENDENT_DATA
        for(size_t y = 0; y < udSize[1]; ++y)
            sum += view.accessor<0, 1>({x, y});
        std::cout << "Sum: " << sum << std::endl;

        return 0;
    }

} // namespace treemaptest

int main(int argc, char ** argv)
{
    return treemaptest::main(argc, argv);
}

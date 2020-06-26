#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include "common.h"

namespace st {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
}

namespace llama::mapping::tree {
    template<>
    struct ToString<st::Pos> {
        auto operator()(const st::Pos) { return "Pos"; }
    };
    template<>
    struct ToString<st::X> {
        auto operator()(const st::X) { return "X"; }
    };
    template<>
    struct ToString<st::Y> {
        auto operator()(const st::Y) { return "Y"; }
    };
    template<>
    struct ToString<st::Z> {
        auto operator()(const st::Z) { return "Z"; }
    };
    template<>
    struct ToString<st::Momentum> {
        auto operator()(const st::Momentum) { return "Momentum"; }
    };
    template<>
    struct ToString<st::Weight> {
        auto operator()(const st::Weight) { return "Weight"; }
    };
}

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

TEST_CASE("treemapping") {
    constexpr std::size_t userDomainSize = 1024 * 12;

    using UD = llama::UserDomain<2>;
    const UD udSize{userDomainSize, userDomainSize};

    auto treeOperationList = llama::makeTuple(
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

        llama::mapping::tree::functor::LeafOnlyRT{}
        , llama::mapping::tree::functor::Idem{}
    );

    using Mapping = llama::mapping::tree::Mapping<UD, Name, decltype(treeOperationList)>;
    const Mapping mapping(udSize, treeOperationList);

    CHECK(sizeof(Mapping::BasicTree) == 24);
    auto raw = prettyPrintType(mapping.basicTree);
#ifdef _WIN32
    boost::replace_all(raw, "__int64", "long");
#endif
    CHECK(raw == R"(llama::mapping::tree::TreeElement<
    llama::NoName,
    llama::Tuple<
        llama::mapping::tree::TreeElement<
            llama::NoName,
            llama::Tuple<
                llama::mapping::tree::TreeElement<
                    st::Pos,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            st::X,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            st::Y,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            st::Z,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::TreeElement<
                    st::Weight,
                    float,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::TreeElement<
                    st::Momentum,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            st::Z,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            st::Y,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            st::X,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >
            >,
            unsigned long
        >
    >,
    unsigned long
>)");

    CHECK(sizeof(Mapping::ResultTree) == 56);
    auto raw2 = prettyPrintType(mapping.resultTree);
#ifdef _WIN32
    boost::replace_all(raw2, "__int64", "long");
#endif
    CHECK(raw2 == R"(llama::mapping::tree::TreeElement<
    llama::NoName,
    llama::Tuple<
        llama::mapping::tree::TreeElement<
            llama::NoName,
            llama::Tuple<
                llama::mapping::tree::TreeElement<
                    st::Pos,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            st::X,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            st::Y,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            st::Z,
                            double,
                            unsigned long
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::TreeElement<
                    st::Weight,
                    float,
                    unsigned long
                >,
                llama::mapping::tree::TreeElement<
                    st::Momentum,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            st::Z,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            st::Y,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            st::X,
                            double,
                            unsigned long
                        >
                    >,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >
            >,
            std::integral_constant<
                unsigned long,
                1
            >
        >
    >,
    std::integral_constant<
        unsigned long,
        1
    >
>)");

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "12288 * [ 12288 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 "
          "* X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "1 * [ 1 * [ 1 * Pos[ 150994944 * X(double) , 150994944 * Y(double) , 150994944 * Z(double) ] , 150994944 "
          "* Weight(float) , 1 * Momentum[ 150994944 * Z(double) , 150994944 * Y(double) , 150994944 * X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 7851737088);
    CHECK(mapping.getBlobByte<2, 1>({50, 100}) == 5440733984);
    CHECK(mapping.getBlobByte<2, 1>({50, 101}) == 5440733992);

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>;
    auto view = Factory::allocView(mapping);

    for (size_t x = 0; x < udSize[0]; ++x)
        for (size_t y = 0; y < udSize[1]; ++y) {
            auto datum = view(x, y);
            llama::AdditionFunctor<decltype(datum), decltype(datum), st::Pos> as{datum, datum};
            llama::ForEach<Name, st::Momentum>::apply(as);
            //~ auto datum2 = view( x+1, y );
            //~ datum( st::Pos(), st::Y() ) += datum2( st::Pos(), st::Y() );
        }
    double sum = 0.0;
    for (size_t x = 0; x < udSize[0]; ++x)
        for (size_t y = 0; y < udSize[1]; ++y)
            sum += view.accessor<0, 1>({ x, y });
    CHECK(sum == 0);
}

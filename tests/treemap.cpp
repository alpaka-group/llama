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
    CHECK(prettyPrintType(mapping.basicTree) == R"(struct llama::mapping::tree::TreeElement<
    struct llama::NoName,struct llama::Tuple<
        struct llama::mapping::tree::TreeElement<
            struct llama::NoName,struct llama::Tuple<
                struct llama::mapping::tree::TreeElement<
                    struct st::Pos,struct llama::Tuple<
                        struct llama::mapping::tree::TreeElement<
                            struct st::X,double,struct std::integral_constant<
                                unsigned __int64,1
                            >
                            >,struct llama::mapping::tree::TreeElement<
                                struct st::Y,double,struct std::integral_constant<
                                    unsigned __int64,1
                                >
                                >,struct llama::mapping::tree::TreeElement<
                                    struct st::Z,double,struct std::integral_constant<
                                        unsigned __int64,1
                                    >
                                >
                                >,struct std::integral_constant<
                                    unsigned __int64,1
                                >
                                >,struct llama::mapping::tree::TreeElement<
                                    struct st::Weight,float,struct std::integral_constant<
                                        unsigned __int64,1
                                    >
                                    >,struct llama::mapping::tree::TreeElement<
                                        struct st::Momentum,struct llama::Tuple<
                                            struct llama::mapping::tree::TreeElement<
                                                struct st::Z,double,struct std::integral_constant<
                                                    unsigned __int64,1
                                                >
                                                >,struct llama::mapping::tree::TreeElement<
                                                    struct st::Y,double,struct std::integral_constant<
                                                        unsigned __int64,1
                                                    >
                                                    >,struct llama::mapping::tree::TreeElement<
                                                        struct st::X,double,struct std::integral_constant<
                                                            unsigned __int64,1
                                                        >
                                                    >
                                                    >,struct std::integral_constant<
                                                        unsigned __int64,1
                                                    >
                                                >
                                                >,unsigned __int64
                                            >
                                            >,unsigned __int64
                                        >)");

    CHECK(sizeof(Mapping::ResultTree) == 56);
    CHECK(prettyPrintType(mapping.resultTree) == R"(struct llama::mapping::tree::TreeElement<
    struct llama::NoName,struct llama::Tuple<
        struct llama::mapping::tree::TreeElement<
            struct llama::NoName,struct llama::Tuple<
                struct llama::mapping::tree::TreeElement<
                    struct st::Pos,struct llama::Tuple<
                        struct llama::mapping::tree::TreeElement<
                            struct st::X,double,unsigned __int64
                            >,struct llama::mapping::tree::TreeElement<
                                struct st::Y,double,unsigned __int64
                                >,struct llama::mapping::tree::TreeElement<
                                    struct st::Z,double,unsigned __int64
                                >
                                >,struct std::integral_constant<
                                    unsigned __int64,1
                                >
                                >,struct llama::mapping::tree::TreeElement<
                                    struct st::Weight,float,unsigned __int64
                                    >,struct llama::mapping::tree::TreeElement<
                                        struct st::Momentum,struct llama::Tuple<
                                            struct llama::mapping::tree::TreeElement<
                                                struct st::Z,double,unsigned __int64
                                                >,struct llama::mapping::tree::TreeElement<
                                                    struct st::Y,double,unsigned __int64
                                                    >,struct llama::mapping::tree::TreeElement<
                                                        struct st::X,double,unsigned __int64
                                                    >
                                                    >,struct std::integral_constant<
                                                        unsigned __int64,1
                                                    >
                                                >
                                                >,struct std::integral_constant<
                                                    unsigned __int64,1
                                                >
                                            >
                                            >,struct std::integral_constant<
                                                unsigned __int64,1
                                            >
                                        >)");

    CHECK(llama::mapping::tree::toString(mapping.basicTree) == "12288 * [ 12288 * [ 1 * Pos[ 1 * X(double __cdecl(void)) , 1 * Y(double __cdecl(void)) , 1 * Z(double __cdecl(void)) ] , 1 * Weight(float __cdecl(void)) , 1 * Momentum[ 1 * Z(double __cdecl(void)) , 1 * Y(double __cdecl(void)) , 1 * X(double __cdecl(void)) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) == "1 * [ 1 * [ 1 * Pos[ 150994944 * X(double __cdecl(void)) , 150994944 * Y(double __cdecl(void)) , 150994944 * Z(double __cdecl(void)) ] , 150994944 * Weight(float __cdecl(void)) , 1 * Momentum[ 150994944 * Z(double __cdecl(void)) , 150994944 * Y(double __cdecl(void)) , 150994944 * X(double __cdecl(void)) ] ] ]");

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

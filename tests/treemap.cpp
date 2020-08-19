#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

namespace tag
{
    // clang-format off
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
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
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Weight, float>,
    llama::DE<tag::Momentum, llama::DS<
        llama::DE<tag::Z, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::X, double>
    >>
>;
// clang-format on

TEST_CASE("treemapping.empty")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList = llama::Tuple{};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 16);
        CHECK(mapping.getBlobByte<1>(coord) == 24);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 28);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 36);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 44);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 52);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 60);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 68);
        CHECK(mapping.getBlobByte<1>(coord) == 76);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 80);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 88);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 96);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 840);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 848);
        CHECK(mapping.getBlobByte<1>(coord) == 856);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 860);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 868);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 876);
    }
}

TEST_CASE("treemapping.Idem")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList
        = llama::Tuple{llama::mapping::tree::functor::Idem()};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 16);
        CHECK(mapping.getBlobByte<1>(coord) == 24);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 28);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 36);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 44);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 52);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 60);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 68);
        CHECK(mapping.getBlobByte<1>(coord) == 76);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 80);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 88);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 96);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 840);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 848);
        CHECK(mapping.getBlobByte<1>(coord) == 856);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 860);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 868);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 876);
    }
}

TEST_CASE("treemapping.LeafOnlyRT")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList
        = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "1 * [ 1 * [ 1 * Pos[ 256 * X(double) , 256 * Y(double) , 256 * Z(double) ] , 256 * Weight(float) , 1 * Momentum[ 256 * Z(double) , 256 * Y(double) "
          ", 256 * X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 2048);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 4096);
        CHECK(mapping.getBlobByte<1>(coord) == 6144);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 7168);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 9216);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 11264);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 2056);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 4104);
        CHECK(mapping.getBlobByte<1>(coord) == 6148);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 7176);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 9224);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 11272);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 128);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 2176);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 4224);
        CHECK(mapping.getBlobByte<1>(coord) == 6208);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 7296);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 9344);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 11392);
    }
}

TEST_CASE("treemapping.MoveRTDown<>")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList
        = llama::Tuple{llama::mapping::tree::functor::MoveRTDown<
            llama::mapping::tree::TreeCoord<>>{4}};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "4 * [ 64 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 16);
        CHECK(mapping.getBlobByte<1>(coord) == 24);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 28);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 36);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 44);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 52);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 60);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 68);
        CHECK(mapping.getBlobByte<1>(coord) == 76);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 80);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 88);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 96);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 840);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 848);
        CHECK(mapping.getBlobByte<1>(coord) == 856);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 860);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 868);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 876);
    }
}

TEST_CASE("treemapping.MoveRTDown<0>")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList
        = llama::Tuple{llama::mapping::tree::functor::MoveRTDown<
            llama::mapping::tree::TreeCoord<0>>{4}};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 4 * [ 4 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 4 * Weight(float) , 4 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 16);
        CHECK(mapping.getBlobByte<1>(coord) == 96);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 112);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 120);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 128);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 24);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 32);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 40);
        CHECK(mapping.getBlobByte<1>(coord) == 100);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 136);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 144);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 152);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 840);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 848);
        CHECK(mapping.getBlobByte<1>(coord) == 928);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 944);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 952);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 960);
    }
}

TEST_CASE("treemapping.MoveRTDown<0,0>")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList
        = llama::Tuple{llama::mapping::tree::functor::MoveRTDown<
            llama::mapping::tree::TreeCoord<0, 0>>{4}};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 16 * [ 1 * Pos[ 4 * X(double) , 4 * Y(double) , 4 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 31744);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 32);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 64);
        CHECK(mapping.getBlobByte<1>(coord) == 96);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 100);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 108);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 116);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 124);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 156);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 188);
        CHECK(mapping.getBlobByte<1>(coord) == 220);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 224);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 232);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 240);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 1984);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 2016);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 2048);
        CHECK(mapping.getBlobByte<1>(coord) == 2080);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 2084);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 2092);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 2100);
    }
}

TEST_CASE("treemapping.MoveRTDownFixed<>")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList = llama::Tuple{
        llama::mapping::tree::functor::
            MoveRTDownFixed<llama::mapping::tree::TreeCoord<>, 4>{}};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "4 * [ 64 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 16);
        CHECK(mapping.getBlobByte<1>(coord) == 24);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 28);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 36);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 44);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 52);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 60);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 68);
        CHECK(mapping.getBlobByte<1>(coord) == 76);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 80);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 88);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 96);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 840);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 848);
        CHECK(mapping.getBlobByte<1>(coord) == 856);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 860);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 868);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 876);
    }
}

TEST_CASE("treemapping.MoveRTDownFixed<0>")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList = llama::Tuple{
        llama::mapping::tree::functor::
            MoveRTDownFixed<llama::mapping::tree::TreeCoord<0>, 4>{}};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 4 * [ 4 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 4 * Weight(float) , 4 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 16);
        CHECK(mapping.getBlobByte<1>(coord) == 96);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 112);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 120);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 128);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 24);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 32);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 40);
        CHECK(mapping.getBlobByte<1>(coord) == 100);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 136);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 144);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 152);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 840);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 848);
        CHECK(mapping.getBlobByte<1>(coord) == 928);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 944);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 952);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 960);
    }
}

TEST_CASE("treemapping.MoveRTDownFixed<0,0>")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList = llama::Tuple{
        llama::mapping::tree::functor::
            MoveRTDownFixed<llama::mapping::tree::TreeCoord<0, 0>, 4>{}};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 16 * [ 1 * Pos[ 4 * X(double) , 4 * Y(double) , 4 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 31744);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 32);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 64);
        CHECK(mapping.getBlobByte<1>(coord) == 96);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 100);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 108);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 116);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 124);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 156);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 188);
        CHECK(mapping.getBlobByte<1>(coord) == 220);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 224);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 232);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 240);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 1984);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 2016);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 2048);
        CHECK(mapping.getBlobByte<1>(coord) == 2080);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 2084);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 2092);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 2100);
    }
}

TEST_CASE("treemapping.vectorblocks.runtime")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    const auto vectorWidth = 8;

    auto treeOperationList = llama::Tuple{
        llama::mapping::tree::functor::MoveRTDown<
            llama::mapping::tree::TreeCoord<0>>{
            vectorWidth}, // move 8 down from UserDomain (to
                          // Position/Weight/Momentum)
        llama::mapping::tree::functor::MoveRTDown<
            llama::mapping::tree::TreeCoord<0, 0>>{
            vectorWidth}, // move 8 down from Position (to X/Y/Z)
        llama::mapping::tree::functor::MoveRTDown<
            llama::mapping::tree::TreeCoord<0, 2>>{
            vectorWidth}, // move 8 down from Momentum (to X/Y/Z)
    };
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 2 * [ 1 * Pos[ 8 * X(double) , 8 * Y(double) , 8 * Z(double) ] , 8 * Weight(float) , "
          "1 * Momentum[ 8 * Z(double) , 8 * Y(double) , 8 * X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 64);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 128);
        CHECK(mapping.getBlobByte<1>(coord) == 192);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 224);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 288);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 352);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 72);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 136);
        CHECK(mapping.getBlobByte<1>(coord) == 196);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 232);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 296);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 360);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 896);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 960);
        CHECK(mapping.getBlobByte<1>(coord) == 1024);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 1056);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 1120);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 1184);
    }
}

TEST_CASE("treemapping.vectorblocks.compiletime")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    constexpr auto vectorWidth = 8;

    auto treeOperationList = llama::Tuple{
        llama::mapping::tree::functor::MoveRTDownFixed<
            llama::mapping::tree::TreeCoord<0>,
            vectorWidth>{}, // move 8 down from UserDomain (to
                            // Position/Weight/Momentum)
        llama::mapping::tree::functor::MoveRTDownFixed<
            llama::mapping::tree::TreeCoord<0, 0>,
            vectorWidth>{}, // move 8 down from Position (to X/Y/Z)
        llama::mapping::tree::functor::MoveRTDownFixed<
            llama::mapping::tree::TreeCoord<0, 2>,
            vectorWidth>{}, // move 8 down from Momentum (to X/Y/Z)
    };
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    CHECK(llama::mapping::tree::toString(mapping.basicTree) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * "
          "X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "16 * [ 2 * [ 1 * Pos[ 8 * X(double) , 8 * Y(double) , 8 * Z(double) ] , 8 * Weight(float) , "
          "1 * Momentum[ 8 * Z(double) , 8 * Y(double) , 8 * X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 13312);

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 0);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 64);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 128);
        CHECK(mapping.getBlobByte<1>(coord) == 192);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 224);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 288);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 352);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 8);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 72);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 136);
        CHECK(mapping.getBlobByte<1>(coord) == 196);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 232);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 296);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 360);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobByte<0, 0>(coord) == 832);
        CHECK(mapping.getBlobByte<0, 1>(coord) == 896);
        CHECK(mapping.getBlobByte<0, 2>(coord) == 960);
        CHECK(mapping.getBlobByte<1>(coord) == 1024);
        CHECK(mapping.getBlobByte<2, 0>(coord) == 1056);
        CHECK(mapping.getBlobByte<2, 1>(coord) == 1120);
        CHECK(mapping.getBlobByte<2, 2>(coord) == 1184);
    }
}

TEST_CASE("treemapping.getNode")
{
    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{16, 16};

    auto treeOperationList = llama::Tuple{};
    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

    using namespace llama::mapping::tree;
    using namespace llama::mapping::tree::operations;

    CHECK(toString(getNode<TreeCoord<>>(mapping.resultTree)) ==
          "16 * [ 16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) "
          ", 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * X(double) ] ] ]");
    CHECK(toString(getNode<TreeCoord<0>>(mapping.resultTree)) ==
          "16 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * "
          "Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * X(double) ] ]");
    CHECK(
        toString(getNode<TreeCoord<0, 0>>(mapping.resultTree))
        == "1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ]");
    CHECK(
        toString(getNode<TreeCoord<0, 0, 0>>(mapping.resultTree))
        == "1 * X(double)");
    CHECK(
        toString(getNode<TreeCoord<0, 0, 1>>(mapping.resultTree))
        == "1 * Y(double)");
    CHECK(
        toString(getNode<TreeCoord<0, 0, 2>>(mapping.resultTree))
        == "1 * Z(double)");
    CHECK(
        toString(getNode<TreeCoord<0, 1>>(mapping.resultTree))
        == "1 * Weight(float)");
    CHECK(
        toString(getNode<TreeCoord<0, 2>>(mapping.resultTree))
        == "1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 * X(double) ]");
    CHECK(
        toString(getNode<TreeCoord<0, 2, 0>>(mapping.resultTree))
        == "1 * Z(double)");
    CHECK(
        toString(getNode<TreeCoord<0, 2, 1>>(mapping.resultTree))
        == "1 * Y(double)");
    CHECK(
        toString(getNode<TreeCoord<0, 2, 2>>(mapping.resultTree))
        == "1 * X(double)");
}

TEST_CASE("treemapping")
{
    constexpr std::size_t userDomainSize = 12;

    using UserDomain = llama::UserDomain<2>;
    const UserDomain userDomain{userDomainSize, userDomainSize};

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

        llama::mapping::tree::functor::LeafOnlyRT{},
        llama::mapping::tree::functor::Idem{}};

    using Mapping = llama::mapping::tree::
        Mapping<UserDomain, Name, decltype(treeOperationList)>;
    const Mapping mapping(userDomain, treeOperationList);

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
                    tag::Pos,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            tag::X,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::Y,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::Z,
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
                    tag::Weight,
                    float,
                    std::integral_constant<
                        unsigned long,
                        1
                    >
                >,
                llama::mapping::tree::TreeElement<
                    tag::Momentum,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            tag::Z,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::Y,
                            double,
                            std::integral_constant<
                                unsigned long,
                                1
                            >
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::X,
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
                    tag::Pos,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            tag::X,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::Y,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::Z,
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
                    tag::Weight,
                    float,
                    unsigned long
                >,
                llama::mapping::tree::TreeElement<
                    tag::Momentum,
                    llama::Tuple<
                        llama::mapping::tree::TreeElement<
                            tag::Z,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::Y,
                            double,
                            unsigned long
                        >,
                        llama::mapping::tree::TreeElement<
                            tag::X,
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
          "12 * [ 12 * [ 1 * Pos[ 1 * X(double) , 1 * Y(double) , 1 * Z(double) ] , 1 * Weight(float) , 1 * Momentum[ 1 * Z(double) , 1 * Y(double) , 1 "
          "* X(double) ] ] ]");
    CHECK(llama::mapping::tree::toString(mapping.resultTree) ==
          "1 * [ 1 * [ 1 * Pos[ 144 * X(double) , 144 * Y(double) , 144 * Z(double) ] , 144 * Weight(float) , 1 * Momentum[ 144 * Z(double) , 144 * Y(double) , 144 * X(double) ] ] ]");

    CHECK(mapping.getBlobSize(0) == 7488);
    CHECK(mapping.getBlobByte<2, 1>({50, 100}) == 10784);
    CHECK(mapping.getBlobByte<2, 1>({50, 101}) == 10792);

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>;
    auto view = Factory::allocView(mapping);
    zeroStorage(view);

    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
        {
            auto datum = view(x, y);
            llama::AdditionFunctor<decltype(datum), decltype(datum), tag::Pos>
                as{datum, datum};
            llama::ForEach<Name, tag::Momentum>::apply(as);
            //~ auto datum2 = view( x+1, y );
            //~ datum( tag::Pos(), tag::Y() ) += datum2( tag::Pos(), tag::Y() );
        }
    double sum = 0.0;
    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
            sum += view.accessor<0, 1>({x, y});
    CHECK(sum == 0);
}

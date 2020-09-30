#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
}

using Name = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>,
        llama::DE<tag::Z, float>
    >>
>;
// clang-format on

TEST_CASE("ArrayDomain.CTAD")
{
    llama::ArrayDomain ud0 {};
    llama::ArrayDomain ud1 {1};
    llama::ArrayDomain ud2 {1, 1};
    llama::ArrayDomain ud3 {1, 1, 1};

    STATIC_REQUIRE(decltype(ud0)::rank == 0);
    STATIC_REQUIRE(decltype(ud1)::rank == 1);
    STATIC_REQUIRE(decltype(ud2)::rank == 2);
    STATIC_REQUIRE(decltype(ud3)::rank == 3);
}

TEST_CASE("ArrayDomain.dim0")
{
    using ArrayDomain = llama::ArrayDomain<0>;
    ArrayDomain arrayDomain {};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping {arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain {}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("ArrayDomain.dim1")
{
    using ArrayDomain = llama::ArrayDomain<1>;
    ArrayDomain arrayDomain {16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping {arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain {0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("ArrayDomain.dim2")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain {16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping {arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain {0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("ArrayDomain.dim3")
{
    using ArrayDomain = llama::ArrayDomain<3>;
    ArrayDomain arrayDomain {16, 16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping {arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain {0, 0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("ArrayDomain.dim10")
{
    using ArrayDomain = llama::ArrayDomain<10>;
    ArrayDomain arrayDomain {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping {arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("ArrayDomain.ctor")
{
    llama::ArrayDomain<1> ud {};
    CHECK(ud[0] == 0);
}

TEST_CASE("UserDomainCoordRange1D")
{
    llama::ArrayDomain<1> ud {3};

    std::vector<llama::ArrayDomain<1>> coords;
    for (auto coord : llama::UserDomainCoordRange {ud})
        coords.push_back(coord);

    CHECK(coords == std::vector<llama::ArrayDomain<1>> {{0}, {1}, {2}});
}

TEST_CASE("UserDomainCoordRange2D")
{
    llama::ArrayDomain<2> ud {3, 3};

    std::vector<llama::ArrayDomain<2>> coords;
    for (auto coord : llama::UserDomainCoordRange {ud})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::ArrayDomain<2>> {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("UserDomainCoordRange3D")
{
    llama::ArrayDomain<3> ud {3, 3, 3};

    std::vector<llama::ArrayDomain<3>> coords;
    for (auto coord : llama::UserDomainCoordRange {ud})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::ArrayDomain<3>> {
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

TEST_CASE("UserDomainCoordRange3D.destructering")
{
    llama::ArrayDomain<3> ud {1, 1, 1};

    for (auto [x, y, z] : llama::UserDomainCoordRange {ud})
    {
        CHECK(x == 0);
        CHECK(y == 0);
        CHECK(z == 0);
    }
}

TEST_CASE("Morton")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain {2, 3};

    llama::mapping::LinearizeUserDomainMorton lin;
    CHECK(lin.size(ArrayDomain {2, 3}) == 4 * 4);
    CHECK(lin.size(ArrayDomain {2, 4}) == 4 * 4);
    CHECK(lin.size(ArrayDomain {2, 5}) == 8 * 8);
    CHECK(lin.size(ArrayDomain {8, 8}) == 8 * 8);

    CHECK(lin(ArrayDomain {0, 0}, {}) == 0);
    CHECK(lin(ArrayDomain {0, 1}, {}) == 1);
    CHECK(lin(ArrayDomain {0, 2}, {}) == 4);
    CHECK(lin(ArrayDomain {0, 3}, {}) == 5);
    CHECK(lin(ArrayDomain {1, 0}, {}) == 2);
    CHECK(lin(ArrayDomain {1, 1}, {}) == 3);
    CHECK(lin(ArrayDomain {1, 2}, {}) == 6);
    CHECK(lin(ArrayDomain {1, 3}, {}) == 7);
    CHECK(lin(ArrayDomain {2, 0}, {}) == 8);
    CHECK(lin(ArrayDomain {2, 1}, {}) == 9);
    CHECK(lin(ArrayDomain {2, 2}, {}) == 12);
    CHECK(lin(ArrayDomain {2, 3}, {}) == 13);
    CHECK(lin(ArrayDomain {3, 0}, {}) == 10);
    CHECK(lin(ArrayDomain {3, 1}, {}) == 11);
    CHECK(lin(ArrayDomain {3, 2}, {}) == 14);
    CHECK(lin(ArrayDomain {3, 3}, {}) == 15);
}

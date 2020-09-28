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

TEST_CASE("UserDomain.CTAD")
{
    llama::UserDomain ud0 {};
    llama::UserDomain ud1 {1};
    llama::UserDomain ud2 {1, 1};
    llama::UserDomain ud3 {1, 1, 1};

    STATIC_REQUIRE(decltype(ud0)::rank == 0);
    STATIC_REQUIRE(decltype(ud1)::rank == 1);
    STATIC_REQUIRE(decltype(ud2)::rank == 2);
    STATIC_REQUIRE(decltype(ud3)::rank == 3);
}

TEST_CASE("UserDomain.dim0")
{
    using UserDomain = llama::UserDomain<0>;
    UserDomain userDomain {};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    float& x = view(UserDomain {}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("UserDomain.dim1")
{
    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain {16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    float& x = view(UserDomain {0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("UserDomain.dim2")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    float& x = view(UserDomain {0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("UserDomain.dim3")
{
    using UserDomain = llama::UserDomain<3>;
    UserDomain userDomain {16, 16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    float& x = view(UserDomain {0, 0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("UserDomain.dim10")
{
    using UserDomain = llama::UserDomain<10>;
    UserDomain userDomain {2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    float& x = view(UserDomain {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("UserDomain.ctor")
{
    llama::UserDomain<1> ud {};
    CHECK(ud[0] == 0);
}

TEST_CASE("UserDomainCoordRange1D")
{
    llama::UserDomain<1> ud {3};

    std::vector<llama::UserDomain<1>> coords;
    for (auto coord : llama::UserDomainCoordRange {ud})
        coords.push_back(coord);

    CHECK(coords == std::vector<llama::UserDomain<1>> {{0}, {1}, {2}});
}

TEST_CASE("UserDomainCoordRange2D")
{
    llama::UserDomain<2> ud {3, 3};

    std::vector<llama::UserDomain<2>> coords;
    for (auto coord : llama::UserDomainCoordRange {ud})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::UserDomain<2>> {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("UserDomainCoordRange3D")
{
    llama::UserDomain<3> ud {3, 3, 3};

    std::vector<llama::UserDomain<3>> coords;
    for (auto coord : llama::UserDomainCoordRange {ud})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::UserDomain<3>> {
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

TEST_CASE("UserDomainCoordRange3D.destructering")
{
    llama::UserDomain<3> ud {1, 1, 1};

    for (auto [x, y, z] : llama::UserDomainCoordRange {ud})
    {
        CHECK(x == 0);
        CHECK(y == 0);
        CHECK(z == 0);
    }
}

TEST_CASE("Morton")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain {2, 3};

    llama::mapping::LinearizeUserDomainMorton lin;
    CHECK(lin.size(UserDomain {2, 3}) == 4 * 4);
    CHECK(lin.size(UserDomain {2, 4}) == 4 * 4);
    CHECK(lin.size(UserDomain {2, 5}) == 8 * 8);
    CHECK(lin.size(UserDomain {8, 8}) == 8 * 8);

    CHECK(lin(UserDomain {0, 0}, {}) == 0);
    CHECK(lin(UserDomain {0, 1}, {}) == 1);
    CHECK(lin(UserDomain {0, 2}, {}) == 4);
    CHECK(lin(UserDomain {0, 3}, {}) == 5);
    CHECK(lin(UserDomain {1, 0}, {}) == 2);
    CHECK(lin(UserDomain {1, 1}, {}) == 3);
    CHECK(lin(UserDomain {1, 2}, {}) == 6);
    CHECK(lin(UserDomain {1, 3}, {}) == 7);
    CHECK(lin(UserDomain {2, 0}, {}) == 8);
    CHECK(lin(UserDomain {2, 1}, {}) == 9);
    CHECK(lin(UserDomain {2, 2}, {}) == 12);
    CHECK(lin(UserDomain {2, 3}, {}) == 13);
    CHECK(lin(UserDomain {3, 0}, {}) == 10);
    CHECK(lin(UserDomain {3, 1}, {}) == 11);
    CHECK(lin(UserDomain {3, 2}, {}) == 14);
    CHECK(lin(UserDomain {3, 3}, {}) == 15);
}

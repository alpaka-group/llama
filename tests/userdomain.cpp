#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

TEST_CASE("userdomain ctor")
{
    llama::UserDomain<1> ud{};
    CHECK(ud[0] == 0);
}

TEST_CASE("UserDomainCoordRange1D")
{
    llama::UserDomain<1> ud{3};

    std::vector<llama::UserDomain<1>> coords;
    for(auto coord : llama::UserDomainCoordRange{ud}) coords.push_back(coord);

    CHECK(coords == std::vector<llama::UserDomain<1>>{{0}, {1}, {2}});
}

TEST_CASE("UserDomainCoordRange2D")
{
    llama::UserDomain<2> ud{3, 3};

    std::vector<llama::UserDomain<2>> coords;
    for(auto coord : llama::UserDomainCoordRange{ud}) coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::UserDomain<2>>{
            {0, 0},
            {0, 1},
            {0, 2},
            {1, 0},
            {1, 1},
            {1, 2},
            {2, 0},
            {2, 1},
            {2, 2}});
}

TEST_CASE("UserDomainCoordRange3D")
{
    llama::UserDomain<3> ud{3, 3, 3};

    std::vector<llama::UserDomain<3>> coords;
    for(auto coord : llama::UserDomainCoordRange{ud}) coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::UserDomain<3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2},
            {0, 2, 0}, {0, 2, 1}, {0, 2, 2}, {1, 0, 0}, {1, 0, 1}, {1, 0, 2},
            {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2},
            {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

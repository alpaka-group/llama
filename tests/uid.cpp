#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Flags {};
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>,
        llama::DE<tag::Z, float>
    >>,
    llama::DE<llama::NoName, int>,
    llama::DE<tag::Vel,llama::DS<
        llama::DE<tag::Z, double>,
        llama::DE<tag::X, double>
    >>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;

using Other = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::Z, float>,
        llama::DE<tag::Y, float>
    >>
>;
// clang-format on

TEST_CASE("GetCoordFromTags")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle                             >, llama::DatumCoord<    >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos                   >, llama::DatumCoord<0   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::X           >, llama::DatumCoord<0, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Y           >, llama::DatumCoord<0, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Z           >, llama::DatumCoord<0, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, llama::NoName              >, llama::DatumCoord<1   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::Z           >, llama::DatumCoord<2, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::X           >, llama::DatumCoord<2, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags                 >, llama::DatumCoord<3   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<0>>, llama::DatumCoord<3, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<1>>, llama::DatumCoord<3, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<2>>, llama::DatumCoord<3, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<3>>, llama::DatumCoord<3, 3>>);
    // clang-format on
}

TEST_CASE("GetTag")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<0, 0>>, tag::X       >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<0   >>, tag::Pos     >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<    >>, llama::NoName>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<2, 1>>, tag::X       >);
    // clang-format on
}

TEST_CASE("hasSameTags")
{
    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::GetCoordFromTags<Particle, tag::Pos>, // Base A
            llama::DatumCoord<0>, // Local A
            Particle, // DD B
            llama::GetCoordFromTags<Particle, tag::Vel>, // Base B
            llama::DatumCoord<0> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::GetCoordFromTags<Particle, tag::Pos>, // Base A
            llama::DatumCoord<0>, // Local A
            Particle, // DD B
            llama::GetCoordFromTags<Particle, tag::Vel>, // Base B
            llama::DatumCoord<1> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<0, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<0, 2>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<2, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            > == false);
}

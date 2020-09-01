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

static_assert(
    std::is_same_v<llama::GetCoordFromUID<Particle>, llama::DatumCoord<>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Pos>,
              llama::DatumCoord<0>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Pos, tag::X>,
              llama::DatumCoord<0, 0>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Pos, tag::Y>,
              llama::DatumCoord<0, 1>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Pos, tag::Z>,
              llama::DatumCoord<0, 2>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, llama::NoName>,
              llama::DatumCoord<1>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Vel, tag::Z>,
              llama::DatumCoord<2, 0>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Vel, tag::X>,
              llama::DatumCoord<2, 1>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Flags>,
              llama::DatumCoord<3>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Flags, llama::Index<0>>,
              llama::DatumCoord<3, 0>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Flags, llama::Index<1>>,
              llama::DatumCoord<3, 1>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Flags, llama::Index<2>>,
              llama::DatumCoord<3, 2>>);
static_assert(std::is_same_v<
              llama::GetCoordFromUID<Particle, tag::Flags, llama::Index<3>>,
              llama::DatumCoord<3, 3>>);

TEST_CASE("prettyPrintType")
{
    CHECK(
        prettyPrintType(llama::GetUID<
                        Particle,
                        llama::GetCoordFromUID<Particle, tag::Pos, tag::X>>())
        == "tag::X");
    CHECK(
        prettyPrintType(
            llama::
                GetUID<Particle, llama::GetCoordFromUID<Particle, tag::Pos>>())
        == "tag::Pos");
    CHECK(
        prettyPrintType(
            llama::GetUID<Particle, llama::GetCoordFromUID<Particle>>())
        == "llama::NoName");
    CHECK(
        prettyPrintType(llama::GetUID<
                        Particle,
                        llama::GetCoordFromUID<Particle, tag::Vel, tag::X>>())
        == "tag::X");
}

TEST_CASE("CompareUID")
{
    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::GetCoordFromUID<Particle, tag::Pos>, // Base A
            llama::DatumCoord<0>, // Local A
            Particle, // DD B
            llama::GetCoordFromUID<Particle, tag::Vel>, // Base B
            llama::DatumCoord<0> // Local B
            > == false);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::GetCoordFromUID<Particle, tag::Pos>, // Base A
            llama::DatumCoord<0>, // Local A
            Particle, // DD B
            llama::GetCoordFromUID<Particle, tag::Vel>, // Base B
            llama::DatumCoord<1> // Local B
            > == true);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<0, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            > == false);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<0, 2>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            > == true);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<2, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            > == false);
}

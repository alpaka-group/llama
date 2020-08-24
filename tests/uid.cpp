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
    >>
>;

using Other = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::Z, float>,
        llama::DE<tag::Y, float>
    >>
>;
// clang-format on

TEST_CASE("uid")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{2, 2};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    using MappingOther = llama::mapping::SoA<UserDomain, Other>;

    Mapping mapping{userDomain};
    MappingOther mappingOther{userDomain};

    auto particle = allocView(mapping);
    auto other = allocView(mappingOther);

    // Setting some test values
    particle(0u, 0u)(tag::Pos(), tag::X()) = 0.0f;
    particle(0u, 0u)(tag::Pos(), tag::Y()) = 0.0f;
    particle(0u, 0u)(tag::Pos(), tag::Z()) = 13.37f;
    particle(0u, 0u)(tag::Vel(), tag::X()) = 4.2f;
    particle(0u, 0u)(tag::Vel(), tag::Z()) = 0.0f;

    other(0u, 0u)(tag::Pos(), tag::Y()) = 5.f;
    other(0u, 0u)(tag::Pos(), tag::Z()) = 2.3f;

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

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::GetCoordFromUID<Particle, tag::Pos>, // Base A
            llama::DatumCoord<0>, // Local A
            Particle, // DD B
            llama::GetCoordFromUID<Particle, tag::Vel>, // Base B
            llama::DatumCoord<0> // Local B
            >::value
        == false);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::GetCoordFromUID<Particle, tag::Pos>, // Base A
            llama::DatumCoord<0>, // Local A
            Particle, // DD B
            llama::GetCoordFromUID<Particle, tag::Vel>, // Base B
            llama::DatumCoord<1> // Local B
            >::value
        == true);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<0, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            >::value
        == false);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<0, 2>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            >::value
        == true);

    CHECK(
        llama::CompareUID<
            Particle, // DD A
            llama::DatumCoord<>, // Base A
            llama::DatumCoord<2, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<>, // Base B
            llama::DatumCoord<0, 0> // Local B
            >::value
        == false);

    CHECK(particle(0u, 0u)(tag::Pos(), tag::Z()) == 13.37f);
    particle(0u, 0u) += other(0u, 0u);
    CHECK(particle(0u, 0u)(tag::Pos(), tag::Z()) == 15.67f);
}

#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include "common.h"

namespace st {
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
}

using Particle = llama::DS<
    llama::DE< st::Pos, llama::DS<
        llama::DE< st::X, float >,
        llama::DE< st::Y, float >,
        llama::DE< st::Z, float >
    > >,
    llama::DE< llama::NoName, int >,
    llama::DE< st::Vel,llama::DS<
        llama::DE< st::Z, double >,
        llama::DE< st::X, double >
    > >
>;

using Other = llama::DS<
    llama::DE< st::Pos, llama::DS<
        llama::DE< st::Z, float >,
        llama::DE< st::Y, float >
    > >
>;

TEST_CASE("uid") {
    using UD = llama::UserDomain<2>;
    UD udSize{2, 2};

    using Mapping = llama::mapping::SoA<UD, Particle, llama::LinearizeUserDomainAdress<UD::count>>;
    using MappingOther = llama::mapping::SoA<UD, Other, llama::LinearizeUserDomainAdress<UD::count>>;

    Mapping mapping{udSize};
    MappingOther mappingOther{udSize};

    auto particle = llama::Factory<Mapping, llama::allocator::SharedPtr<256>>::allocView(mapping);
    auto other = llama::Factory<MappingOther,llama::allocator::SharedPtr<256>>::allocView(mappingOther);

    // Setting some test values
    particle(0u, 0u)(st::Pos(), st::X()) = 0.0f;
    particle(0u, 0u)(st::Pos(), st::Y()) = 0.0f;
    particle(0u, 0u)(st::Pos(), st::Z()) = 13.37f;
    particle(0u, 0u)(st::Vel(), st::X()) = 4.2f;
    particle(0u, 0u)(st::Vel(), st::Z()) = 0.0f;

    other(0u, 0u)(st::Pos(), st::Y()) = 5.f;
    other(0u, 0u)(st::Pos(), st::Z()) = 2.3f;

    CHECK(prettyPrintType(llama::GetUID<Particle, llama::GetCoordFromUID<Particle, st::Pos, st::X>>()) == "struct st::X");
    CHECK(prettyPrintType(llama::GetUID<Particle, llama::GetCoordFromUID<Particle, st::Pos>>()) == "struct st::Pos");
    CHECK(prettyPrintType(llama::GetUID<Particle, llama::GetCoordFromUID<Particle>>()) == "struct llama::NoName");
    CHECK(prettyPrintType(llama::GetUID<Particle, llama::GetCoordFromUID<Particle, st::Vel, st::X>>()) == "struct st::X");

    CHECK(llama::CompareUID<
        Particle,                                    // DD A
        llama::GetCoordFromUID< Particle, st::Pos >, // Base A
        llama::DatumCoord<0>,                        // Local A
        Particle,                                    // DD B
        llama::GetCoordFromUID< Particle, st::Vel >, // Base B
        llama::DatumCoord<0>                         // Local B
    >::value == false);

    CHECK(llama::CompareUID<
        Particle,                                    // DD A
        llama::GetCoordFromUID< Particle, st::Pos >, // Base A
        llama::DatumCoord<0>,                        // Local A
        Particle,                                    // DD B
        llama::GetCoordFromUID< Particle, st::Vel >, // Base B
        llama::DatumCoord<1>                         // Local B
    >::value == true);

    CHECK(llama::CompareUID<
                Particle,               // DD A
                llama::DatumCoord< >,   // Base A
                llama::DatumCoord<0,0>, // Local A
                Other,                  // DD B
                llama::DatumCoord< >,   // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value == false);

    CHECK(llama::CompareUID<
                Particle,               // DD A
                llama::DatumCoord< >,   // Base A
                llama::DatumCoord<0,2>, // Local A
                Other,                  // DD B
                llama::DatumCoord< >,   // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value == true);

    CHECK(llama::CompareUID<
                Particle,               // DD A
                llama::DatumCoord< >,   // Base A
                llama::DatumCoord<2,0>, // Local A
                Other,                  // DD B
                llama::DatumCoord< >,   // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value == false);

    CHECK(particle(0u, 0u)(st::Pos(), st::Z()) == 13.37f);
    particle(0u, 0u) += other(0u, 0u);
    CHECK(particle(0u, 0u)(st::Pos(), st::Z()) == 15.67f);
}

#include <iostream>
#include <utility>
#include <llama/llama.hpp>

#include "../common/demangle.hpp"



namespace st
{
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

int main(int argc,char * * argv)
{
    using UD = llama::UserDomain< 2 >;
    UD udSize{ 2, 2 };
    using Mapping = llama::mapping::SoA<
        UD,
        Particle,
        llama::LinearizeUserDomainAdress< UD::count >
    >;
    Mapping mapping( udSize );
    auto particle = llama::Factory<
        Mapping,
        llama::allocator::SharedPtr< 256 >
    >::allocView( mapping );

    using MappingOther = llama::mapping::SoA<
        UD,
        Other,
        llama::LinearizeUserDomainAdress< UD::count >
    >;
    MappingOther mappingOther( udSize );
    auto other = llama::Factory<
        MappingOther,
        llama::allocator::SharedPtr< 256 >
    >::allocView( mappingOther );

    particle( 0u, 0u )( st::Pos(), st::X() ) = 0.0f;
    particle( 0u, 0u )( st::Pos(), st::Y() ) = 0.0f;
    particle( 0u, 0u )( st::Pos(), st::Z() ) = 13.37f;

    particle( 0u, 0u )( st::Vel(), st::X() ) = 4.2f;
    particle( 0u, 0u )( st::Vel(), st::Z() ) = 0.0f;

    other( 0u, 0u )( st::Pos(), st::Y() ) = 5.f;
    other( 0u, 0u )( st::Pos(), st::Z() ) = 2.3f;

    std::cout
        << "UID of Particle.Pos.X: "
        << type( typename llama::GetUID<
                Particle,
                llama::GetCoordFromUID<
                    Particle,
                    st::Pos,
                    st::X
                >
            >() )
        << std::endl;
    std::cout
        << "UID of Particle.Pos: "
        << type( typename llama::GetUID<
                Particle,
                llama::GetCoordFromUID<
                    Particle,
                    st::Pos
                >
            >() )
        << std::endl;
    std::cout
        << "UID of Particle: "
        << type( typename llama::GetUID<
                Particle,
                llama::GetCoordFromUID<
                    Particle
                >
            >() )
        << std::endl;
    std::cout
        << "UID of Particle.Vel.X: "
        << type( typename llama::GetUID<
                Particle,
                llama::GetCoordFromUID<
                    Particle,
                    st::Vel,
                    st::X
                >
            >() )
        << std::endl;

    std::cout
        << "Particle.Pos.0 == Particle.Vel.0 (X == Z) -> "
        << llama::CompareUID<
                Particle,                                    // DD A
                llama::GetCoordFromUID< Particle, st::Pos >, // Base A
                llama::DatumCoord<0>,                        // Local A
                Particle,                                    // DD B
                llama::GetCoordFromUID< Particle, st::Vel >, // Base B
                llama::DatumCoord<0>                         // Local B
            >::value
        << std::endl;
    std::cout
        << "Particle.Pos.0 == Particle.Vel.1 (X == X) -> "
        << llama::CompareUID<
                Particle,                                    // DD A
                llama::GetCoordFromUID< Particle, st::Pos >, // Base A
                llama::DatumCoord<0>,                        // Local A
                Particle,                                    // DD B
                llama::GetCoordFromUID< Particle, st::Vel >, // Base B
                llama::DatumCoord<1>                         // Local B
            >::value
        << std::endl;
    std::cout
        << "Particle.0.0 == Other.0.0 (Pos.X == Pos.Z) -> "
        << llama::CompareUID<
                Particle,               // DD A
                llama::DatumCoord< >,   // Base A
                llama::DatumCoord<0,0>, // Local A
                Other,                  // DD B
                llama::DatumCoord< >,   // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value
        << std::endl;
    std::cout
        << "Particle.0.2 == Other.0.0 (Pos.Z == Pos.Z) -> "
        << llama::CompareUID<
                Particle,               // DD A
                llama::DatumCoord< >,   // Base A
                llama::DatumCoord<0,2>, // Local A
                Other,                  // DD B
                llama::DatumCoord< >,   // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value
        << std::endl;
    std::cout
        << "Particle.2.0 == Other.0.0 (Vel.Z == Pos.Z) -> "
        << llama::CompareUID<
                Particle,               // DD A
                llama::DatumCoord< >,   // Base A
                llama::DatumCoord<2,0>, // Local A
                Other,                  // DD B
                llama::DatumCoord< >,   // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value
        << std::endl;

    std::cout
        << "Before: view( 0u, 0u )( st::Pos(), st::Z() ) = "
        << particle( 0u, 0u )( st::Pos(), st::Z() )
        << std::endl;


    //~ auto p = particle( 0u, 0u );
    //~ auto o = other( 0u, 0u );
    //~ llama::AdditionIfSameUIDFunctor<
        //~ decltype(p),
        //~ llama::GetCoordFromUID< Particle, st::Pos >,
        //~ llama::DatumCoord<2>,
        //~ decltype(o),
        //~ llama::GetCoordFromUID< Other, st::Pos >,
        //~ llama::DatumCoord<0>
    //~ > aisuf{p,o};
    //~ aisuf();

    particle( 0u, 0u ) += other( 0u, 0u );

    std::cout
        << "After:  view( 0u, 0u )( st::Pos(), st::Z() ) = "
        << particle( 0u, 0u )( st::Pos(), st::Z() )
        << std::endl;

    return 0;
}

#include <iostream>
#include <utility>
#include <llama/llama.hpp>

LLAMA_DEFINE_DATUMDOMAIN(
    Particle, (
        ( Pos, LLAMA_DATUMSTRUCT, (
            ( X, LLAMA_ATOMTYPE, float ),
            ( Y, LLAMA_ATOMTYPE, float ),
            ( Z, LLAMA_ATOMTYPE, float )
        ) ),
        ( NotUsed, LLAMA_ATOMTYPE, int ),
        ( Vel, LLAMA_DATUMSTRUCT, (
            ( Z, LLAMA_ATOMTYPE, double ),
            ( X, LLAMA_ATOMTYPE, double )
        ) )
    )
)

LLAMA_DEFINE_DATUMDOMAIN(
    Other, (
        ( Pos, LLAMA_DATUMSTRUCT, (
            ( Z, LLAMA_ATOMTYPE, float ),
            ( Y, LLAMA_ATOMTYPE, float )
        ) )
    )
)

#include "demangle.hpp"

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

    particle( 0u, 0u )( Particle::Pos::X() ) = 0.0f;
    particle( 0u, 0u )( Particle::Pos::Y() ) = 0.0f;
    particle( 0u, 0u )( Particle::Pos::Z() ) = 13.37f;

    particle( 0u, 0u )( Particle::Vel::X() ) = 4.2f;
    particle( 0u, 0u )( Particle::Vel::Z() ) = 0.0f;

    other( 0u, 0u )( Other::Pos::Y() ) = 5.f;
    other( 0u, 0u )( Other::Pos::Z() ) = 2.3f;

    std::cout
        << "UID of Particle::Pos::X "
        << type( typename llama::GetUIDFromName<
                Particle,
                Particle::Pos::X
            >() )
        << std::endl;
    std::cout
        << "UID of Particle::Pos "
        << type( typename llama::GetUIDFromName<
                Particle,
                Particle::Pos
            >() )
        << std::endl;
    std::cout
        << "UID of Particle "
        << type( typename llama::GetUIDFromName<
                Particle,
                Particle
            >() )
        << std::endl;
    std::cout
        << "UID of Particle::Vel::X "
        << type( typename llama::GetUIDFromName<
                Particle,
                Particle::Vel::X
            >() )
        << std::endl;

    std::cout
        << "Particle::Pos::<0> == Particle::Vel::<0> (X == Z) -> "
        << llama::CompareUID<
                Particle,             // DD A
                Particle::Pos,        // Base A
                llama::DatumCoord<0>, // Local A
                Particle,             // DD B
                Particle::Vel,        // Base B
                llama::DatumCoord<0>  // Local B
            >::value
        << std::endl;
    std::cout
        << "Particle::Pos::<0> == Particle::Vel::<1> (X == X) -> "
        << llama::CompareUID<
                Particle,             // DD A
                Particle::Pos,        // Base A
                llama::DatumCoord<0>, // Local A
                Particle,             // DD B
                Particle::Vel,        // Base B
                llama::DatumCoord<1>  // Local B
            >::value
        << std::endl;

    std::cout
        << "Particle::<0,0> == Other::<0,0> (Pos::X == Pos::Z) -> "
        << llama::CompareUID<
                Particle,               // DD A
                Particle,               // Base A
                llama::DatumCoord<0,0>, // Local A
                Other,                  // DD B
                Other,                  // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value
        << std::endl;
    std::cout
        << "Particle::<0,2> == Other::<0,0> (Pos::Z == Pos::Z) -> "
        << llama::CompareUID<
                Particle,               // DD A
                Particle,               // Base A
                llama::DatumCoord<0,2>, // Local A
                Other,                  // DD B
                Other,                  // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value
        << std::endl;
    std::cout
        << "Particle::<2,0> == Other::<0,0> (Vel::Z == Pos::Z) -> "
        << llama::CompareUID<
                Particle,               // DD A
                Particle,               // Base A
                llama::DatumCoord<2,0>, // Local A
                Other,                  // DD B
                Other,                  // Base B
                llama::DatumCoord<0,0>  // Local B
            >::value
        << std::endl;

    std::cout
        << "Before: view( 0u, 0u )( Particle::Pos::Z() ) = "
        << particle( 0u, 0u )( Particle::Pos::Z() )
        << std::endl;


    //~ auto p = particle( 0u, 0u );
    //~ auto o = other( 0u, 0u );
    //~ llama::AdditionIfSameUIDFunctor<
        //~ decltype(p),
        //~ Particle::Pos,
        //~ llama::DatumCoord<2>,
        //~ decltype(o),
        //~ Other::Pos,
        //~ llama::DatumCoord<0>
    //~ > aisuf{p,o};
    //~ aisuf();

    particle( 0u, 0u ) += other( 0u, 0u );

    std::cout
        << "After:  view( 0u, 0u )( Particle::Pos::Z() ) = "
        << particle( 0u, 0u )( Particle::Pos::Z() )
        << std::endl;

    return 0;
}

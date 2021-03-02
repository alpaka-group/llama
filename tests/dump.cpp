#include "common.h"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>
#include <string>

namespace
{
    template <std::size_t N>
    using Padding = std::array<std::byte, N>;
}

// clang-format off
namespace tag
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Mass {};
    struct Flags {};
    struct Id {};
    struct Pad {};
}

using Vec = llama::DS<
    llama::DE<tag::X, float>,
    llama::DE<tag::Y, float>,
    llama::DE<tag::Z, float>
>;
//using DVec = llama::DS<
//    llama::DE<tag::X, double>,
//    llama::DE<tag::Y, double>,
//    llama::DE<tag::Z, double>
//>;
using Particle = llama::DS<
    llama::DE<tag::Pos, Vec>,
    llama::DE<tag::Vel, /*DVec*/Vec>,
    llama::DE<tag::Mass, float>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;

// example with bad alignment:
using ParticleUnaligned = llama::DS<
    llama::DE<tag::Id, std::uint16_t>,
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>
    >>,
    llama::DE<tag::Mass, double>,
    llama::DE<tag::Flags, llama::DA<bool, 3>>
>;

// bad alignment fixed with explicit padding:
using ParticleAligned = llama::DS<
    llama::DE<tag::Id, std::uint16_t>,
    llama::DE<tag::Pad, Padding<2>>,
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>
    >>,
    llama::DE<tag::Pad, Padding<4>>,
    llama::DE<tag::Mass, double>,
    llama::DE<tag::Flags, llama::DA<bool, 3>>,
    llama::DE<tag::Pad, Padding<5>>
>;
// clang-format on

namespace
{
    llama::ArrayDomain arrayDomain{32};
    using ArrayDomain = decltype(arrayDomain);

    template <typename Mapping>
    void dump(const Mapping& mapping, std::string filename)
    {
        std::ofstream{filename + ".svg"} << llama::toSvg(mapping);
        std::ofstream{filename + ".html"} << llama::toHtml(mapping);
    }
} // namespace

TEST_CASE("dump.AoS")
{
    dump(llama::mapping::AoS{arrayDomain, Particle{}}, "AoSMapping");
}

TEST_CASE("dump.SoA")
{
    dump(llama::mapping::SoA{arrayDomain, Particle{}}, "SoAMapping");
}

TEST_CASE("dump.SoA.MultiBlob")
{
    dump(llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}}, "SoAMappingMultiBlob");
}

TEST_CASE("dump.AoSoA.8")
{
    dump(llama::mapping::AoSoA<ArrayDomain, Particle, 8>{arrayDomain}, "AoSoAMapping8");
}

TEST_CASE("dump.AoSoA.32")
{
    dump(llama::mapping::AoSoA<ArrayDomain, Particle, 32>{arrayDomain}, "AoSoAMapping32");
}

TEST_CASE("dump.Split.SoA.AoS.1Buffer")
{
    // split out velocity (written in nbody, the rest is read)
    dump(
        llama::mapping::
            Split<ArrayDomain, Particle, llama::DatumCoord<1>, llama::mapping::SoA, llama::mapping::PackedAoS>{
                arrayDomain},
        "Split.SoA.AoS.1Buffer");
}

TEST_CASE("dump.Split.SoA.AoS.2Buffer")
{
    // split out velocity as AoS into separate buffer
    dump(
        llama::mapping::
            Split<ArrayDomain, Particle, llama::DatumCoord<1>, llama::mapping::SoA, llama::mapping::PackedAoS, true>{
                arrayDomain},
        "Split.SoA.AoS.2Buffer");
}

TEST_CASE("dump.Split.AoSoA8.AoS.One.3Buffer")
{
    // split out velocity as AoSoA8 and mass into a single value, the rest as AoS
    dump(
        llama::mapping::Split<
            ArrayDomain,
            Particle,
            llama::DatumCoord<1>,
            llama::mapping::PreconfiguredAoSoA<8>::type,
            llama::mapping::
                PreconfiguredSplit<llama::DatumCoord<1>, llama::mapping::One, llama::mapping::PackedAoS, true>::type,
            true>{arrayDomain},
        "Split.AoSoA8.SoA.One.3Buffer");
}

TEST_CASE("dump.Split.AoSoA8.AoS.One.SoA.4Buffer")
{
    // split out velocity as AoSoA8, mass into a single value, position into AoS, and the flags into SoA, makes 4
    // buffers
    dump(
        llama::mapping::Split<
            ArrayDomain,
            Particle,
            llama::DatumCoord<1>,
            llama::mapping::PreconfiguredAoSoA<8>::type,
            llama::mapping::PreconfiguredSplit<
                llama::DatumCoord<1>,
                llama::mapping::One,
                llama::mapping::
                    PreconfiguredSplit<llama::DatumCoord<0>, llama::mapping::PackedAoS, llama::mapping::SoA, true>::
                        type,
                true>::type,
            true>{arrayDomain},
        "Split.AoSoA8.AoS.One.SoA.4Buffer");
}

TEST_CASE("dump.AoS.Unaligned")
{
    dump(llama::mapping::AoS{arrayDomain, ParticleUnaligned{}}, "AoS.Unaligned");
}

TEST_CASE("dump.AoS.Aligned")
{
    dump(llama::mapping::AoS<decltype(arrayDomain), ParticleUnaligned, true>{arrayDomain}, "AoS.Aligned");
}

TEST_CASE("dump.AoS.AlignedExplicit")
{
    dump(llama::mapping::AoS{arrayDomain, ParticleAligned{}}, "AoS.AlignedExplicit");
}

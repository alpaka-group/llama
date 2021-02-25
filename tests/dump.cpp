#include "common.h"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>
#include <string>

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
    //struct Id {};
    //struct Pad {};
}

using Vec = llama::DS<
    llama::DE<tag::X, float>,
    llama::DE<tag::Y, float>,
    llama::DE<tag::Z, float>
>;
using Particle = llama::DS<
    llama::DE<tag::Pos, Vec>,
    llama::DE<tag::Vel,Vec>,
    llama::DE<tag::Mass, float>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;

// example with bad alignment:
//using Particle = llama::DS<
//    llama::DE<tag::Id, std::uint16_t>,
//    llama::DE<tag::Pos, llama::DS<
//        llama::DE<tag::X, float>,
//        llama::DE<tag::Y, float>
//    >>,
//    llama::DE<tag::Mass, double>,
//    llama::DE<tag::Flags, llama::DA<bool, 3>>
//>;

// bad alignment fixed with explicit padding:
//using Particle = llama::DS<
//    llama::DE<tag::Id, std::uint16_t>,
//    llama::DE<tag::Pad, std::uint16_t>,
//    llama::DE<tag::Pos, llama::DS<
//        llama::DE<tag::X, float>,
//        llama::DE<tag::Y, float>
//    >>,
//    llama::DE<tag::Pad, float>,
//    llama::DE<tag::Mass, double>,
//    llama::DE<tag::Flags, llama::DA<bool, 3>>,
//    llama::DE<tag::Pad, llama::DA<bool, 5>>
//>;
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

TEST_CASE("dump.SplitMapping.SoA.AoS.1Buffer")
{
    // split out velocity (written in nbody, the rest is read)
    dump(
        llama::mapping::
            SplitMapping<ArrayDomain, Particle, llama::DatumCoord<1>, llama::mapping::SoA, llama::mapping::AoS>{
                arrayDomain},
        "SplitMapping.SoA.AoS.1Buffer");
}

TEST_CASE("dump.SplitMapping.SoA.AoS.2Buffer")
{
    // split out velocity as AoS into separate buffer
    dump(
        llama::mapping::
            SplitMapping<ArrayDomain, Particle, llama::DatumCoord<1>, llama::mapping::SoA, llama::mapping::AoS, true>{
                arrayDomain},
        "SplitMapping.SoA.AoS.2Buffer");
}

TEST_CASE("dump.SplitMapping.AoSoA8.AoS.One.3Buffer")
{
    // split out velocity as AoSoA8 and mass into a single value, the rest as AoS
    dump(
        llama::mapping::SplitMapping<
            ArrayDomain,
            Particle,
            llama::DatumCoord<1>,
            llama::mapping::PreconfiguredAoSoA<8>::type,
            llama::mapping::
                PreconfiguredSplitMapping<llama::DatumCoord<0>, llama::mapping::AoS, llama::mapping::One, true>::type,
            true>{arrayDomain},
        "SplitMapping.AoSoA8.SoA.One.3Buffer");
}

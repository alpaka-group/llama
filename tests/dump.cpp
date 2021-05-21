#include "common.h"

#include <catch2/catch.hpp>
#include <fmt/core.h>
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
    struct Id {};
    struct Pad {};
} // namespace tag

using Vec = llama::Record<
    llama::Field<tag::X, float>,
    llama::Field<tag::Y, float>,
    llama::Field<tag::Z, float>
>;
//using DVec = llama::Record<
//    llama::Field<tag::X, double>,
//    llama::Field<tag::Y, double>,
//    llama::Field<tag::Z, double>
//>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec>,
    llama::Field<tag::Vel, /*DVec*/Vec>,
    llama::Field<tag::Mass, float>,
    llama::Field<tag::Flags, bool[4]>
>;
// clang-format on

namespace
{
    llama::ArrayDims arrayDims{32};
    using ArrayDims = decltype(arrayDims);

    template <typename Mapping>
    void dump(const Mapping& mapping)
    {
        // undocumented Catch feature, see: https://github.com/catchorg/Catch2/issues/510
        const auto filename = Catch::getResultCapture().getCurrentTestName();
        std::ofstream{filename + ".svg"} << llama::toSvg(mapping);
        std::ofstream{filename + ".html"} << llama::toHtml(mapping);
    }
} // namespace

TEST_CASE("dump.Particle.AoS")
{
    dump(llama::mapping::AoS{arrayDims, Particle{}});
}

TEST_CASE("dump.Particle.SoA")
{
    dump(llama::mapping::SoA{arrayDims, Particle{}});
}

TEST_CASE("dump.Particle.SoA_MB")
{
    dump(llama::mapping::SoA<ArrayDims, Particle, true>{arrayDims});
}

TEST_CASE("dump.Particle.AoSoA8")
{
    dump(llama::mapping::AoSoA<ArrayDims, Particle, 8>{arrayDims});
}

TEST_CASE("dump.Particle.AoSoA32")
{
    dump(llama::mapping::AoSoA<ArrayDims, Particle, 32>{arrayDims});
}

TEST_CASE("dump.Particle.Split.SoA.AoS.1Buffer")
{
    // split out velocity (written in nbody, the rest is read)
    dump(
        llama::mapping::
            Split<ArrayDims, Particle, llama::RecordCoord<1>, llama::mapping::SingleBlobSoA, llama::mapping::PackedAoS>{
                arrayDims});
}

TEST_CASE("dump.Particle.Split.SoA.AoS.2Buffer")
{
    // split out velocity as AoS into separate buffer
    dump(llama::mapping::Split<
         ArrayDims,
         Particle,
         llama::RecordCoord<1>,
         llama::mapping::SingleBlobSoA,
         llama::mapping::PackedAoS,
         true>{arrayDims});
}

TEST_CASE("dump.Particle.Split.AoSoA8.AoS.One")
{
    // split out velocity as AoSoA8 and mass into a single value, the rest as AoS
    dump(llama::mapping::Split<
         ArrayDims,
         Particle,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::
             PreconfiguredSplit<llama::RecordCoord<1>, llama::mapping::One, llama::mapping::PackedAoS, true>::type,
         true>{arrayDims});
}

TEST_CASE("dump.Particle.Split.AoSoA8.AoS.One.SoA")
{
    // split out velocity as AoSoA8, mass into a single value, position into AoS, and the flags into SoA, makes 4
    // buffers
    dump(llama::mapping::Split<
         ArrayDims,
         Particle,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::PreconfiguredSplit<
             llama::RecordCoord<1>,
             llama::mapping::One,
             llama::mapping::PreconfiguredSplit<
                 llama::RecordCoord<0>,
                 llama::mapping::PackedAoS,
                 llama::mapping::SingleBlobSoA,
                 true>::type,
             true>::type,
         true>{arrayDims});
}

// clang-format off
using Vec2 = llama::Record<
    llama::Field<tag::X, float>,
    llama::Field<tag::Y, float>
>;
using ParticleUnaligned = llama::Record<
    llama::Field<tag::Id, std::uint16_t>,
    llama::Field<tag::Pos, Vec2>,
    llama::Field<tag::Mass, double>,
    llama::Field<tag::Flags, bool[3]>
>;
// clang-format on

TEST_CASE("dump.ParticleUnaligned.AoS")
{
    dump(llama::mapping::AoS{arrayDims, ParticleUnaligned{}});
}

TEST_CASE("dump.ParticleUnaligned.AoS_Aligned")
{
    dump(llama::mapping::AoS<ArrayDims, ParticleUnaligned, true>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.SoA")
{
    dump(llama::mapping::SoA{arrayDims, ParticleUnaligned{}});
}

TEST_CASE("dump.ParticleUnaligned.SoA_MB")
{
    dump(llama::mapping::SoA<ArrayDims, ParticleUnaligned, true>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.AoSoA8")
{
    dump(llama::mapping::AoSoA<ArrayDims, ParticleUnaligned, 8>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.AoSoA32")
{
    dump(llama::mapping::AoSoA<ArrayDims, ParticleUnaligned, 32>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA.AoS.1Buffer")
{
    dump(llama::mapping::Split<
         ArrayDims,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::SingleBlobSoA,
         llama::mapping::PackedAoS>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA.AoS.2Buffer")
{
    dump(llama::mapping::Split<
         ArrayDims,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::SingleBlobSoA,
         llama::mapping::PackedAoS,
         true>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA_MB.AoS_Aligned.One")
{
    dump(llama::mapping::Split<
         ArrayDims,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredSoA<true>::type,
         llama::mapping::
             PreconfiguredSplit<llama::RecordCoord<1>, llama::mapping::One, llama::mapping::AlignedAoS, true>::type,
         true>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.Split.AoSoA8.AoS_Aligned.One")
{
    dump(llama::mapping::Split<
         ArrayDims,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::
             PreconfiguredSplit<llama::RecordCoord<1>, llama::mapping::One, llama::mapping::AlignedAoS, true>::type,
         true>{arrayDims});
}

TEST_CASE("dump.ParticleUnaligned.Split.AoSoA8.SoA.One.AoS")
{
    // split out velocity as AoSoA8 and mass into a single value, the rest as AoS
    dump(llama::mapping::Split<
         ArrayDims,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::PreconfiguredSplit<
             llama::RecordCoord<1>,
             llama::mapping::One,
             llama::mapping::PreconfiguredSplit<
                 llama::RecordCoord<0>,
                 llama::mapping::PreconfiguredSoA<>::type,
                 llama::mapping::PackedAoS,
                 true>::type,
             true>::type,
         true>{arrayDims});
}

TEST_CASE("AoS.Aligned")
{
    const auto mapping = llama::mapping::AoS<ArrayDims, ParticleUnaligned, true>{arrayDims};
    auto view = llama::allocView(mapping);
    llama::forEachLeaf<ParticleUnaligned>(
        [&](auto rc)
        {
            llama::forEachADCoord(
                arrayDims,
                [&](auto ac)
                {
                    const auto addr = &view(ac)(rc);
                    INFO(fmt::format(
                        "address {}, type {}, array dim {}",
                        static_cast<void*>(addr),
                        llama::structName(*addr),
                        ac[0]));
                    CHECK(reinterpret_cast<std::intptr_t>(addr) % sizeof(*addr) == 0);
                });
        });
}

namespace
{
    template <std::size_t N>
    using Padding = std::array<std::byte, N>;
} // namespace

// bad alignment fixed with explicit padding:
// clang-format off
using ParticleAligned = llama::Record<
    llama::Field<tag::Id, std::uint16_t>,
    llama::Field<tag::Pad, Padding<2>>,
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, float>
    >>,
    llama::Field<tag::Pad, Padding<4>>,
    llama::Field<tag::Mass, double>,
    llama::Field<tag::Flags, bool[3]>,
    llama::Field<tag::Pad, Padding<5>>
>;
// clang-format on

TEST_CASE("dump.ParticleAligned.AoS")
{
    dump(llama::mapping::AoS{arrayDims, ParticleAligned{}});
}

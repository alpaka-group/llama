#include "common.hpp"

#include <fmt/core.h>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <string>

// AppleClang 13.0 on MacOS 11.0 crashes (segfault) when compiling any of these tests
#if !(defined(__APPLE__) && __clang_major__ == 13 && __clang_minor__ == 0)
namespace
{
    llama::ArrayExtentsDynamic<1> extents{32};
    using ArrayExtents = decltype(extents);

    template<typename Mapping>
    void dump(const Mapping& mapping)
    {
        // undocumented Catch feature, see: https://github.com/catchorg/Catch2/issues/510
        const auto filename = Catch::getResultCapture().getCurrentTestName();
        std::ofstream{filename + ".svg"} << llama::toSvg(mapping);
        std::ofstream{filename + ".html"} << llama::toHtml(mapping);
    }
} // namespace

TEST_CASE("dump.Particle.AoS_Aligned")
{
    dump(llama::mapping::AlignedAoS<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Particle.AoS_Packed")
{
    dump(llama::mapping::PackedAoS<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Particle.SoA_SB")
{
    dump(llama::mapping::SingleBlobSoA<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Particle.SoA_MB")
{
    dump(llama::mapping::MultiBlobSoA<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Particle.AoSoA8")
{
    dump(llama::mapping::AoSoA<ArrayExtents, Particle, 8>{extents});
}

TEST_CASE("dump.Particle.AoSoA32")
{
    dump(llama::mapping::AoSoA<ArrayExtents, Particle, 32>{extents});
}

TEST_CASE("dump.Particle.Split.SoA.AoS.1Buffer")
{
    // split out velocity (written in nbody, the rest is read)
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         llama::RecordCoord<2>,
         llama::mapping::SingleBlobSoA,
         llama::mapping::PackedAoS>{extents});
}

TEST_CASE("dump.Particle.Split.SoA.AoS.2Buffer")
{
    // split out velocity as AoS into separate buffer
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         llama::RecordCoord<2>,
         llama::mapping::SingleBlobSoA,
         llama::mapping::PackedAoS,
         true>{extents});
}

TEST_CASE("dump.Particle.Split.AoSoA8.AoS.One")
{
    // split out velocity as AoSoA8 and mass into a single value, the rest as AoS
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         llama::RecordCoord<2>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::
             PreconfiguredSplit<llama::RecordCoord<1>, llama::mapping::PackedOne, llama::mapping::PackedAoS, true>::
                 type,
         true>{extents});
}

TEST_CASE("dump.Particle.Split.AoSoA8.AoS.One.SoA")
{
    // split out velocity as AoSoA8, mass into a single value, position into AoS, and the flags into SoA, makes 4
    // buffers
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         llama::RecordCoord<2>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::PreconfiguredSplit<
             llama::RecordCoord<1>,
             llama::mapping::PackedOne,
             llama::mapping::PreconfiguredSplit<
                 llama::RecordCoord<0>,
                 llama::mapping::PackedAoS,
                 llama::mapping::SingleBlobSoA,
                 true>::type,
             true>::type,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.AoS")
{
    dump(llama::mapping::PackedAoS<ArrayExtents, ParticleUnaligned>{extents});
}

TEST_CASE("dump.ParticleUnaligned.AoS_Aligned")
{
    dump(llama::mapping::AlignedAoS<ArrayExtents, ParticleUnaligned>{extents});
}

TEST_CASE("dump.ParticleUnaligned.AoS_Aligned_Min")
{
    dump(llama::mapping::MinAlignedAoS<ArrayExtents, ParticleUnaligned>{extents});
}

TEST_CASE("dump.ParticleUnaligned.SoA_SB")
{
    dump(llama::mapping::SingleBlobSoA<ArrayExtents, ParticleUnaligned>{extents});
}

TEST_CASE("dump.ParticleUnaligned.SoA_MB")
{
    dump(llama::mapping::MultiBlobSoA<ArrayExtents, ParticleUnaligned>{extents});
}

TEST_CASE("dump.ParticleUnaligned.AoSoA4")
{
    dump(llama::mapping::AoSoA<ArrayExtents, ParticleUnaligned, 4>{extents});
}

TEST_CASE("dump.ParticleUnaligned.AoSoA8")
{
    dump(llama::mapping::AoSoA<ArrayExtents, ParticleUnaligned, 8>{extents});
}

TEST_CASE("dump.ParticleUnaligned.AoSoA8_27elements")
{
    dump(llama::mapping::AoSoA<ArrayExtents, ParticleUnaligned, 8>{{27}});
}

TEST_CASE("dump.ParticleUnaligned.AoSoA32")
{
    dump(llama::mapping::AoSoA<ArrayExtents, ParticleUnaligned, 32>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA.AoS.1Buffer")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::SingleBlobSoA,
         llama::mapping::PackedAoS>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA.AoS.2Buffer")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::SingleBlobSoA,
         llama::mapping::PackedAoS,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA_MB.AoS_Aligned.One")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredSoA<true>::type,
         llama::mapping::
             PreconfiguredSplit<llama::RecordCoord<1>, llama::mapping::PackedOne, llama::mapping::AlignedAoS, true>::
                 type,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.AoSoA8.AoS_Aligned.One")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::
             PreconfiguredSplit<llama::RecordCoord<1>, llama::mapping::PackedOne, llama::mapping::AlignedAoS, true>::
                 type,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.AoSoA8.SoA.One.AoS")
{
    // split out velocity as AoSoA8 and mass into a single value, the rest as AoS
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::PreconfiguredAoSoA<8>::type,
         llama::mapping::PreconfiguredSplit<
             llama::RecordCoord<1>,
             llama::mapping::PackedOne,
             llama::mapping::PreconfiguredSplit<
                 llama::RecordCoord<0>,
                 llama::mapping::PreconfiguredSoA<>::type,
                 llama::mapping::PackedAoS,
                 true>::type,
             true>::type,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.Multilist.SoA.One")
{
    // split out Pos and Vel into SoA, the rest into One
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         boost::mp11::mp_list<llama::RecordCoord<0>, llama::RecordCoord<2>>,
         llama::mapping::PreconfiguredSoA<>::type,
         llama::mapping::AlignedOne,
         true>{extents});
}

TEST_CASE("AoS.Aligned")
{
    const auto mapping = llama::mapping::AlignedAoS<ArrayExtents, ParticleUnaligned>{extents};
    auto view = llama::allocView(mapping);
    llama::forEachLeafCoord<ParticleUnaligned>(
        [&](auto rc)
        {
            llama::forEachADCoord(
                extents,
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
    template<std::size_t N>
    using Padding = std::array<std::byte, N>;
} // namespace

// clang-format off
namespace tag
{
    struct Pad {};
} // namespace tag

// bad alignment fixed with explicit padding:
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

TEST_CASE("dump.ParticleAligned.PackedAoS")
{
    dump(llama::mapping::PackedAoS<ArrayExtents, ParticleAligned>{extents});
}
#endif

// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "common.hpp"

#include <filesystem>
#include <fstream>
#include <string>

// AppleClang 13.0 and clang 11.1 crash (segfault) when compiling any of these tests
#if !(defined(__APPLE__) && __clang_major__ == 13 && __clang_minor__ == 0)                                            \
    && !(defined(__clang__) && __clang_major__ == 11 && __clang_minor__ == 1)
namespace
{
    llama::ArrayExtentsDynamic<std::size_t, 1> extents{32};
    using ArrayExtents = decltype(extents);

    template<typename Mapping>
    void dump(const Mapping& mapping)
    {
        const auto outputDir = std::string{"dump"};
        std::filesystem::create_directory(outputDir);
        // undocumented Catch feature, see: https://github.com/catchorg/Catch2/issues/510
        const auto filename = outputDir + "/" + Catch::getResultCapture().getCurrentTestName();
        std::ofstream{filename + ".svg"} << llama::toSvg(mapping);
        std::ofstream{filename + ".html"} << llama::toHtml(mapping);
    }
} // namespace

TEST_CASE("dump.different_extents")
{
    auto mapping1 = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, Particle>{{32}};
    auto refSvg = llama::toSvg(mapping1);
    auto refHtml = llama::toHtml(mapping1);

    auto mapping2 = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<int, 1>, Particle>{{32}};
    CHECK(refSvg == llama::toSvg(mapping2));
    CHECK(refHtml == llama::toHtml(mapping2));

    auto mapping3 = llama::mapping::AlignedAoS<llama::ArrayExtents<std::size_t, 32>, Particle>{{}};
    CHECK(refSvg == llama::toSvg(mapping3));
    CHECK(refHtml == llama::toHtml(mapping3));

    auto mapping4 = llama::mapping::AlignedAoS<llama::ArrayExtents<int, 32>, Particle>{{}};
    CHECK(refSvg == llama::toSvg(mapping4));
    CHECK(refHtml == llama::toHtml(mapping4));
}

TEST_CASE("dump.xmlEscape")
{
    CHECK(llama::internal::xmlEscape("position") == "position");
    CHECK(llama::internal::xmlEscape("position<position_pic>") == "position&lt;position_pic&gt;");
    CHECK(llama::internal::xmlEscape("a<b>c&d\"e'f") == "a&lt;b&gt;c&amp;d&quot;e&apos;f");
}

TEST_CASE("dump.int")
{
    dump(llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, int>{{32}});
}

TEST_CASE("dump.Particle.AoS_Aligned")
{
    dump(llama::mapping::AlignedAoS<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Particle.AoS_Packed")
{
    dump(llama::mapping::PackedAoS<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Particle.SoA_SB_Packed")
{
    dump(llama::mapping::PackedSingleBlobSoA<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Particle.SoA_SB_Aligned")
{
    dump(llama::mapping::AlignedSingleBlobSoA<ArrayExtents, Particle>{extents});
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
         llama::mapping::PackedSingleBlobSoA,
         llama::mapping::PackedAoS>{extents});
}

TEST_CASE("dump.Particle.Split.SoA.AoS.2Buffer")
{
    // split out velocity as AoS into separate buffer
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         llama::RecordCoord<2>,
         llama::mapping::PackedSingleBlobSoA,
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
         llama::mapping::BindAoSoA<8>::fn,
         llama::mapping::BindSplit<llama::RecordCoord<1>, llama::mapping::PackedOne, llama::mapping::PackedAoS, true>::
             fn,
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
         llama::mapping::BindAoSoA<8>::fn,
         llama::mapping::BindSplit<
             llama::RecordCoord<1>,
             llama::mapping::PackedOne,
             llama::mapping::BindSplit<
                 llama::RecordCoord<0>,
                 llama::mapping::PackedAoS,
                 llama::mapping::PackedSingleBlobSoA,
                 true>::fn,
             true>::fn,
         true>{extents});
}

TEST_CASE("dump.Particle.BitPacked")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         llama::RecordCoord<0>,
         llama::mapping::
             BindSplit<llama::RecordCoord<0, 2>, llama::mapping::BitPackedFloatSoA, llama::mapping::PackedAoS, true>::
                 fn,
         llama::mapping::BindSplit<
             llama::RecordCoord<0>,
             llama::mapping::PackedOne,
             llama::mapping::BindSplit<
                 llama::RecordCoord<0>,
                 llama::mapping::BitPackedFloatSoA,
                 llama::mapping::BindBitPackedIntSoA<llama::Constant<1>>::fn,
                 true>::fn,
             true>::fn,
         true>{
        std::tuple{std::tuple{extents, 3, 3}, std::tuple{extents}},
        std::tuple{std::tuple{}, std::tuple{std::tuple{extents, 5, 5}, std::tuple{extents}}}});
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

TEST_CASE("dump.ParticleUnaligned.SoA_SB_Packed")
{
    dump(llama::mapping::PackedSingleBlobSoA<ArrayExtents, ParticleUnaligned>{extents});
}

TEST_CASE("dump.ParticleUnaligned.SoA_SB_Aligned")
{
    dump(llama::mapping::AlignedSingleBlobSoA<ArrayExtents, ParticleUnaligned>{extents});
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
         llama::mapping::PackedSingleBlobSoA,
         llama::mapping::PackedAoS>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA.AoS.2Buffer")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::PackedSingleBlobSoA,
         llama::mapping::PackedAoS,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.SoA_MB.AoS_Aligned.One")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::BindSoA<llama::mapping::Blobs::OnePerField>::fn,
         llama::mapping::
             BindSplit<llama::RecordCoord<1>, llama::mapping::PackedOne, llama::mapping::AlignedAoS, true>::fn,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.AoSoA8.AoS_Aligned.One")
{
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::BindAoSoA<8>::fn,
         llama::mapping::
             BindSplit<llama::RecordCoord<1>, llama::mapping::PackedOne, llama::mapping::AlignedAoS, true>::fn,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.AoSoA8.SoA.One.AoS")
{
    // split out velocity as AoSoA8 and mass into a single value, the rest as AoS
    dump(llama::mapping::Split<
         ArrayExtents,
         ParticleUnaligned,
         llama::RecordCoord<1>,
         llama::mapping::BindAoSoA<8>::fn,
         llama::mapping::BindSplit<
             llama::RecordCoord<1>,
             llama::mapping::PackedOne,
             llama::mapping::
                 BindSplit<llama::RecordCoord<0>, llama::mapping::BindSoA<>::fn, llama::mapping::PackedAoS, true>::fn,
             true>::fn,
         true>{extents});
}

TEST_CASE("dump.ParticleUnaligned.Split.Multilist.SoA.One")
{
    // split out Pos and Vel into SoA, the rest into One
    dump(llama::mapping::Split<
         ArrayExtents,
         Particle,
         mp_list<llama::RecordCoord<0>, llama::RecordCoord<2>>,
         llama::mapping::BindSoA<>::fn,
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
            llama::forEachArrayIndex(
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

TEST_CASE("dump.Particle.ModulusMapping.8")
{
    dump(ModulusMapping<ArrayExtents, Particle, 8>{extents});
}

TEST_CASE("dump.Particle.MapEverythingToZero")
{
    dump(MapEverythingToZero<ArrayExtents, Particle>{extents});
}

TEST_CASE("dump.Triangle.TriangleAoSWithComputedNormal")
{
    dump(TriangleAoSWithComputedNormal<ArrayExtents, Triangle>{extents});
}

TEST_CASE("dump.picongpu.frame.256")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    using Mapping = llama::mapping::
        SoA<ArrayExtents, picongpu::Frame, llama::mapping::Blobs::Single, llama::mapping::SubArrayAlignment::Align>;
    auto mapping = Mapping{ArrayExtents{256}};
    dump(mapping);
}

TEST_CASE("dump.picongpu.frame_openPMD.25")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    using Mapping = llama::mapping::SoA<
        ArrayExtents,
        picongpu::FrameOpenPMD,
        llama::mapping::Blobs::OnePerField,
        llama::mapping::SubArrayAlignment::Pack>;
    auto mapping = Mapping{ArrayExtents{25}};
    dump(mapping);
}

TEST_CASE("dump.AdePT.track")
{
    dump(llama::mapping::AoS<llama::ArrayExtents<int, 8>, Track>{{}});
}

TEST_CASE("dump.LHCb.Custom4")
{
    dump(LhcbCustom4{{3}});
}

TEST_CASE("dump.LHCb.Custom8")
{
    dump(LhcbCustom8{{3}});
}
#endif

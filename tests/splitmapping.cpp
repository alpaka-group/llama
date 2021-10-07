#include "common.hpp"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>

TEST_CASE("Split.SoA_SingleBlob.AoS_Packed.1Buffer")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<2>;
    auto extents = ArrayExtents{16, 16};

    // we layout Pos as SoA, the rest as AoS
    auto mapping = llama::mapping::
        Split<ArrayExtents, Particle, llama::RecordCoord<0>, llama::mapping::SingleBlobSoA, llama::mapping::PackedAoS>{
            extents};

    constexpr auto mapping1Size = 6120;
    const auto ai = llama::ArrayIndex{0, 0};
    CHECK(mapping.blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset{0, 0});
    CHECK(mapping.blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset{0, 2048});
    CHECK(mapping.blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset{0, 4096});
    CHECK(mapping.blobNrAndOffset<1>(ai) == llama::NrAndOffset{0, mapping1Size + 24});
    CHECK(mapping.blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset{0, mapping1Size + 28});
    CHECK(mapping.blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset{0, mapping1Size + 36});
    CHECK(mapping.blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset{0, mapping1Size + 44});
    CHECK(mapping.blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset{0, mapping1Size + 52});
    CHECK(mapping.blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset{0, mapping1Size + 53});
    CHECK(mapping.blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset{0, mapping1Size + 54});
    CHECK(mapping.blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset{0, mapping1Size + 55});
}

TEST_CASE("Split.AoSoA8.AoS_Packed.One.SoA_SingleBlob.4Buffer")
{
    // split out momentum as AoSoA8, mass into a single value, position into AoS, and the flags into SoA, makes 4
    // buffers
    using ArrayExtents = llama::ArrayExtentsDynamic<1>;
    auto extents = ArrayExtents{32};
    auto mapping = llama::mapping::Split<
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
        true>{extents};

    CHECK(mapping.blobNrAndOffset<0, 0>({0}) == llama::NrAndOffset{2, 0});
    CHECK(mapping.blobNrAndOffset<0, 1>({0}) == llama::NrAndOffset{2, 8});
    CHECK(mapping.blobNrAndOffset<0, 2>({0}) == llama::NrAndOffset{2, 16});
    CHECK(mapping.blobNrAndOffset<1>({0}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({0}) == llama::NrAndOffset{0, 0});
    CHECK(mapping.blobNrAndOffset<2, 1>({0}) == llama::NrAndOffset{0, 64});
    CHECK(mapping.blobNrAndOffset<2, 2>({0}) == llama::NrAndOffset{0, 128});
    CHECK(mapping.blobNrAndOffset<3, 0>({0}) == llama::NrAndOffset{3, 0});
    CHECK(mapping.blobNrAndOffset<3, 1>({0}) == llama::NrAndOffset{3, 32});
    CHECK(mapping.blobNrAndOffset<3, 2>({0}) == llama::NrAndOffset{3, 64});
    CHECK(mapping.blobNrAndOffset<3, 3>({0}) == llama::NrAndOffset{3, 96});

    CHECK(mapping.blobNrAndOffset<0, 0>({31}) == llama::NrAndOffset{2, 744});
    CHECK(mapping.blobNrAndOffset<0, 1>({31}) == llama::NrAndOffset{2, 752});
    CHECK(mapping.blobNrAndOffset<0, 2>({31}) == llama::NrAndOffset{2, 760});
    CHECK(mapping.blobNrAndOffset<1>({31}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({31}) == llama::NrAndOffset{0, 632});
    CHECK(mapping.blobNrAndOffset<2, 1>({31}) == llama::NrAndOffset{0, 696});
    CHECK(mapping.blobNrAndOffset<2, 2>({31}) == llama::NrAndOffset{0, 760});
    CHECK(mapping.blobNrAndOffset<3, 0>({31}) == llama::NrAndOffset{3, 31});
    CHECK(mapping.blobNrAndOffset<3, 1>({31}) == llama::NrAndOffset{3, 63});
    CHECK(mapping.blobNrAndOffset<3, 2>({31}) == llama::NrAndOffset{3, 95});
    CHECK(mapping.blobNrAndOffset<3, 3>({31}) == llama::NrAndOffset{3, 127});

    // std::ofstream{"Split.AoSoA8.AoS.One.SoA.4Buffer.svg"} << llama::toSvg(mapping);
}

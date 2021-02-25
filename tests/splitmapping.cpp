#include "common.h"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Flags {};
}

// clang-format off
using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Weight, float>,
    llama::DE<tag::Momentum, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;
// clang-format on

TEST_CASE("splitmapping.SoA.AoS.1Buffer")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};

    // we layout Pos as SoA, the rest as AoS
    auto mapping = llama::mapping::
        SplitMapping<ArrayDomain, Particle, llama::DatumCoord<0>, llama::mapping::SoA, llama::mapping::AoS>{
            arrayDomain};

    constexpr auto mapping1Size = 6120;
    const auto coord = ArrayDomain{0, 0};
    CHECK(mapping.getBlobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 0});
    CHECK(mapping.getBlobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{0, 2048});
    CHECK(mapping.getBlobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{0, 4096});
    CHECK(mapping.getBlobNrAndOffset<1>(coord) == llama::NrAndOffset{0, mapping1Size + 24});
    CHECK(mapping.getBlobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{0, mapping1Size + 28});
    CHECK(mapping.getBlobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{0, mapping1Size + 36});
    CHECK(mapping.getBlobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{0, mapping1Size + 44});
    CHECK(mapping.getBlobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{0, mapping1Size + 52});
    CHECK(mapping.getBlobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{0, mapping1Size + 53});
    CHECK(mapping.getBlobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{0, mapping1Size + 54});
    CHECK(mapping.getBlobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{0, mapping1Size + 55});
}

TEST_CASE("splitmapping.AoSoA8.AoS.One.SoA.4Buffer")
{
    // split out momentum as AoSoA8, mass into a single value, position into AoS, and the flags into SoA, makes 4
    // buffers
    using ArrayDomain = llama::ArrayDomain<1>;
    auto arrayDomain = ArrayDomain{32};
    auto mapping = llama::mapping::SplitMapping<
        ArrayDomain,
        Particle,
        llama::DatumCoord<2>,
        llama::mapping::PreconfiguredAoSoA<8>::type,
        llama::mapping::PreconfiguredSplitMapping<
            llama::DatumCoord<1>,
            llama::mapping::One,
            llama::mapping::
                PreconfiguredSplitMapping<llama::DatumCoord<0>, llama::mapping::AoS, llama::mapping::SoA, true>::type,
            true>::type,
        true>{arrayDomain};

    CHECK(mapping.getBlobNrAndOffset<0, 0>({0}) == llama::NrAndOffset{2, 0});
    CHECK(mapping.getBlobNrAndOffset<0, 1>({0}) == llama::NrAndOffset{2, 8});
    CHECK(mapping.getBlobNrAndOffset<0, 2>({0}) == llama::NrAndOffset{2, 16});
    CHECK(mapping.getBlobNrAndOffset<1>({0}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.getBlobNrAndOffset<2, 0>({0}) == llama::NrAndOffset{0, 0});
    CHECK(mapping.getBlobNrAndOffset<2, 1>({0}) == llama::NrAndOffset{0, 64});
    CHECK(mapping.getBlobNrAndOffset<2, 2>({0}) == llama::NrAndOffset{0, 128});
    CHECK(mapping.getBlobNrAndOffset<3, 0>({0}) == llama::NrAndOffset{3, 0});
    CHECK(mapping.getBlobNrAndOffset<3, 1>({0}) == llama::NrAndOffset{3, 32});
    CHECK(mapping.getBlobNrAndOffset<3, 2>({0}) == llama::NrAndOffset{3, 64});
    CHECK(mapping.getBlobNrAndOffset<3, 3>({0}) == llama::NrAndOffset{3, 96});

    CHECK(mapping.getBlobNrAndOffset<0, 0>({31}) == llama::NrAndOffset{2, 744});
    CHECK(mapping.getBlobNrAndOffset<0, 1>({31}) == llama::NrAndOffset{2, 752});
    CHECK(mapping.getBlobNrAndOffset<0, 2>({31}) == llama::NrAndOffset{2, 760});
    CHECK(mapping.getBlobNrAndOffset<1>({31}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.getBlobNrAndOffset<2, 0>({31}) == llama::NrAndOffset{0, 632});
    CHECK(mapping.getBlobNrAndOffset<2, 1>({31}) == llama::NrAndOffset{0, 696});
    CHECK(mapping.getBlobNrAndOffset<2, 2>({31}) == llama::NrAndOffset{0, 760});
    CHECK(mapping.getBlobNrAndOffset<3, 0>({31}) == llama::NrAndOffset{3, 31});
    CHECK(mapping.getBlobNrAndOffset<3, 1>({31}) == llama::NrAndOffset{3, 63});
    CHECK(mapping.getBlobNrAndOffset<3, 2>({31}) == llama::NrAndOffset{3, 95});
    CHECK(mapping.getBlobNrAndOffset<3, 3>({31}) == llama::NrAndOffset{3, 127});

    std::ofstream{"Splitmapping.AoSoA8.AoS.One.SoA.4Buffer.svg"} << llama::toSvg(mapping);
}

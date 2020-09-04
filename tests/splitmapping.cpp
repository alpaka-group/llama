#include "common.h"

#include <catch2/catch.hpp>
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

TEST_CASE("splitmapping")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain {16, 16};

    // we layout Pos as SoA, the rest as AoS
    auto mapping = llama::mapping::
        SplitMapping<ArrayDomain, Particle, llama::DatumCoord<0>, llama::mapping::SoA, llama::mapping::AoS> {
            arrayDomain};

    constexpr auto mapping1Size = 14336;
    const auto coord = ArrayDomain {0, 0};
    CHECK(mapping.getBlobNrAndOffset<0, 0>(coord) == llama::NrAndOffset {0, 0});
    CHECK(mapping.getBlobNrAndOffset<0, 1>(coord) == llama::NrAndOffset {0, 2048});
    CHECK(mapping.getBlobNrAndOffset<0, 2>(coord) == llama::NrAndOffset {0, 4096});
    CHECK(mapping.getBlobNrAndOffset<1>(coord) == llama::NrAndOffset {0, mapping1Size + 24});
    CHECK(mapping.getBlobNrAndOffset<2, 0>(coord) == llama::NrAndOffset {0, mapping1Size + 28});
    CHECK(mapping.getBlobNrAndOffset<2, 1>(coord) == llama::NrAndOffset {0, mapping1Size + 36});
    CHECK(mapping.getBlobNrAndOffset<2, 2>(coord) == llama::NrAndOffset {0, mapping1Size + 44});
    CHECK(mapping.getBlobNrAndOffset<3, 0>(coord) == llama::NrAndOffset {0, mapping1Size + 52});
    CHECK(mapping.getBlobNrAndOffset<3, 1>(coord) == llama::NrAndOffset {0, mapping1Size + 53});
    CHECK(mapping.getBlobNrAndOffset<3, 2>(coord) == llama::NrAndOffset {0, mapping1Size + 54});
    CHECK(mapping.getBlobNrAndOffset<3, 3>(coord) == llama::NrAndOffset {0, mapping1Size + 55});
}

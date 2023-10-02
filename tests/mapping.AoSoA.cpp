// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

TEST_CASE("mapping.maxLanes")
{
    STATIC_REQUIRE(llama::mapping::maxLanes<Particle, 128> == 2);
    STATIC_REQUIRE(llama::mapping::maxLanes<Particle, 256> == 4);
    STATIC_REQUIRE(llama::mapping::maxLanes<Particle, 512> == 8);

    STATIC_REQUIRE(llama::mapping::maxLanes<float, 128> == 4);
    STATIC_REQUIRE(llama::mapping::maxLanes<float, 256> == 8);
    STATIC_REQUIRE(llama::mapping::maxLanes<float, 512> == 16);

    using RecordDim1 = llama::Record<llama::Field<tag::X, std::int8_t>, llama::Field<tag::Y, std::uint8_t>>;
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim1, 128> == 16);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim1, 256> == 32);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim1, 512> == 64);

    using RecordDim2 = llama::Record<llama::Field<tag::X, std::int8_t>, llama::Field<tag::Y, std::int16_t>>;
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim2, 128> == 8);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim2, 256> == 16);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim2, 512> == 32);
}

TEST_CASE("mapping.AoSoA.4.Pack.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping
            = llama::mapping::AoSoA<decltype(arrayExtents), Particle, 4, llama::mapping::FieldAlignment::Pack>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 96);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 112);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 144);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 176);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 208);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 212);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 216);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 220);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 120);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 152);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 184);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 209);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 213);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 217);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 221);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 928);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 960);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 992);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 1008);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 1040);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 1072);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 1104);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1108);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 1112);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 1116);
        }

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 14336);
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.AoSoA.4.Align.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping
            = llama::mapping::AoSoA<decltype(arrayExtents), Particle, 4, llama::mapping::FieldAlignment::Align>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 96);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 128);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 160);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 192);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 224);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 228);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 232);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 236);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 136);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 168);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 200);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 225);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 229);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 233);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 237);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 1024);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1056);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 1088);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 1120);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 1152);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 1184);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 1216);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 1248);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1252);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 1256);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 1260);
        }

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 16384);
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("AoSoA.size_round_up")
{
    using AoSoA = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, Particle, 4>;
    constexpr auto psize = llama::sizeOf<Particle, true>;

    CHECK(AoSoA{{0}}.blobSize(0) == 0 * psize);
    CHECK(AoSoA{{1}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{2}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{3}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{4}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{5}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{6}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{7}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{8}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{9}}.blobSize(0) == 12 * psize);
}

TEST_CASE("AoSoA.address_within_bounds")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 1>;
    using AoSoA = llama::mapping::AoSoA<ArrayExtents, Particle, 4>;

    const auto ad = ArrayExtents{3};
    auto mapping = AoSoA{ad};
    for(auto i : llama::ArrayIndexRange{ad})
        llama::forEachLeafCoord<Particle>([&](auto rc)
                                          { CHECK(mapping.blobNrAndOffset(i, rc).offset < mapping.blobSize(0)); });
}

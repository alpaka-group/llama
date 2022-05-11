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

TEST_CASE("mapping.AoSoA.4.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::AoSoA<decltype(arrayExtents), Particle, 4>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

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
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("AoSoA.size_round_up")
{
    using AoSoA = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<1>, Particle, 4>;
    constexpr auto psize = llama::sizeOf<Particle>;

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
    using ArrayExtents = llama::ArrayExtentsDynamic<1>;
    using AoSoA = llama::mapping::AoSoA<ArrayExtents, Particle, 4>;

    const auto ad = ArrayExtents{3};
    auto mapping = AoSoA{ad};
    for(auto i : llama::ArrayIndexRange{ad})
        llama::forEachLeafCoord<Particle>([&](auto rc)
                                          { CHECK(mapping.blobNrAndOffset(i, rc).offset < mapping.blobSize(0)); });
}

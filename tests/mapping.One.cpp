#include "common.hpp"

TEST_CASE("mapping.One.Packed.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::PackedOne<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobSize(0) == 56);
        for(const auto ai : {ArrayIndex{0, 0}, ArrayIndex{0, 1}, ArrayIndex{1, 0}})
        {
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 44);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 52);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 53);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 54);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 55);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("mapping.One.Aligned.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::AlignedOne<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobSize(0) == 60);
        for(const auto ai : {ArrayIndex{0, 0}, ArrayIndex{0, 1}, ArrayIndex{1, 0}})
        {
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 48);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 57);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 58);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 59);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("mapping.One.MinAligned.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::MinAlignedOne<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobSize(0) == 56);
        for(const auto ai : {ArrayIndex{0, 0}, ArrayIndex{0, 1}, ArrayIndex{1, 0}})
        {
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 48);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 2);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 3);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

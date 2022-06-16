#include "common.hpp"

TEST_CASE("mapping.AoS.Packed.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::PackedAoS<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 14336);

        {
            const auto ai = ArrayIndex{0, 0};
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

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 84);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 92);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 108);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 109);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 110);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 111);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 904);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 912);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 920);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 924);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 932);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 940);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 948);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 949);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 950);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 951);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.AoS.Packed.fortran.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping
            = llama::mapping::PackedAoS<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsFortran>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 14336);

        {
            const auto ai = ArrayIndex{0, 0};
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

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 904);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 912);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 920);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 924);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 932);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 940);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 948);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 949);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 950);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 951);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 84);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 92);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 108);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 109);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 110);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 111);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.AoS.Packed.morton.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping
            = llama::mapping::PackedAoS<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsMorton>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 14336);

        {
            const auto ai = ArrayIndex{0, 0};
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

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 84);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 92);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 108);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 109);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 110);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 111);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 112);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 120);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 128);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 136);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 140);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 148);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 156);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 164);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 165);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 166);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 167);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.AoS.Aligned.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::AlignedAoS<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 16384);

        {
            const auto ai = ArrayIndex{0, 0};
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

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 88);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 96);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 104);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 112);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 120);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 121);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 122);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 123);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 1024);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1032);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 1040);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 1048);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 1056);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 1064);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 1072);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 1080);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1081);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 1082);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 1083);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.AoS.aligned_min.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::MinAlignedAoS<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 14336);

        {
            const auto ai = ArrayIndex{0, 0};
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

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 60);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 88);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 96);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 104);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 57);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 58);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 59);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 904);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 912);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 920);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 900);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 928);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 936);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 944);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 897);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 898);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 899);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

#include "common.hpp"

TEST_CASE("mapping.SoA.SingleBlob.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::SingleBlobSoA<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2048);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4096);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6144);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7168);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9216);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11264);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13312);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13568);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13824);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14080);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2056);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4104);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6148);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7176);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9224);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11272);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13313);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13569);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13825);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14081);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 128);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2176);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4224);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6208);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7296);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9344);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11392);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13328);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13584);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13840);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14096);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.SoA.SingleBlob.fortran.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::
            SingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsFortran>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2048);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4096);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6144);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7168);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9216);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11264);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13312);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13568);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13824);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14080);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 128);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2176);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4224);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6208);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7296);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9344);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11392);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13328);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13584);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13840);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14096);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2056);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4104);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6148);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7176);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9224);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11272);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13313);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13569);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13825);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14081);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.SoA.SingleBlob.morton.address")
{
    struct Value
    {
    };

    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::
            SingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsMorton>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2048);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4096);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6144);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7168);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9216);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11264);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13312);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13568);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13824);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14080);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2056);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4104);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6148);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7176);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9224);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11272);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13313);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13569);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13825);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14081);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2064);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4112);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6152);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7184);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9232);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11280);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13314);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13570);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13826);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14082);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

TEST_CASE("mapping.SoA.MultiBlob.address")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::MultiBlobSoA<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;
        using SizeType = typename Mapping::ArrayExtents::value_type;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset<SizeType>{0, 0});
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset<SizeType>{1, 0});
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset<SizeType>{2, 0});
            CHECK(mapping.template blobNrAndOffset<1>(ai) == llama::NrAndOffset<SizeType>{3, 0});
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset<SizeType>{4, 0});
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset<SizeType>{5, 0});
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset<SizeType>{6, 0});
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset<SizeType>{7, 0});
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset<SizeType>{8, 0});
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset<SizeType>{9, 0});
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset<SizeType>{10, 0});
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset<SizeType>{0, 8});
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset<SizeType>{1, 8});
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset<SizeType>{2, 8});
            CHECK(mapping.template blobNrAndOffset<1>(ai) == llama::NrAndOffset<SizeType>{3, 4});
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset<SizeType>{4, 8});
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset<SizeType>{5, 8});
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset<SizeType>{6, 8});
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset<SizeType>{7, 1});
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset<SizeType>{8, 1});
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset<SizeType>{9, 1});
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset<SizeType>{10, 1});
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset<SizeType>{0, 128});
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset<SizeType>{1, 128});
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset<SizeType>{2, 128});
            CHECK(mapping.template blobNrAndOffset<1>(ai) == llama::NrAndOffset<SizeType>{3, 64});
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset<SizeType>{4, 128});
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset<SizeType>{5, 128});
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset<SizeType>{6, 128});
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset<SizeType>{7, 16});
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset<SizeType>{8, 16});
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset<SizeType>{9, 16});
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset<SizeType>{10, 16});
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{16, 16});
    test(llama::ArrayExtents<int, 16, llama::dyn>{16});
    test(llama::ArrayExtents<int, llama::dyn, 16>{16});
    test(llama::ArrayExtents<int, 16, 16>{});
}

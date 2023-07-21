// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "common.hpp"

TEST_CASE("mapping.SoA.SingleBlob.Packed.address")
{
    auto test = [](auto arrayExtents)
    {
        CAPTURE(arrayExtents);

        using Mapping = llama::mapping::PackedSingleBlobSoA<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 10472);

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1496);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 2992);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4488);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5236);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6732);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8228);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9724);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9911);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10098);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10285);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1504);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3000);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4492);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5244);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6740);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8236);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9725);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9912);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10099);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10286);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 136);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1632);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3128);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4556);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5372);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6868);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8364);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9741);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9928);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10115);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10302);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{11, 17});
    test(llama::ArrayExtents<int, 11, llama::dyn>{17});
    test(llama::ArrayExtents<int, llama::dyn, 17>{11});
    test(llama::ArrayExtents<int, 11, 17>{});
}

TEST_CASE("mapping.SoA.SingleBlob.Packed.fortran.address")
{
    auto test = [](auto arrayExtents)
    {
        CAPTURE(arrayExtents);

        using Mapping = llama::mapping::
            PackedSingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsFortran>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 10472);

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1496);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 2992);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4488);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5236);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6732);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8228);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9724);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9911);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10098);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10285);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 88);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1584);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3080);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4532);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5324);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6820);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8316);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9735);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9922);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10109);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10296);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1504);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3000);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4492);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5244);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6740);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8236);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9725);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9912);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10099);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10286);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{11, 17});
    test(llama::ArrayExtents<int, 11, llama::dyn>{17});
    test(llama::ArrayExtents<int, llama::dyn, 17>{11});
    test(llama::ArrayExtents<int, 11, 17>{});
}

TEST_CASE("mapping.SoA.SingleBlob.Packed.morton.address")
{
    auto test = [](auto arrayExtents)
    {
        CAPTURE(arrayExtents);

        using Mapping = llama::mapping::
            PackedSingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsMorton>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 57344);

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8192);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16384);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24576);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28672);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36864);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 45056);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 53248);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 54272);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 55296);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 56320);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8200);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16392);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24580);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28680);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36872);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 45064);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 53249);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 54273);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 55297);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 56321);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8208);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16400);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24584);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28688);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36880);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 45072);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 53250);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 54274);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 55298);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 56322);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{11, 17});
    test(llama::ArrayExtents<int, 11, llama::dyn>{17});
    test(llama::ArrayExtents<int, llama::dyn, 17>{11});
    test(llama::ArrayExtents<int, 11, 17>{});
}

TEST_CASE("mapping.SoA.SingleBlob.Aligned.address")
{
    auto test = [](auto arrayExtents)
    {
        CAPTURE(arrayExtents);

        using Mapping = llama::mapping::AlignedSingleBlobSoA<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 10472 + 4);

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1496);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 2992);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4488);
            // 4 bytes alignment inserted after float block
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5236 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6732 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8228 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9724 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9911 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10098 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10285 + 4);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1504);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3000);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4492);
            // 4 bytes alignment inserted after float block
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5244 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6740 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8236 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9725 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9912 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10099 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10286 + 4);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 136);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1632);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3128);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4556);
            // 4 bytes alignment inserted after float block
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5372 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6868 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8364 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9741 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9928 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10115 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10302 + 4);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{11, 17});
    test(llama::ArrayExtents<int, 11, llama::dyn>{17});
    test(llama::ArrayExtents<int, llama::dyn, 17>{11});
    test(llama::ArrayExtents<int, 11, 17>{});
}

TEST_CASE("mapping.SoA.SingleBlob.Aligned.fortran.address")
{
    auto test = [](auto arrayExtents)
    {
        CAPTURE(arrayExtents);

        using Mapping = llama::mapping::
            AlignedSingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsFortran>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 10472 + 4);

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1496);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 2992);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4488);
            // 4 bytes alignment inserted after float block
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5236 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6732 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8228 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9724 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9911 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10098 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10285 + 4);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 88);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1584);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3080);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4532);
            // 4 bytes alignment inserted after float block
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5324 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6820 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8316 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9735 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9922 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10109 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10296 + 4);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1504);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 3000);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4492);
            // 4 bytes alignment inserted after float block
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 5244 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 6740 + 4);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 8236 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 9725 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 9912 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 10099 + 4);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 10286 + 4);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{11, 17});
    test(llama::ArrayExtents<int, 11, llama::dyn>{17});
    test(llama::ArrayExtents<int, llama::dyn, 17>{11});
    test(llama::ArrayExtents<int, 11, 17>{});
}

TEST_CASE("mapping.SoA.SingleBlob.Aligned.morton.address")
{
    auto test = [](auto arrayExtents)
    {
        CAPTURE(arrayExtents);

        using Mapping = llama::mapping::
            AlignedSingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsMorton>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

        STATIC_REQUIRE(mapping.blobCount == 1);
        CHECK(mapping.blobSize(0) == 57344);

        // the morton linearizer needs to round up the flat array extent to a square number, so no addition padding is
        // created

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8192);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16384);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24576);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28672);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36864);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 45056);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 53248);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 54272);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 55296);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 56320);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8200);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16392);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24580);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28680);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36872);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 45064);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 53249);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 54273);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 55297);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 56321);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8208);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16400);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24584);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28688);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36880);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 45072);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 53250);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 54274);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 55298);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 56322);
        }
    };
    test(llama::ArrayExtentsDynamic<std::size_t, 2>{11, 17});
    test(llama::ArrayExtents<int, 11, llama::dyn>{17});
    test(llama::ArrayExtents<int, llama::dyn, 17>{11});
    test(llama::ArrayExtents<int, 11, 17>{});
}


TEST_CASE("mapping.SoA.MultiBlob.address")
{
    auto test = [](auto arrayExtents)
    {
        CAPTURE(arrayExtents);

        using Mapping = llama::mapping::MultiBlobSoA<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayExtents::Index;
        using SizeType = typename Mapping::ArrayExtents::value_type;

        STATIC_REQUIRE(mapping.blobCount == 11);
        CHECK(mapping.blobSize(0) == 2048);
        CHECK(mapping.blobSize(1) == 2048);
        CHECK(mapping.blobSize(2) == 2048);
        CHECK(mapping.blobSize(3) == 1024);
        CHECK(mapping.blobSize(4) == 2048);
        CHECK(mapping.blobSize(5) == 2048);
        CHECK(mapping.blobSize(6) == 2048);
        CHECK(mapping.blobSize(7) == 256);
        CHECK(mapping.blobSize(8) == 256);
        CHECK(mapping.blobSize(9) == 256);
        CHECK(mapping.blobSize(10) == 256);

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

#include "common.hpp"

#include <catch2/catch.hpp>
#include <llama/Concepts.hpp>
#include <llama/llama.hpp>

#ifdef __cpp_lib_concepts
TEST_CASE("mapping.concepts")
{
    STATIC_REQUIRE(llama::Mapping<llama::mapping::AlignedAoS<llama::ArrayDims<2>, Particle>>);
    STATIC_REQUIRE(llama::Mapping<llama::mapping::PackedAoS<llama::ArrayDims<2>, Particle>>);
    STATIC_REQUIRE(llama::Mapping<llama::mapping::SingleBlobSoA<llama::ArrayDims<2>, Particle>>);
    STATIC_REQUIRE(llama::Mapping<llama::mapping::MultiBlobSoA<llama::ArrayDims<2>, Particle>>);
    STATIC_REQUIRE(llama::Mapping<llama::mapping::AoSoA<llama::ArrayDims<2>, Particle, 8>>);
}
#endif

TEST_CASE("mapping.traits")
{
    using AAoS = llama::mapping::AlignedAoS<llama::ArrayDims<2>, Particle>;
    using PAoS = llama::mapping::PackedAoS<llama::ArrayDims<2>, Particle>;
    using SBSoA = llama::mapping::SingleBlobSoA<llama::ArrayDims<2>, Particle>;
    using MBSoA = llama::mapping::MultiBlobSoA<llama::ArrayDims<2>, Particle>;
    using AoAoS = llama::mapping::AoSoA<llama::ArrayDims<2>, Particle, 8>;
    using One = llama::mapping::One<llama::ArrayDims<2>, Particle>;

    STATIC_REQUIRE(llama::mapping::isAoS<AAoS>);
    STATIC_REQUIRE(llama::mapping::isAoS<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoS<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoS<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<One>);

    STATIC_REQUIRE(!llama::mapping::isSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<PAoS>);
    STATIC_REQUIRE(llama::mapping::isSoA<SBSoA>);
    STATIC_REQUIRE(llama::mapping::isSoA<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<One>);

    STATIC_REQUIRE(!llama::mapping::isAoSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<MBSoA>);
    STATIC_REQUIRE(llama::mapping::isAoSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<One>);

    STATIC_REQUIRE(!llama::mapping::isOne<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isOne<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isOne<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isOne<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isOne<AoAoS>);
    STATIC_REQUIRE(llama::mapping::isOne<One>);
}

TEST_CASE("address.AoS.Packed")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::PackedAoS<ArrayDims, Particle>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 951);
    }
}

TEST_CASE("address.AoS.Packed.fortran")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping
        = llama::mapping::PackedAoS<ArrayDims, Particle, llama::mapping::LinearizeArrayDimsFortran>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 951);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 111);
    }
}

TEST_CASE("address.AoS.Packed.morton")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::PackedAoS<ArrayDims, Particle, llama::mapping::LinearizeArrayDimsMorton>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 112);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 120);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 136);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 140);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 148);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 156);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 164);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 165);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 166);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 167);
    }
}

TEST_CASE("address.AoS.Aligned")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::AlignedAoS<ArrayDims, Particle>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 40);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 48);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 57);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 58);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 59);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 88);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 96);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 104);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 112);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 120);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 121);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 122);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 123);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 1024);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 1032);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 1040);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 1048);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 1056);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 1064);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 1072);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 1080);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 1081);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 1082);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 1083);
    }
}

TEST_CASE("address.AoS.aligned_min")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::MinAlignedAoS<ArrayDims, Particle>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 4);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 40);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 48);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 1);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 2);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 3);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 80);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 60);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 88);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 96);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 104);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 57);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 58);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 59);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 904);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 912);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 920);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 900);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 928);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 936);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 944);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 897);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 898);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 899);
    }
}

TEST_CASE("address.SoA.SingleBlob")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::SingleBlobSoA<ArrayDims, Particle>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14081);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14096);
    }
}

TEST_CASE("address.SoA.SingleBlob.fortran")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping
        = llama::mapping::SingleBlobSoA<ArrayDims, Particle, llama::mapping::LinearizeArrayDimsFortran>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14096);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14081);
    }
}

TEST_CASE("address.SoA.SingleBlob.morton")
{
    struct Value
    {
    };

    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping
        = llama::mapping::SingleBlobSoA<ArrayDims, Particle, llama::mapping::LinearizeArrayDimsMorton>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14081);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 2064);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 4112);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 6152);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 7184);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 9232);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 11280);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 13314);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 13570);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 13826);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 14082);
    }
}

TEST_CASE("address.SoA.MultiBlob")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::MultiBlobSoA<ArrayDims, Particle>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 0});
        CHECK(mapping.blobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{1, 0});
        CHECK(mapping.blobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{2, 0});
        CHECK(mapping.blobNrAndOffset<1>(coord) == llama::NrAndOffset{3, 0});
        CHECK(mapping.blobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{4, 0});
        CHECK(mapping.blobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{5, 0});
        CHECK(mapping.blobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{6, 0});
        CHECK(mapping.blobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{7, 0});
        CHECK(mapping.blobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{8, 0});
        CHECK(mapping.blobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{9, 0});
        CHECK(mapping.blobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{10, 0});
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 8});
        CHECK(mapping.blobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{1, 8});
        CHECK(mapping.blobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{2, 8});
        CHECK(mapping.blobNrAndOffset<1>(coord) == llama::NrAndOffset{3, 4});
        CHECK(mapping.blobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{4, 8});
        CHECK(mapping.blobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{5, 8});
        CHECK(mapping.blobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{6, 8});
        CHECK(mapping.blobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{7, 1});
        CHECK(mapping.blobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{8, 1});
        CHECK(mapping.blobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{9, 1});
        CHECK(mapping.blobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{10, 1});
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 128});
        CHECK(mapping.blobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{1, 128});
        CHECK(mapping.blobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{2, 128});
        CHECK(mapping.blobNrAndOffset<1>(coord) == llama::NrAndOffset{3, 64});
        CHECK(mapping.blobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{4, 128});
        CHECK(mapping.blobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{5, 128});
        CHECK(mapping.blobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{6, 128});
        CHECK(mapping.blobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{7, 16});
        CHECK(mapping.blobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{8, 16});
        CHECK(mapping.blobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{9, 16});
        CHECK(mapping.blobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{10, 16});
    }
}

TEST_CASE("address.AoSoA.4")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::AoSoA<ArrayDims, Particle, 4>{arrayDims};

    {
        const auto coord = ArrayDims{0, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 64);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 96);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 112);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 144);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 176);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 208);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 212);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 216);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 220);
    }

    {
        const auto coord = ArrayDims{0, 1};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 40);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 100);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 120);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 152);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 184);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 209);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 213);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 217);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 221);
    }

    {
        const auto coord = ArrayDims{1, 0};
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 928);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 960);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 992);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 1008);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 1040);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 1072);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 1104);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 1108);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 1112);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 1116);
    }
}

TEST_CASE("address.PackedOne")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::PackedOne<ArrayDims, Particle>{arrayDims};

    STATIC_REQUIRE(mapping.blobSize(0) == 56);
    for(const auto coord : {ArrayDims{0, 0}, ArrayDims{0, 1}, ArrayDims{1, 0}})
    {
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 55);
    }
}

TEST_CASE("address.AlignedOne")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::AlignedOne<ArrayDims, Particle>{arrayDims};

    STATIC_REQUIRE(mapping.blobSize(0) == 60);
    for(const auto coord : {ArrayDims{0, 0}, ArrayDims{0, 1}, ArrayDims{1, 0}})
    {
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 40);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 48);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 56);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 57);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 58);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 59);
    }
}

TEST_CASE("address.MinAlignedOne")
{
    using ArrayDims = llama::ArrayDims<2>;
    auto arrayDims = ArrayDims{16, 16};
    auto mapping = llama::mapping::MinAlignedOne<ArrayDims, Particle>{arrayDims};

    STATIC_REQUIRE(mapping.blobSize(0) == 56);
    for(const auto coord : {ArrayDims{0, 0}, ArrayDims{0, 1}, ArrayDims{1, 0}})
    {
        CHECK(mapping.blobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.blobNrAndOffset<0, 1>(coord).offset == 16);
        CHECK(mapping.blobNrAndOffset<0, 2>(coord).offset == 24);
        CHECK(mapping.blobNrAndOffset<1>(coord).offset == 4);
        CHECK(mapping.blobNrAndOffset<2, 0>(coord).offset == 32);
        CHECK(mapping.blobNrAndOffset<2, 1>(coord).offset == 40);
        CHECK(mapping.blobNrAndOffset<2, 2>(coord).offset == 48);
        CHECK(mapping.blobNrAndOffset<3, 0>(coord).offset == 0);
        CHECK(mapping.blobNrAndOffset<3, 1>(coord).offset == 1);
        CHECK(mapping.blobNrAndOffset<3, 2>(coord).offset == 2);
        CHECK(mapping.blobNrAndOffset<3, 3>(coord).offset == 3);
    }
}

TEST_CASE("maxLanes")
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

TEST_CASE("AoSoA.size_round_up")
{
    using AoSoA = llama::mapping::AoSoA<llama::ArrayDims<1>, Particle, 4>;
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
    using AD = llama::ArrayDims<1>;
    using AoSoA = llama::mapping::AoSoA<AD, Particle, 4>;

    const auto ad = AD{3};
    auto mapping = AoSoA{ad};
    for(auto i : llama::ArrayDimsIndexRange{ad})
        llama::forEachLeafCoord<Particle>([&](auto rc)
                                          { CHECK(mapping.blobNrAndOffset(i, rc).offset < mapping.blobSize(0)); });
}

TEST_CASE("FlattenRecordDimInOrder")
{
    using F = llama::mapping::FlattenRecordDimInOrder<Particle>;
    STATIC_REQUIRE(
        std::is_same_v<
            F::FlatRecordDim,
            boost::mp11::mp_list<double, double, double, float, double, double, double, bool, bool, bool, bool>>);
    STATIC_REQUIRE(F::flatIndex<0, 0> == 0);
    STATIC_REQUIRE(F::flatIndex<0, 1> == 1);
    STATIC_REQUIRE(F::flatIndex<0, 2> == 2);
    STATIC_REQUIRE(F::flatIndex<1> == 3);
    STATIC_REQUIRE(F::flatIndex<2, 0> == 4);
    STATIC_REQUIRE(F::flatIndex<2, 1> == 5);
    STATIC_REQUIRE(F::flatIndex<2, 2> == 6);
    STATIC_REQUIRE(F::flatIndex<3, 0> == 7);
    STATIC_REQUIRE(F::flatIndex<3, 1> == 8);
    STATIC_REQUIRE(F::flatIndex<3, 2> == 9);
    STATIC_REQUIRE(F::flatIndex<3, 3> == 10);
}

TEST_CASE("FlattenRecordDimIncreasingAlignment")
{
    using F = llama::mapping::FlattenRecordDimIncreasingAlignment<Particle>;
    STATIC_REQUIRE(
        std::is_same_v<
            F::FlatRecordDim,
            boost::mp11::mp_list<bool, bool, bool, bool, float, double, double, double, double, double, double>>);
    STATIC_REQUIRE(F::flatIndex<0, 0> == 5);
    STATIC_REQUIRE(F::flatIndex<0, 1> == 6);
    STATIC_REQUIRE(F::flatIndex<0, 2> == 7);
    STATIC_REQUIRE(F::flatIndex<1> == 4);
    STATIC_REQUIRE(F::flatIndex<2, 0> == 8);
    STATIC_REQUIRE(F::flatIndex<2, 1> == 9);
    STATIC_REQUIRE(F::flatIndex<2, 2> == 10);
    STATIC_REQUIRE(F::flatIndex<3, 0> == 0);
    STATIC_REQUIRE(F::flatIndex<3, 1> == 1);
    STATIC_REQUIRE(F::flatIndex<3, 2> == 2);
    STATIC_REQUIRE(F::flatIndex<3, 3> == 3);
}

TEST_CASE("FlattenRecordDimDecreasingAlignment")
{
    using F = llama::mapping::FlattenRecordDimDecreasingAlignment<Particle>;
    STATIC_REQUIRE(
        std::is_same_v<
            F::FlatRecordDim,
            boost::mp11::mp_list<double, double, double, double, double, double, float, bool, bool, bool, bool>>);
    STATIC_REQUIRE(F::flatIndex<0, 0> == 0);
    STATIC_REQUIRE(F::flatIndex<0, 1> == 1);
    STATIC_REQUIRE(F::flatIndex<0, 2> == 2);
    STATIC_REQUIRE(F::flatIndex<1> == 6);
    STATIC_REQUIRE(F::flatIndex<2, 0> == 3);
    STATIC_REQUIRE(F::flatIndex<2, 1> == 4);
    STATIC_REQUIRE(F::flatIndex<2, 2> == 5);
    STATIC_REQUIRE(F::flatIndex<3, 0> == 7);
    STATIC_REQUIRE(F::flatIndex<3, 1> == 8);
    STATIC_REQUIRE(F::flatIndex<3, 2> == 9);
    STATIC_REQUIRE(F::flatIndex<3, 3> == 10);
}

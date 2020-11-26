#include "common.h"

#include <catch2/catch.hpp>
#include <llama/Concepts.hpp>
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
    llama::DE<tag::Flags, bool[4]>
>;
// clang-format on


#ifdef __cpp_concepts
TEST_CASE("mapping.concepts")
{
    STATIC_REQUIRE(llama::Mapping<llama::mapping::AoS<llama::ArrayDomain<2>, Particle>>);
    STATIC_REQUIRE(llama::Mapping<llama::mapping::SoA<llama::ArrayDomain<2>, Particle>>);
    STATIC_REQUIRE(llama::Mapping<llama::mapping::AoSoA<llama::ArrayDomain<2>, Particle, 8>>);
}
#endif

TEST_CASE("address.AoS")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping = llama::mapping::AoS<ArrayDomain, Particle>{arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 951);
    }
}

TEST_CASE("address.AoS.fortran")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping
        = llama::mapping::AoS<ArrayDomain, Particle, false, llama::mapping::LinearizeArrayDomainFortran>{arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 951);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 111);
    }
}

TEST_CASE("address.AoS.morton")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping
        = llama::mapping::AoS<ArrayDomain, Particle, false, llama::mapping::LinearizeArrayDomainMorton>{arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 112);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 120);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 128);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 136);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 140);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 148);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 156);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 164);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 165);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 166);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 167);
    }
}

TEST_CASE("address.AoS.aligned")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping = llama::mapping::AoS<ArrayDomain, Particle, true>{arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 32);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 40);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 48);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 57);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 58);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 59);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 88);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 96);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 104);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 112);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 120);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 121);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 122);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 123);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 1024);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 1032);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 1040);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 1048);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 1056);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 1064);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 1072);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 1080);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 1081);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 1082);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 1083);
    }
}

TEST_CASE("address.SoA")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping = llama::mapping::SoA<ArrayDomain, Particle>{arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14081);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14096);
    }
}

TEST_CASE("address.SoA.fortran")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping
        = llama::mapping::SoA<ArrayDomain, Particle, std::false_type, llama::mapping::LinearizeArrayDomainFortran>{
            arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14096);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14081);
    }
}

TEST_CASE("address.SoA.morton")
{
    struct Value
    {
    };

    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping
        = llama::mapping::SoA<ArrayDomain, Particle, std::false_type, llama::mapping::LinearizeArrayDomainMorton>{
            arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14081);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2064);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4112);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6152);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7184);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9232);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11280);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13314);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13570);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13826);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14082);
    }
}

TEST_CASE("address.SoA.MultiBlob")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping = llama::mapping::SoA<ArrayDomain, Particle, std::true_type>{arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 0});
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{1, 0});
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{2, 0});
        CHECK(mapping.getBlobNrAndOffset<1>(coord) == llama::NrAndOffset{3, 0});
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{4, 0});
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{5, 0});
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{6, 0});
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{7, 0});
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{8, 0});
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{9, 0});
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{10, 0});
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 8});
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{1, 8});
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{2, 8});
        CHECK(mapping.getBlobNrAndOffset<1>(coord) == llama::NrAndOffset{3, 4});
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{4, 8});
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{5, 8});
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{6, 8});
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{7, 1});
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{8, 1});
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{9, 1});
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{10, 1});
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord) == llama::NrAndOffset{0, 128});
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord) == llama::NrAndOffset{1, 128});
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord) == llama::NrAndOffset{2, 128});
        CHECK(mapping.getBlobNrAndOffset<1>(coord) == llama::NrAndOffset{3, 64});
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord) == llama::NrAndOffset{4, 128});
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord) == llama::NrAndOffset{5, 128});
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord) == llama::NrAndOffset{6, 128});
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord) == llama::NrAndOffset{7, 16});
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord) == llama::NrAndOffset{8, 16});
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord) == llama::NrAndOffset{9, 16});
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord) == llama::NrAndOffset{10, 16});
    }
}

TEST_CASE("address.AoSoA.4")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    auto arrayDomain = ArrayDomain{16, 16};
    auto mapping = llama::mapping::AoSoA<ArrayDomain, Particle, 4>{arrayDomain};

    {
        const auto coord = ArrayDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 32);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 96);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 112);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 144);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 176);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 208);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 212);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 216);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 220);
    }

    {
        const auto coord = ArrayDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 40);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 120);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 152);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 184);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 209);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 213);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 217);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 221);
    }

    {
        const auto coord = ArrayDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 928);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 960);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 992);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 1008);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 1040);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 1072);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 1104);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 1108);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 1112);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 1116);
    }
}

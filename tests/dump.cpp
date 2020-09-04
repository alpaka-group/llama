#include "common.h"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/llama.hpp>
#include <llama/mapping/Dump.hpp>

// clang-format off
namespace tag
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Mass {};
    struct Flags {};
}

using Vec = llama::DS<
    llama::DE<tag::X, float>,
    llama::DE<tag::Y, float>,
    llama::DE<tag::Z, float>
>;
using Particle = llama::DS<
    llama::DE<tag::Pos, Vec>,
    llama::DE<tag::Vel,Vec>,
    llama::DE<tag::Mass, float>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;
// clang-format on

TEST_CASE("dump")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{8, 8};

    {
        std::ofstream f{"AoSMapping.svg"};
        f << llama::mapping::toSvg(
            llama::mapping::AoS<UserDomain, Particle>{userDomain});
    }
    {
        std::ofstream f{"SoAMapping.svg"};
        f << llama::mapping::toSvg(
            llama::mapping::SoA<UserDomain, Particle>{userDomain});
    }
    {
        std::ofstream f{"AoSoAMapping8.svg"};
        f << llama::mapping::toSvg(
            llama::mapping::AoSoA<UserDomain, Particle, 8>{userDomain});
    }
    {
        std::ofstream f{"AoSoAMapping32_cuda.svg"};
        f << llama::mapping::toSvg(
            llama::mapping::AoSoA<UserDomain, Particle, 32>{userDomain});
    }
}

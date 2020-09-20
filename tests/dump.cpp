#include "common.h"

#include <catch2/catch.hpp>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>
#include <string>

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

template<typename Mapping>
void dump(const Mapping & mapping, std::string filename)
{
    std::ofstream{filename + ".svg"} << llama::toSvg(mapping);
    std::ofstream{filename + ".html"} << llama::toHtml(mapping);
}

TEST_CASE("dump")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{8, 8};

    dump(llama::mapping::AoS<UserDomain, Particle>{userDomain}, "AoSMapping");
    dump(llama::mapping::SoA<UserDomain, Particle>{userDomain}, "SoAMapping");
    dump(
        llama::mapping::AoSoA<UserDomain, Particle, 8>{userDomain},
        "AoSoAMapping8");
    dump(
        llama::mapping::AoSoA<UserDomain, Particle, 32>{userDomain},
        "AoSoAMapping32");
}

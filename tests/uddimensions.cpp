#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
}

using Name = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>,
        llama::DE<tag::Z, float>
    >>
>;
// clang-format on

TEST_CASE("dim1")
{
    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UserDomain{0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("dim2")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UserDomain{0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("dim3")
{
    using UserDomain = llama::UserDomain<3>;
    UserDomain userDomain{16, 16, 16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UserDomain{0, 0, 0}).access<tag::Pos, tag::X>();
    x = 0;
}

TEST_CASE("dim10")
{
    using UserDomain = llama::UserDomain<10>;
    UserDomain userDomain{2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UserDomain{0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
                    .access<tag::Pos, tag::X>();
    x = 0;
}

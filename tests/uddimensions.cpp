#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

namespace st
{
    struct Pos
    {};
    struct X
    {};
    struct Y
    {};
    struct Z
    {};
}

using Name = llama::DS<llama::DE<
    st::Pos,
    llama::DS<
        llama::DE<st::X, float>,
        llama::DE<st::Y, float>,
        llama::DE<st::Z, float>>>>;

TEST_CASE("dim1")
{
    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::
        SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UD{0}).access<st::Pos, st::X>();
    x = 0;
}

TEST_CASE("dim2")
{
    using UD = llama::UserDomain<2>;
    UD udSize{16, 16};

    using Mapping = llama::mapping::
        SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UD{0, 0}).access<st::Pos, st::X>();
    x = 0;
}

TEST_CASE("dim3")
{
    using UD = llama::UserDomain<3>;
    UD udSize{16, 16, 16};

    using Mapping = llama::mapping::
        SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UD{0, 0, 0}).access<st::Pos, st::X>();
    x = 0;
}

TEST_CASE("dim10")
{
    using UD = llama::UserDomain<10>;
    UD udSize{2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    using Mapping = llama::mapping::
        SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    float & x = view(UD{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}).access<st::Pos, st::X>();
    x = 0;
}

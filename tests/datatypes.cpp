#include "common.h"

#include <array>
#include <atomic>
#include <catch2/catch.hpp>
#include <complex>
#include <llama/llama.hpp>
#include <vector>

TEST_CASE("type int")
{
    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, int>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    int & e = view(UserDomain{0}).access<Tag>();
    e = 0;
}

TEST_CASE("type std::complex<float>")
{
    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, std::complex<float>>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::complex<float> & e = view(UserDomain{0}).access<Tag>();
    e = {2, 3};
}

TEST_CASE("type std::array<float, 4>")
{
    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, std::array<float, 4>>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::array<float, 4> & e = view(UserDomain{0}).access<Tag>();
    e = {2, 3, 4, 5};
}

TEST_CASE("type std::vector<float>")
{
    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, std::vector<float>>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::vector<float> & e = view(UserDomain{0}).access<Tag>();
    // e = {2, 3, 4, 5}; // FIXME: LLAMA memory is uninitialized
}

TEST_CASE("type std::atomic<int>")
{
    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, std::atomic<int>>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::atomic<int> & e = view(UserDomain{0}).access<Tag>();
    // e++; // FIXME: LLAMA memory is uninitialized
}

TEST_CASE("type noncopyable")
{
    struct Element
    {
        Element() = default;
        Element(const Element &) = delete;
        auto operator=(const Element &) -> Element & = delete;
        Element(Element &&) noexcept = default;
        auto operator=(Element &&) noexcept -> Element & = default;

        int value;
    };

    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, Element>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element & e = view(UserDomain{0}).access<Tag>();
    e.value = 0;
}

TEST_CASE("type nonmoveable")
{
    struct Element
    {
        Element() = default;
        Element(const Element &) = delete;
        auto operator=(const Element &) -> Element & = delete;
        Element(Element &&) noexcept = delete;
        auto operator=(Element &&) noexcept -> Element & = delete;

        int value;
    };

    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, Element>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element & e = view(UserDomain{0}).access<Tag>();
    e.value = 0;
}

TEST_CASE("type not defaultconstructible")
{
    struct Element
    {
        Element() = delete;
        int value;
    };

    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, Element>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element & e = view(UserDomain{0}).access<Tag>();
    e.value = 0;
}

TEST_CASE("type nottrivial ctor")
{
    struct Element
    {
        int value = 42;
    };

    struct Tag
    {};
    using Name = llama::DS<llama::DE<Tag, Element>>;

    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{16};

    using Mapping = llama::mapping::SoA<
        UserDomain,
        Name,
        llama::LinearizeUserDomainAdress<UserDomain::count>>;
    Mapping mapping{userDomain};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element & e = view(UserDomain{0}).access<Tag>();
    // CHECK(e.value == 42); // FIXME: LLAMA memory is uninitialized
}

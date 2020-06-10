#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <complex>
#include <array>
#include <vector>
#include <atomic>
#include "common.h"

TEST_CASE("type int") {
    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, int>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    int& e = view(UD{0}).access<Tag>();
    e = 0;
}

TEST_CASE("type std::complex<float>") {
    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, std::complex<float>>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::complex<float>& e = view(UD{0}).access<Tag>();
    e = {2, 3};
}

TEST_CASE("type std::array<float, 4>") {
    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, std::array<float, 4>>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::array<float, 4>& e = view(UD{0}).access<Tag>();
    e = {2, 3, 4, 5};
}

TEST_CASE("type std::vector<float>") {
    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, std::vector<float>>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::vector<float>& e = view(UD{0}).access<Tag>();
    e = {2, 3, 4, 5};
}

TEST_CASE("type std::atomic<int>") {
    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, std::atomic<int>>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    std::atomic<int>& e = view(UD{0}).access<Tag>();
    e++;
}

TEST_CASE("type noncopyable") {
    struct Element {
        Element() = default;
        Element(const Element&) = delete;
        auto operator=(const Element&) ->Element& = delete;
        Element(Element&&) noexcept = default;
        auto operator=(Element&&) noexcept -> Element& = default;

        int value;
    };

    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, Element>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element& e = view(UD{0}).access<Tag>();
    e.value = 0;
}

TEST_CASE("type nonmoveable") {
    struct Element {
        Element() = default;
        Element(const Element&) = delete;
        auto operator=(const Element&)->Element & = delete;
        Element(Element&&) noexcept = delete;
        auto operator=(Element&&) noexcept -> Element & = delete;

        int value;
    };

    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, Element>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element& e = view(UD{0}).access<Tag>();
    e.value = 0;
}

TEST_CASE("type not defaultconstructible") {
    struct Element {
        Element() = delete;
        int value;
    };

    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, Element>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element& e = view(UD{0}).access<Tag>();
    e.value = 0;
}

TEST_CASE("type nottrivial ctor") {
    struct Element {
        int value = 42;
    };

    struct Tag {};
    using Name = llama::DS<
        llama::DE<Tag, Element>
    >;

    using UD = llama::UserDomain<1>;
    UD udSize{16};

    using Mapping = llama::mapping::SoA<UD, Name, llama::LinearizeUserDomainAdress<UD::count>>;
    Mapping mapping{udSize};

    using Factory = llama::Factory<Mapping, llama::allocator::SharedPtr<>>;
    auto view = Factory::allocView(mapping);

    Element& e = view(UD{0}).access<Tag>();
    CHECK(e.value == 42);
}


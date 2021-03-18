#include <algorithm>
#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <numeric>

// clang-format off
namespace tag {
    struct X {};
    struct Y {};
    struct Z {};
}

using Position = llama::DS<
    llama::DE<tag::X, int>,
    llama::DE<tag::Y, int>,
    llama::DE<tag::Z, int>
>;
// clang-format on

TEST_CASE("iterator")
{
    using ArrayDomain = llama::ArrayDomain<1>;
    constexpr auto arrayDomain = ArrayDomain{32};
    constexpr auto mapping = llama::mapping::AoS<ArrayDomain, Position>{arrayDomain};
    auto view = llama::allocView(mapping);

    for (auto vd : view)
    {
        vd(tag::X{}) = 1;
        vd(tag::Y{}) = 2;
        vd(tag::Z{}) = 3;
    }
    std::transform(begin(view), end(view), begin(view), [](auto vd) { return vd * 2; });
    const int sumY = std::accumulate(begin(view), end(view), 0, [](int acc, auto vd) { return acc + vd(tag::Y{}); });
    CHECK(sumY == 128);
}

TEST_CASE("iterator.std_copy")
{
    using ArrayDomain = llama::ArrayDomain<1>;
    constexpr auto arrayDomain = ArrayDomain{32};
    auto aosView = llama::allocView(llama::mapping::AoS<ArrayDomain, Position>{arrayDomain});
    auto soaView = llama::allocView(llama::mapping::SoA<ArrayDomain, Position>{arrayDomain});

    int i = 0;
    for (auto vd : aosView)
    {
        vd(tag::X{}) = ++i;
        vd(tag::Y{}) = ++i;
        vd(tag::Z{}) = ++i;
    }
    std::copy(begin(aosView), end(aosView), begin(soaView));
    i = 0;
    for (auto vd : soaView)
    {
        CHECK(vd(tag::X{}) == ++i);
        CHECK(vd(tag::Y{}) == ++i);
        CHECK(vd(tag::Z{}) == ++i);
    }
}

TEST_CASE("iterator.transform_reduce")
{
    constexpr auto arrayDomain = llama::ArrayDomain<1>{32};
    auto aosView = llama::allocView(llama::mapping::AoS{arrayDomain, Position{}});
    auto soaView = llama::allocView(llama::mapping::SoA{arrayDomain, Position{}});

    int i = 0;
    for (auto vd : aosView)
    {
        vd(tag::X{}) = ++i;
        vd(tag::Y{}) = ++i;
        vd(tag::Z{}) = ++i;
    }
    for (auto vd : soaView)
    {
        vd(tag::X{}) = ++i;
        vd(tag::Y{}) = ++i;
        vd(tag::Z{}) = ++i;
    }
    // returned type is a llama::One<Particle>
    auto [sumX, sumY, sumZ] = std::transform_reduce(begin(aosView), end(aosView), begin(soaView), llama::One<Position>{});

    CHECK(sumX == 242672);
    CHECK(sumY == 248816);
    CHECK(sumZ == 255024);
}

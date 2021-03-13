#include <algorithm>
#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <numeric>

// clang-format off
namespace tag {
    struct X {};
    struct Y {};
    struct Z {};
} // namespace tag

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
    auto [sumX, sumY, sumZ]
        = std::transform_reduce(begin(aosView), end(aosView), begin(soaView), llama::One<Position>{});

    CHECK(sumX == 242672);
    CHECK(sumY == 248816);
    CHECK(sumZ == 255024);
}

// TODO: clang 10 and 11 fail to compile this currently with the issue described here:
// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
// let's try again with clang 12
// Intel LLVM compiler is also using the clang frontend
#if __has_include(<ranges>) && defined(__cpp_concepts) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER)
#    include <ranges>

TEST_CASE("ranges")
{
    using ArrayDomain = llama::ArrayDomain<1>;
    constexpr auto arrayDomain = ArrayDomain{32};
    constexpr auto mapping = llama::mapping::AoS<ArrayDomain, Position>{arrayDomain};
    auto view = llama::allocView(mapping);

    STATIC_REQUIRE(std::ranges::range<decltype(view)>);

    int i = 0;
    for (auto vd : view)
    {
        vd(tag::X{}) = ++i;
        vd(tag::Y{}) = ++i;
        vd(tag::Z{}) = ++i;
    }

    std::vector<int> v;
    for (auto y : view | std::views::filter([](auto vd) { return vd(tag::X{}) % 10 == 0; })
             | std::views::transform([](auto vd) { return vd(tag::Y{}); }) | std::views::take(2))
        v.push_back(y);
    CHECK(v == std::vector<int>{11, 41});
}
#endif

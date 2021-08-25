#include <algorithm>
#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <numeric>
#include <random>

// clang-format off
namespace tag {
    struct X {};
    struct Y {};
    struct Z {};
} // namespace tag

using Position = llama::Record<
    llama::Field<tag::X, int>,
    llama::Field<tag::Y, int>,
    llama::Field<tag::Z, int>
>;
// clang-format on

TEST_CASE("iterator")
{
    auto test = [](auto arrayDims)
    {
        using ArrayDims = decltype(arrayDims);
        auto mapping = llama::mapping::AoS<ArrayDims, Position>{arrayDims};
        auto view = llama::allocView(mapping);

        for(auto vd : view)
        {
            vd(tag::X{}) = 1;
            vd(tag::Y{}) = 2;
            vd(tag::Z{}) = 3;
        }
        std::transform(begin(view), end(view), begin(view), [](auto vd) { return vd * 2; });
        const auto& cview = std::as_const(view);
        const int sumY
            = std::accumulate(begin(cview), end(cview), 0, [](int acc, auto vd) { return acc + vd(tag::Y{}); });
        CHECK(sumY == 128); // NOLINT(bugprone-infinite-loop)
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.std_copy")
{
    auto test = [](auto arrayDims)
    {
        auto aosView = llama::allocView(llama::mapping::AoS{arrayDims, Position{}});
        auto soaView = llama::allocView(llama::mapping::SoA{arrayDims, Position{}});

        int i = 0;
        for(auto vd : aosView)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }
        std::copy(begin(aosView), end(aosView), begin(soaView));
        i = 0;
        for(auto vd : soaView)
        {
            CHECK(vd(tag::X{}) == ++i);
            CHECK(vd(tag::Y{}) == ++i);
            CHECK(vd(tag::Z{}) == ++i);
        }
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.transform_reduce")
{
    auto test = [](auto arrayDims)
    {
        auto aosView = llama::allocView(llama::mapping::AoS{arrayDims, Position{}});
        auto soaView = llama::allocView(llama::mapping::SoA{arrayDims, Position{}});

        int i = 0;
        for(auto vd : aosView)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }
        for(auto vd : soaView)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }
        // returned type is a llama::One<Position>
        auto [sumX, sumY, sumZ]
            = std::transform_reduce(begin(aosView), end(aosView), begin(soaView), llama::One<Position>{});

        CHECK(sumX == 242672);
        CHECK(sumY == 248816);
        CHECK(sumZ == 255024);
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.transform_inplace")
{
    auto test = [](auto arrayDims)
    {
        auto view = llama::allocView(llama::mapping::AoS{arrayDims, Position{}});

        int i = 0;
        for(auto vd : view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }

        std::transform(
            begin(view),
            end(view),
            begin(view),
            [](llama::One<Position> p)
            {
                p *= 2;
                return p;
            });

        i = 0;
        for(auto vd : view)
        {
            CHECK(vd(tag::X{}) == ++i * 2);
            CHECK(vd(tag::Y{}) == ++i * 2);
            CHECK(vd(tag::Z{}) == ++i * 2);
        }
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.transform_to")
{
    auto test = [](auto arrayDims)
    {
        auto view = llama::allocView(llama::mapping::AoS{arrayDims, Position{}});

        int i = 0;
        for(auto vd : view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }

        auto dst = llama::allocView(llama::mapping::SoA{arrayDims, Position{}});
        std::transform(
            begin(view),
            end(view),
            begin(dst),
            [](llama::One<Position> p)
            {
                p *= 2;
                return p;
            });

        i = 0;
        for(auto vd : view)
        {
            CHECK(vd(tag::X{}) == ++i);
            CHECK(vd(tag::Y{}) == ++i);
            CHECK(vd(tag::Z{}) == ++i);
        }
        i = 0;
        for(auto vd : dst)
        {
            CHECK(vd(tag::X{}) == ++i * 2);
            CHECK(vd(tag::Y{}) == ++i * 2);
            CHECK(vd(tag::Z{}) == ++i * 2);
        }
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}

TEST_CASE("iterator.different_record_dim")
{
    struct Pos1
    {
    };
    struct Pos2
    {
    };
    using WrappedPos = llama::Record<llama::Field<Pos1, Position>, llama::Field<Pos2, Position>>;

    auto arrayDims = llama::ArrayDims{32};
    auto aosView = llama::allocView(llama::mapping::AoS{arrayDims, WrappedPos{}});
    auto soaView = llama::allocView(llama::mapping::SoA{arrayDims, Position{}});

    int i = 0;
    for(auto vd : aosView)
    {
        vd(Pos1{}, tag::X{}) = ++i;
        vd(Pos1{}, tag::Y{}) = ++i;
        vd(Pos1{}, tag::Z{}) = ++i;
    }
    std::transform(begin(aosView), end(aosView), begin(soaView), [](auto wp) { return wp(Pos1{}) * 2; });

    i = 0;
    for(auto vd : soaView)
    {
        CHECK(vd(tag::X{}) == ++i * 2);
        CHECK(vd(tag::Y{}) == ++i * 2);
        CHECK(vd(tag::Z{}) == ++i * 2);
    }
}

// TODO: clang 10 and 11 fail to compile this currently with the issue described here:
// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
// let's try again with clang 12
// Intel LLVM compiler is also using the clang frontend
#if CAN_USE_RANGES
#    include <ranges>

TEST_CASE("ranges")
{
    auto test = [](auto arrayDims)
    {
        auto mapping = llama::mapping::AoS{arrayDims, Position{}};
        auto view = llama::allocView(mapping);

        STATIC_REQUIRE(std::ranges::range<decltype(view)>);

        int i = 0;
        for(auto vd : view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }

        std::vector<int> v;
        for(auto y : view | std::views::filter([](auto vd) { return vd(tag::X{}) % 10 == 0; })
                | std::views::transform([](auto vd) { return vd(tag::Y{}); }) | std::views::take(2))
            v.push_back(y);
        CHECK(v == std::vector<int>{11, 41});
    };
    test(llama::ArrayDims{32});
    test(llama::ArrayDims{4, 8});
    test(llama::ArrayDims{4, 2, 4});
}
#endif

TEST_CASE("iterator.sort")
{
    constexpr auto n = 10;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayDims{n}, Position{}});

    std::default_random_engine e{};
    std::uniform_int_distribution<int> d{0, 1000};
    for(auto vd : view)
    {
        vd(tag::X{}) = d(e);
        vd(tag::Y{}) = d(e);
        vd(tag::Z{}) = d(e);
    }
    auto manhattan_less = [](auto a, auto b)
    { return a(tag::X{}) + a(tag::Y{}) + a(tag::Z{}) < b(tag::X{}) + b(tag::Y{}) + b(tag::Z{}); };
    std::sort(begin(view), end(view), manhattan_less);

    for(auto i = 1; i < n; i++)
        CHECK(manhattan_less(view[i - 1], view[i]));
}

#include "common.hpp"

#include <algorithm>
#include <numeric>
#include <random>

using Position = Vec3I;

TEST_CASE("iterator.concepts")
{
    using Mapping = llama::mapping::AoS<llama::ArrayExtentsDynamic<std::size_t, 1>, Position>;
    using View = llama::View<Mapping, std::byte*>;
    using Iterator = typename View::iterator;

    STATIC_REQUIRE(std::is_same_v<std::iterator_traits<Iterator>::iterator_category, std::random_access_iterator_tag>);
    //#ifdef __cpp_lib_concepts
    //    STATIC_REQUIRE(std::random_access_iterator<Iterator>);
    //#endif
}

TEST_CASE("iterator")
{
    auto test = [](auto extents)
    {
        auto mapping = llama::mapping::AoS{extents, Position{}};
        auto view = llama::allocViewUninitialized(mapping);

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
    test(llama::ArrayExtents{32});
    test(llama::ArrayExtents{4, 8});
    test(llama::ArrayExtents{4, 2, 4});
}

TEST_CASE("iterator.std_copy")
{
    auto test = [](auto extents)
    {
        auto aosView = llama::allocViewUninitialized(llama::mapping::AoS{extents, Position{}});
        auto soaView = llama::allocViewUninitialized(llama::mapping::SoA{extents, Position{}});

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
    test(llama::ArrayExtents{32});
    test(llama::ArrayExtents{4, 8});
    test(llama::ArrayExtents{4, 2, 4});
}

TEST_CASE("iterator.transform_reduce")
{
    auto test = [](auto extents)
    {
        auto aosView = llama::allocViewUninitialized(llama::mapping::AoS{extents, Position{}});
        auto soaView = llama::allocViewUninitialized(llama::mapping::SoA{extents, Position{}});

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
    test(llama::ArrayExtents{32});
    test(llama::ArrayExtents{4, 8});
    test(llama::ArrayExtents{4, 2, 4});
}

TEST_CASE("iterator.transform_inplace")
{
    auto test = [](auto extents)
    {
        auto view = llama::allocViewUninitialized(llama::mapping::AoS{extents, Position{}});

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
    test(llama::ArrayExtents{32});
    test(llama::ArrayExtents{4, 8});
    test(llama::ArrayExtents{4, 2, 4});
}

TEST_CASE("iterator.transform_to")
{
    auto test = [](auto extents)
    {
        auto view = llama::allocViewUninitialized(llama::mapping::AoS{extents, Position{}});

        int i = 0;
        for(auto vd : view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }

        auto dst = llama::allocViewUninitialized(llama::mapping::SoA{extents, Position{}});
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
    test(llama::ArrayExtents{32});
    test(llama::ArrayExtents{4, 8});
    test(llama::ArrayExtents{4, 2, 4});
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

    auto extents = llama::ArrayExtents{32};
    auto aosView = llama::allocViewUninitialized(llama::mapping::AoS{extents, WrappedPos{}});
    auto soaView = llama::allocViewUninitialized(llama::mapping::SoA{extents, Position{}});

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

#if CAN_USE_RANGES
#    include <ranges>
TEST_CASE("ranges")
{
    auto test = [](auto extents)
    {
        auto mapping = llama::mapping::AoS{extents, Position{}};
        auto view = llama::allocViewUninitialized(mapping);

        STATIC_REQUIRE(std::ranges::range<decltype(view)>);

        int i = 0;
        for(auto vd : view)
        {
            vd(tag::X{}) = ++i;
            vd(tag::Y{}) = ++i;
            vd(tag::Z{}) = ++i;
        }

        std::vector<int> v;
        // BUG: MSVC errors when we put the range expression below directly into the for loop
        auto range = view | std::views::filter([](auto vd) { return vd(tag::X{}) % 10 == 0; })
            | std::views::transform([](auto vd) { return vd(tag::Y{}); }) | std::views::take(2);
        for(auto y : range)
            v.push_back(y);
        CHECK(v == std::vector<int>{11, 41});
    };
    test(llama::ArrayExtents{32});
    test(llama::ArrayExtents{4, 8});
    test(llama::ArrayExtents{4, 2, 4});
}
#endif

TEST_CASE("iterator.sort")
{
    constexpr auto N = 10;
    auto view = llama::allocViewUninitialized(llama::mapping::AoS{llama::ArrayExtents<std::size_t, N>{}, Position{}});

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

    for(auto i = 1; i < N; i++)
        CHECK(manhattan_less(view[i - 1], view[i]));
}

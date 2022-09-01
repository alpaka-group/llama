#include "common.hpp"

TEST_CASE("Tuple.CTAD")
{
    constexpr auto t0 = llama::Tuple{};
    constexpr auto t1 = llama::Tuple{1};
    constexpr auto t2 = llama::Tuple{1, 1.0f};
    constexpr auto t3 = llama::Tuple{1, 1.0f, nullptr};

    STATIC_REQUIRE(std::is_same_v<decltype(t0), const llama::Tuple<>>);
    STATIC_REQUIRE(std::is_same_v<decltype(t1), const llama::Tuple<int>>);
    STATIC_REQUIRE(std::is_same_v<decltype(t2), const llama::Tuple<int, float>>);
    STATIC_REQUIRE(std::is_same_v<decltype(t3), const llama::Tuple<int, float, std::nullptr_t>>);
}

TEST_CASE("Tuple.size")
{
    using IC = std::integral_constant<int, 1>;

    STATIC_REQUIRE(std::is_empty_v<llama::Tuple<>>);
    STATIC_REQUIRE(sizeof(llama::Tuple<int>) == 1 * sizeof(int));
    STATIC_REQUIRE(sizeof(llama::Tuple<int, int>) == 2 * sizeof(int));
    STATIC_REQUIRE(sizeof(llama::Tuple<int, int, int>) == 3 * sizeof(int));

    STATIC_REQUIRE(std::is_empty_v<llama::Tuple<IC>>);
    STATIC_REQUIRE(std::is_empty_v<llama::Tuple<IC, IC, IC>>);

    STATIC_REQUIRE(sizeof(llama::Tuple<IC, int>) == sizeof(int));
    STATIC_REQUIRE(sizeof(llama::Tuple<int, IC>) == sizeof(int));

    STATIC_REQUIRE(sizeof(llama::Tuple<IC, int, int>) == 2 * sizeof(int));
    STATIC_REQUIRE(sizeof(llama::Tuple<int, IC, int>) == 2 * sizeof(int));
    STATIC_REQUIRE(sizeof(llama::Tuple<int, int, IC>) == 2 * sizeof(int));

    STATIC_REQUIRE(sizeof(llama::Tuple<IC, int, IC>) == sizeof(int));
    STATIC_REQUIRE(sizeof(llama::Tuple<int, IC, IC>) == sizeof(int));
    STATIC_REQUIRE(sizeof(llama::Tuple<IC, IC, int>) == sizeof(int));
}

TEST_CASE("Tuple.get")
{
    constexpr auto t = llama::Tuple{1, 1.0f, nullptr};

    STATIC_REQUIRE(llama::get<0>(t) == 1);
    STATIC_REQUIRE(llama::get<1>(t) == 1.0f);
    STATIC_REQUIRE(llama::get<2>(t) == nullptr);
}

TEST_CASE("Tuple.get_mutable")
{
    auto t = llama::Tuple{1, 1.0f};
    llama::get<0>(t)++;
    llama::get<1>(t)++;
    CHECK(llama::get<0>(t) == 2);
    CHECK(llama::get<1>(t) == 2.0f);
}

TEST_CASE("Tuple.structured_binding")
{
    const auto t = llama::Tuple{1, 1.0f, nullptr};
    const auto [a, b, c] = t;
    CHECK(a == 1);
    CHECK(b == 1.0f);
    CHECK(c == nullptr);
}

#ifndef __INTEL_COMPILER
TEST_CASE("Tuple.converting_ctor")
{
    struct A
    {
        constexpr explicit A(int i) : i{i * 2}
        {
        }
        int i;
    };
    struct B // NOLINT(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
    {
        constexpr explicit B(int* p) : p{p}
        {
        }
        B(const B&) = delete;
        int* p;
    };

    constexpr auto t = llama::Tuple<double, A, B>{1, 42, nullptr};
    STATIC_REQUIRE(llama::get<0>(t) == 1.0);
    STATIC_REQUIRE(llama::get<1>(t).i == 84);
    STATIC_REQUIRE(llama::get<2>(t).p == nullptr);
}
#endif

TEST_CASE("Tuple.copy_ctor")
{
    constexpr auto t = llama::Tuple{1, 1.0f};
    constexpr auto t2 = t;
    STATIC_REQUIRE(t2 == llama::Tuple{1, 1.0f});
}

TEST_CASE("Tuple.copy_assign")
{
    const auto t = llama::Tuple{1, 1.0f};
    llama::Tuple<int, float> t2{};
    t2 = t;
    CHECK(t2 == llama::Tuple{1, 1.0f});
}

TEST_CASE("Tuple.operator==")
{
    STATIC_REQUIRE(llama::Tuple{} == llama::Tuple{});
    STATIC_REQUIRE(llama::Tuple{1, 2.0f, nullptr} == llama::Tuple{1, 2.0f, nullptr});
}

TEST_CASE("Tuple.operator!=")
{
    STATIC_REQUIRE(llama::Tuple{} != llama::Tuple{1});
    STATIC_REQUIRE(llama::Tuple{1} != llama::Tuple{});
    STATIC_REQUIRE(llama::Tuple{1, 2.0f} != llama::Tuple{1, 3.0f});
    STATIC_REQUIRE(llama::Tuple{1, 2.0f, nullptr} != llama::Tuple{nullptr, 2.0f, 1});
}

TEST_CASE("Tuple.tupleCat")
{
    constexpr auto t0 = llama::Tuple{};
    constexpr auto t1 = llama::Tuple{1};
    constexpr auto t2 = llama::Tuple{1, 1.0f};

    STATIC_REQUIRE(llama::tupleCat(t0, t1) == llama::Tuple{1});
    STATIC_REQUIRE(llama::tupleCat(t1, t0) == llama::Tuple{1});
    STATIC_REQUIRE(llama::tupleCat(t1, t2) == llama::Tuple{1, 1, 1.0f});
    STATIC_REQUIRE(llama::tupleCat(t2, t1) == llama::Tuple{1, 1.0f, 1});
}

TEST_CASE("Tuple.tupleTransform")
{
    constexpr auto f = [](auto v) { return v + 1; };
    STATIC_REQUIRE(llama::tupleTransform(llama::Tuple{}, f) == llama::Tuple{});
    STATIC_REQUIRE(llama::tupleTransform(llama::Tuple{1}, f) == llama::Tuple{2});
    STATIC_REQUIRE(llama::tupleTransform(llama::Tuple{1, 1.0f}, f) == llama::Tuple{2, 2.0f});
}

TEST_CASE("Tuple.popFront")
{
    STATIC_REQUIRE(llama::popFront(llama::Tuple{1}) == llama::Tuple{});
    STATIC_REQUIRE(llama::popFront(llama::Tuple{1, 1.0f}) == llama::Tuple{1.0f});
    STATIC_REQUIRE(llama::popFront(llama::Tuple{1.0f, 1, nullptr}) == llama::Tuple{1, nullptr});
}

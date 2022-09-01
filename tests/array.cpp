#include "common.hpp"

TEST_CASE("Array.empty")
{
    STATIC_REQUIRE(llama::Array<int, 0>{}.empty());
    STATIC_REQUIRE(!llama::Array{1}.empty());
}

TEST_CASE("Array.front")
{
    llama::Array a{1, 2, 3};
    CHECK(a.front() == 1);
    CHECK(std::as_const(a).front() == 1);
    a.front() = 4;
    CHECK(a == llama::Array{4, 2, 3});
}

TEST_CASE("Array.back")
{
    llama::Array a{1, 2, 3};
    CHECK(a.back() == 3);
    CHECK(std::as_const(a).back() == 3);
    a.back() = 4;
    CHECK(a == llama::Array{1, 2, 4});
}

TEST_CASE("Array.begin")
{
    llama::Array a{1, 2, 3};
    // NOLINTNEXTLINE(readability-container-data-pointer)
    CHECK(a.begin() == &a[0]);
    CHECK(*a.begin() == 1);
    CHECK(llama::Array<int, 0>{}.begin() == nullptr);
}

TEST_CASE("Array.range_for")
{
    llama::Array a{1, 2, 3};
    int i = 1;
    for(auto e : a)
    {
        CHECK(e == i);
        i++;
    }
}

TEST_CASE("Array.end")
{
    llama::Array a{1, 2, 3};
    CHECK(a.end() == &a[3]);
    CHECK(llama::Array<int, 0>{}.end() == nullptr);
}

TEST_CASE("Array.data")
{
    llama::Array a{1, 2, 3};
    // NOLINTNEXTLINE(readability-container-data-pointer)
    CHECK(a.data() == &a[0]);
    CHECK(a.begin()[0] == 1);
    CHECK(a.begin()[1] == 2);
    CHECK(a.begin()[2] == 3);
    CHECK(llama::Array<int, 0>{}.data() == nullptr);
}

TEST_CASE("Array.operator<<")
{
    auto put = [](auto array)
    {
        std::stringstream ss;
        ss << array;
        return ss.str();
    };

    CHECK(put(llama::Array<int, 0>{}) == "Array{}");
    CHECK(put(llama::Array{1}) == "Array{1}");
    CHECK(put(llama::Array{1, 2, 3}) == "Array{1, 2, 3}");
    CHECK(put(llama::Array{1.1, 2.2, 3.3}) == "Array{1.1, 2.2, 3.3}");
}

TEST_CASE("Array.pushFront")
{
    STATIC_REQUIRE(pushFront(llama::Array<int, 0>{}, 1) == llama::Array{1});
    STATIC_REQUIRE(pushFront(llama::Array{1}, 42) == llama::Array{42, 1});
    STATIC_REQUIRE(pushFront(llama::Array{1, 2}, 42) == llama::Array{42, 1, 2});
    STATIC_REQUIRE(pushFront(llama::Array{1, 2, 3}, 42) == llama::Array{42, 1, 2, 3});
}

TEST_CASE("Array.pushBack")
{
    STATIC_REQUIRE(pushBack(llama::Array<int, 0>{}, 1) == llama::Array{1});
    STATIC_REQUIRE(pushBack(llama::Array{1}, 42) == llama::Array{1, 42});
    STATIC_REQUIRE(pushBack(llama::Array{1, 2}, 42) == llama::Array{1, 2, 42});
    STATIC_REQUIRE(pushBack(llama::Array{1, 2, 3}, 42) == llama::Array{1, 2, 3, 42});
}

TEST_CASE("Array.popFront")
{
    STATIC_REQUIRE(popFront(llama::Array{1}) == llama::Array<int, 0>{});
    STATIC_REQUIRE(popFront(llama::Array{1, 2}) == llama::Array{2});
    STATIC_REQUIRE(popFront(llama::Array{3, 2, 1}) == llama::Array{2, 1});
}

TEST_CASE("Array.popBack")
{
    STATIC_REQUIRE(popBack(llama::Array{1}) == llama::Array<int, 0>{});
    STATIC_REQUIRE(popBack(llama::Array{1, 2}) == llama::Array{1});
    STATIC_REQUIRE(popBack(llama::Array{3, 2, 1}) == llama::Array{3, 2});
}

TEST_CASE("Array.product")
{
    STATIC_REQUIRE(product(llama::Array<int, 0>{}) == 1);
    STATIC_REQUIRE(product(llama::Array{1}) == 1);
    STATIC_REQUIRE(product(llama::Array{1, 2}) == 2);
    STATIC_REQUIRE(product(llama::Array{3, 2, 1}) == 6);
}

TEST_CASE("Array.dot")
{
    STATIC_REQUIRE(llama::dot(llama::Array<int, 0>{}, llama::Array<int, 0>{}) == 0);
    STATIC_REQUIRE(llama::dot(llama::Array{2}, llama::Array{3}) == 6);
    STATIC_REQUIRE(llama::dot(llama::Array{4, 5}, llama::Array{6, 7}) == 59);
    STATIC_REQUIRE(llama::dot(llama::Array{1, 2, 3, 4}, llama::Array{-5, 6, -7, 0}) == -14);
}

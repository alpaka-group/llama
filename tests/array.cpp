#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

TEST_CASE("Array.push_front")
{
    STATIC_REQUIRE(push_front(llama::Array<int, 0>{}, 1) == llama::Array{1});
    STATIC_REQUIRE(push_front(llama::Array{1}, 42) == llama::Array{42, 1});
    STATIC_REQUIRE(push_front(llama::Array{1, 2}, 42) == llama::Array{42, 1, 2});
    STATIC_REQUIRE(push_front(llama::Array{1, 2, 3}, 42) == llama::Array{42, 1, 2, 3});
}

TEST_CASE("Array.push_back")
{
    STATIC_REQUIRE(push_back(llama::Array<int, 0>{}, 1) == llama::Array{1});
    STATIC_REQUIRE(push_back(llama::Array{1}, 42) == llama::Array{1, 42});
    STATIC_REQUIRE(push_back(llama::Array{1, 2}, 42) == llama::Array{1, 2, 42});
    STATIC_REQUIRE(push_back(llama::Array{1, 2, 3}, 42) == llama::Array{1, 2, 3, 42});
}

TEST_CASE("Array.pop_front")
{
    STATIC_REQUIRE(pop_front(llama::Array{1}) == llama::Array<int, 0>{});
    STATIC_REQUIRE(pop_front(llama::Array{1, 2}) == llama::Array{2});
    STATIC_REQUIRE(pop_front(llama::Array{3, 2, 1}) == llama::Array{2, 1});
}

TEST_CASE("Array.pop_back")
{
    STATIC_REQUIRE(pop_back(llama::Array{1}) == llama::Array<int, 0>{});
    STATIC_REQUIRE(pop_back(llama::Array{1, 2}) == llama::Array{1});
    STATIC_REQUIRE(pop_back(llama::Array{3, 2, 1}) == llama::Array{3, 2});
}

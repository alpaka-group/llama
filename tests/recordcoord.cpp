// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

#include <sstream>

TEST_CASE("RecordCoord.List")
{
    STATIC_REQUIRE(std::is_same_v<typename llama::RecordCoord<>::List, mp_list_c<std::size_t>>);
    STATIC_REQUIRE(std::is_same_v<typename llama::RecordCoord<1, 2, 3>::List, mp_list_c<std::size_t, 1, 2, 3>>);
}

TEST_CASE("RecordCoord.front")
{
    STATIC_REQUIRE(llama::RecordCoord<1, 2, 3>::front == 1);
}

TEST_CASE("RecordCoord.back")
{
    STATIC_REQUIRE(llama::RecordCoord<1, 2, 3>::back == 3);
}

TEST_CASE("RecordCoord.size")
{
    STATIC_REQUIRE(llama::RecordCoord<>::size == 0);
    STATIC_REQUIRE(llama::RecordCoord<1>::size == 1);
    STATIC_REQUIRE(llama::RecordCoord<1, 2>::size == 2);
    STATIC_REQUIRE(llama::RecordCoord<1, 2, 3>::size == 3);
}

TEST_CASE("RecordCoord.operator==")
{
    STATIC_REQUIRE(llama::RecordCoord{} == llama::RecordCoord{});
    STATIC_REQUIRE(llama::RecordCoord<1, 2, 3>{} == llama::RecordCoord<1, 2, 3>{});
}

TEST_CASE("RecordCoord.operator!=")
{
    STATIC_REQUIRE(llama::RecordCoord{} != llama::RecordCoord<1>{});
    STATIC_REQUIRE(llama::RecordCoord<1>{} != llama::RecordCoord{});
    STATIC_REQUIRE(llama::RecordCoord<1, 2, 3>{} != llama::RecordCoord<4, 1, 2, 3>{});
}

TEST_CASE("RecordCoord.operator<<")
{
    auto put = [](auto rc)
    {
        std::stringstream ss;
        ss << rc;
        return ss.str();
    };

    CHECK(put(llama::RecordCoord{}) == "RecordCoord<>");
    CHECK(put(llama::RecordCoord<1>{}) == "RecordCoord<1>");
    CHECK(put(llama::RecordCoord<1, 2, 3>{}) == "RecordCoord<1, 2, 3>");
}

TEST_CASE("Cat")
{
    STATIC_REQUIRE(std::is_same_v<llama::Cat<>, llama::RecordCoord<>>);

    STATIC_REQUIRE(std::is_same_v<llama::Cat<llama::RecordCoord<>>, llama::RecordCoord<>>);
    STATIC_REQUIRE(std::is_same_v<llama::Cat<llama::RecordCoord<1, 2, 3>>, llama::RecordCoord<1, 2, 3>>);

    STATIC_REQUIRE(std::is_same_v<llama::Cat<llama::RecordCoord<>, llama::RecordCoord<>>, llama::RecordCoord<>>);
    STATIC_REQUIRE(std::is_same_v<llama::Cat<llama::RecordCoord<1>, llama::RecordCoord<>>, llama::RecordCoord<1>>);
    STATIC_REQUIRE(std::is_same_v<llama::Cat<llama::RecordCoord<>, llama::RecordCoord<1>>, llama::RecordCoord<1>>);
    STATIC_REQUIRE(std::is_same_v<llama::Cat<llama::RecordCoord<1>, llama::RecordCoord<1>>, llama::RecordCoord<1, 1>>);

    STATIC_REQUIRE(std::is_same_v<
                   llama::Cat<llama::RecordCoord<1, 2>, llama::RecordCoord<>, llama::RecordCoord<3>>,
                   llama::RecordCoord<1, 2, 3>>);

    STATIC_REQUIRE(std::is_same_v<
                   llama::Cat<
                       llama::RecordCoord<>,
                       llama::RecordCoord<1, 2>,
                       llama::RecordCoord<3, 4, 5, 6>,
                       llama::RecordCoord<>,
                       llama::RecordCoord<7>>,
                   llama::RecordCoord<1, 2, 3, 4, 5, 6, 7>>);
}

TEST_CASE("cat")
{
    STATIC_REQUIRE(
        llama::cat(
            llama::RecordCoord{},
            llama::RecordCoord<1, 2>{},
            llama::RecordCoord<3, 4, 5, 6>{},
            llama::RecordCoord{},
            llama::RecordCoord<7>{})
        == llama::RecordCoord<1, 2, 3, 4, 5, 6, 7>{});
}

TEST_CASE("recordCoordCommonPrefixIsBigger")
{
    // clang-format off
    STATIC_REQUIRE( llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 1, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 1>, llama::RecordCoord<0, 0, 0>>);

    STATIC_REQUIRE( llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0, 0>, llama::RecordCoord<0, 0   >>);
    STATIC_REQUIRE( llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0, 0>, llama::RecordCoord<0      >>);
    STATIC_REQUIRE( llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0   >, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<0, 0, 0>>);

    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<1, 0, 0>>);
    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 1, 0>>);
    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 0, 1>>);

    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<1, 0, 0>>);
    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<1, 1, 0>>);
    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<1, 1, 1>>);

    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0   >, llama::RecordCoord<0, 0, 1>>);
    STATIC_REQUIRE(!llama::recordCoordCommonPrefixIsBigger<llama::RecordCoord<0      >, llama::RecordCoord<0, 0, 1>>);
    // clang-format on
}

TEST_CASE("_DT")
{
    using namespace llama::literals;
    STATIC_REQUIRE(std::is_same_v<llama::RecordCoord<0>, decltype(0_RC)>);
    STATIC_REQUIRE(std::is_same_v<llama::RecordCoord<1>, decltype(1_RC)>);
    STATIC_REQUIRE(std::is_same_v<llama::RecordCoord<10>, decltype(10_RC)>);
    STATIC_REQUIRE(std::is_same_v<llama::RecordCoord<165463135>, decltype(165463135_RC)>);
}

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

TEST_CASE("RecordCoordCommonPrefixIsBigger")
{
    // clang-format off
    STATIC_REQUIRE( llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 1, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 1>, llama::RecordCoord<0, 0, 0>>);

    STATIC_REQUIRE( llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0, 0>, llama::RecordCoord<0, 0   >>);
    STATIC_REQUIRE( llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0, 0>, llama::RecordCoord<0      >>);
    STATIC_REQUIRE( llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1, 0   >, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<0, 0, 0>>);

    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 0, 0>>);
    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<1, 0, 0>>);
    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 1, 0>>);
    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0, 0>, llama::RecordCoord<0, 0, 1>>);

    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<1, 0, 0>>);
    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<1, 1, 0>>);
    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<1      >, llama::RecordCoord<1, 1, 1>>);

    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0, 0   >, llama::RecordCoord<0, 0, 1>>);
    STATIC_REQUIRE(!llama::RecordCoordCommonPrefixIsBigger<llama::RecordCoord<0      >, llama::RecordCoord<0, 0, 1>>);
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

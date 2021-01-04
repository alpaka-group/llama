#include <catch2/catch.hpp>
#include <llama/llama.hpp>

TEST_CASE("DatumCoordCommonPrefixIsBigger")
{
    // clang-format off
    STATIC_REQUIRE( llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1, 0, 0>, llama::DatumCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 1, 0>, llama::DatumCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 0, 1>, llama::DatumCoord<0, 0, 0>>);

    STATIC_REQUIRE( llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1, 0, 0>, llama::DatumCoord<0, 0   >>);
    STATIC_REQUIRE( llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1, 0, 0>, llama::DatumCoord<0      >>);
    STATIC_REQUIRE( llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1, 0   >, llama::DatumCoord<0, 0, 0>>);
    STATIC_REQUIRE( llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<0, 0, 0>>);

    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 0, 0>>);
    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 0, 0>>);
    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<1, 0, 0>>);
    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 1, 0>>);
    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 0, 1>>);

    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<1, 0, 0>>);
    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<1, 1, 0>>);
    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<1, 1, 1>>);

    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0, 0   >, llama::DatumCoord<0, 0, 1>>);
    STATIC_REQUIRE(!llama::DatumCoordCommonPrefixIsBigger<llama::DatumCoord<0      >, llama::DatumCoord<0, 0, 1>>);
    // clang-format on
}

TEST_CASE("_DT")
{
    using namespace llama::literals;
    STATIC_REQUIRE(std::is_same_v<llama::DatumCoord<0>, decltype(0_DC)>);
    STATIC_REQUIRE(std::is_same_v<llama::DatumCoord<1>, decltype(1_DC)>);
    STATIC_REQUIRE(std::is_same_v<llama::DatumCoord<10>, decltype(10_DC)>);
    STATIC_REQUIRE(std::is_same_v<llama::DatumCoord<165463135>, decltype(165463135_DC)>);
}

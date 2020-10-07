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

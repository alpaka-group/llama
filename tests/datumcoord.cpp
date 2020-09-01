#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
static_assert( llama::DatumCoordIsBigger<llama::DatumCoord<1, 0, 0>, llama::DatumCoord<0, 0, 0>>);
static_assert( llama::DatumCoordIsBigger<llama::DatumCoord<0, 1, 0>, llama::DatumCoord<0, 0, 0>>);
static_assert( llama::DatumCoordIsBigger<llama::DatumCoord<0, 0, 1>, llama::DatumCoord<0, 0, 0>>);

static_assert( llama::DatumCoordIsBigger<llama::DatumCoord<1, 0, 0>, llama::DatumCoord<0, 0   >>);
static_assert( llama::DatumCoordIsBigger<llama::DatumCoord<1, 0, 0>, llama::DatumCoord<0      >>);
static_assert( llama::DatumCoordIsBigger<llama::DatumCoord<1, 0   >, llama::DatumCoord<0, 0, 0>>);
static_assert( llama::DatumCoordIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<0, 0, 0>>);

static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 0, 0>>);
static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 0, 0>>);
static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<1, 0, 0>>);
static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 1, 0>>);
static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<0, 0, 0>, llama::DatumCoord<0, 0, 1>>);

static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<1, 0, 0>>);
static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<1, 1, 0>>);
static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<1      >, llama::DatumCoord<1, 1, 1>>);

static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<0, 0   >, llama::DatumCoord<0, 0, 1>>);
static_assert(!llama::DatumCoordIsBigger<llama::DatumCoord<0      >, llama::DatumCoord<0, 0, 1>>);

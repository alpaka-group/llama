// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

#include <llama/Proofs.hpp>

#ifndef __NVCOMPILER
TEST_CASE("mapsNonOverlappingly.PackedAoS")
{
#    ifdef __cpp_constexpr_dynamic_alloc
    constexpr auto mapping = llama::mapping::PackedAoS<llama::ArrayExtentsDynamic<std::size_t, 2>, Particle>{{32, 32}};
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#    else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#    endif
}

TEST_CASE("mapsNonOverlappingly.AlignedAoS")
{
#    ifdef __cpp_constexpr_dynamic_alloc
    constexpr auto mapping
        = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<std::size_t, 2>, Particle>{{32, 32}};
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#    else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#    endif
}
#endif

TEST_CASE("mapsNonOverlappingly.MapEverythingToZero")
{
#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(
        llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayExtentsDynamic<std::size_t, 1>{1}, double{}}));
    STATIC_REQUIRE(
        !llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayExtentsDynamic<std::size_t, 1>{2}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(
        MapEverythingToZero{llama::ArrayExtentsDynamic<std::size_t, 2>{32, 32}, Particle{}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

TEST_CASE("mapsNonOverlappingly.ModulusMapping")
{
#ifdef __cpp_constexpr_dynamic_alloc
    using Modulus10Mapping = ModulusMapping<llama::ArrayExtentsDynamic<std::size_t, 1>, Particle, 10>;
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{1}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{9}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{10}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{11}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{25}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

TEST_CASE("maps.ModulusMapping")
{
    using Extents = llama::ArrayExtentsDynamic<std::size_t, 1>;
    constexpr auto extents = Extents{128};
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::AoS{extents, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<8>(llama::mapping::AoS{extents, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<16>(llama::mapping::AoS{extents, Particle{}}));

    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::SoA{extents, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<8>(llama::mapping::SoA{extents, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<16>(llama::mapping::SoA{extents, Particle{}}));

    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::AoSoA<Extents, Particle, 8>{extents}));
    STATIC_REQUIRE(
        llama::mapsPiecewiseContiguous<8>(llama::mapping::AoSoA<Extents, Particle, 8>{extents, Particle{}}));
    STATIC_REQUIRE(
        !llama::mapsPiecewiseContiguous<16>(llama::mapping::AoSoA<Extents, Particle, 8>{extents, Particle{}}));
}

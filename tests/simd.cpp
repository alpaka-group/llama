// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

#include <ciso646>
#ifdef _MSC_VER
// MSVC does not support std::experimental::simd and #warning
#elif defined(__NVCOMPILER)
#    pragma message(                                                                                                  \
            "LLAMA SIMD tests disabled for nvc++. It fails to compile std::experimental::simd due to unrecognized intrinsics")
#elif defined(__NVCC__)
#    pragma message(                                                                                                  \
            "LLAMA SIMD tests disabled for nvcc. It fails to compile std::experimental::simd due to invalid type conversions")
#elif defined(_LIBCPP_VERSION)
#    warning "LLAMA SIMD tests disabled for libc++. Their std::experimental::simd implementation is incomplete"
#elif !__has_include(<experimental/simd>)
#    warning "LLAMA SIMD tests disabled. Need std::experimental::simd, which is available in libstdc++ since GCC 11"
#elif defined(__GNUG__) && __GNUC__ >= 12 && defined(__SANITIZE_ADDRESS__) && defined(__AVX512F__)
#    warning "With ASan enabled, g++-12 and later generates an aligned store to an unaligned address with AVX512"
#else
#    include <experimental/simd>

namespace stdx = std::experimental;

template<typename T, typename Abi>
struct llama::SimdTraits<stdx::simd<T, Abi>>
{
    using value_type = T;

    inline static constexpr std::size_t lanes = stdx::simd<T, Abi>::size();

    static LLAMA_FORCE_INLINE auto loadUnaligned(const value_type* mem) -> stdx::simd<T, Abi>
    {
        return {mem, stdx::element_aligned};
    }

    static LLAMA_FORCE_INLINE void storeUnaligned(stdx::simd<T, Abi> simd, value_type* mem)
    {
        simd.copy_to(mem, stdx::element_aligned);
    }

    static LLAMA_FORCE_INLINE auto gather(const value_type* mem, std::array<int, lanes> indices) -> stdx::simd<T, Abi>
    {
        // no native support for gather yet
        return stdx::simd<T, Abi>{[&](auto ic) { return mem[indices[ic]]; }};
    }

    static LLAMA_FORCE_INLINE void scatter(stdx::simd<T, Abi> simd, value_type* mem, std::array<int, lanes> indices)
    {
        // no native support for scatter yet
        for(std::size_t i = 0; i < lanes; i++)
            mem[indices[i]] = simd[i];
    }
};

namespace
{
    /// Turns a Simd vector into a congruent array, so it can be compared and printed by Catch2
    template<typename Simd>
    struct SimdRange : std::array<typename Simd::value_type, Simd::size()>
    {
        explicit SimdRange(Simd simd)
        {
            for(std::size_t i = 0; i < Simd::size(); i++)
                (*this)[i] = simd[i];
        }
    };
} // namespace

// same as Particle, but use a uint8_t instead of bool
using ParticleSimd = llama::Record<
    llama::Field<tag::Pos, Vec3D>,
    llama::Field<tag::Mass, float>,
    llama::Field<tag::Vel, Vec3D>,
    llama::Field<tag::Flags, uint8_t[4]>>;

TEST_CASE("simd.SimdizeN.stdsimd")
{
    STATIC_REQUIRE(std::is_same_v<llama::SimdizeN<float, 8, stdx::fixed_size_simd>, stdx::fixed_size_simd<float, 8>>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::SimdizeN<Vec2F, 8, stdx::fixed_size_simd>,
                   llama::Record<
                       llama::Field<tag::X, stdx::fixed_size_simd<float, 8>>,
                       llama::Field<tag::Y, stdx::fixed_size_simd<float, 8>>>>);
}

TEST_CASE("simd.Simdize.stdsimd")
{
    STATIC_REQUIRE(std::is_same_v<llama::Simdize<float, stdx::native_simd>, stdx::native_simd<float>>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::Simdize<Vec2F, stdx::native_simd>,
                   llama::Record<
                       llama::Field<tag::X, stdx::native_simd<float>>,
                       llama::Field<tag::Y, stdx::native_simd<float>>>>);
}

TEST_CASE("simd.simdLanesWithFullVectorsFor.stdsimd")
{
    STATIC_REQUIRE(llama::simdLanesWithFullVectorsFor<float, stdx::native_simd> == stdx::native_simd<float>::size());
    STATIC_REQUIRE(llama::simdLanesWithFullVectorsFor<Vec2F, stdx::native_simd> == stdx::native_simd<float>::size());
    STATIC_REQUIRE(llama::simdLanesWithFullVectorsFor<Vec3D, stdx::native_simd> == stdx::native_simd<double>::size());
    STATIC_REQUIRE(
        llama::simdLanesWithFullVectorsFor<ParticleSimd, stdx::native_simd> == stdx::native_simd<uint8_t>::size());
}

TEST_CASE("simd.simdLanesWithLeastRegistersFor.stdsimd")
{
    STATIC_REQUIRE(
        llama::simdLanesWithLeastRegistersFor<float, stdx::native_simd> == stdx::native_simd<float>::size());
    STATIC_REQUIRE(
        llama::simdLanesWithLeastRegistersFor<Vec2F, stdx::native_simd> == stdx::native_simd<float>::size());
    STATIC_REQUIRE(
        llama::simdLanesWithLeastRegistersFor<Vec3D, stdx::native_simd> == stdx::native_simd<double>::size());
    STATIC_REQUIRE(
        llama::simdLanesWithLeastRegistersFor<
            ParticleSimd,
            stdx::native_simd>
        == stdx::native_simd<double>::size()); // 11 registers with 4 lanes used each
}

TEST_CASE("simd.SimdN.stdsimd")
{
    STATIC_REQUIRE(std::is_same_v<llama::SimdN<float, 8, stdx::fixed_size_simd>, stdx::fixed_size_simd<float, 8>>);
    STATIC_REQUIRE(std::is_same_v<llama::SimdN<float, 1, stdx::fixed_size_simd>, float>);

    {
        llama::SimdN<Vec2F, 8, stdx::fixed_size_simd> v;
        auto& x = v(tag::X{});
        auto& y = v(tag::Y{});
        STATIC_REQUIRE(std::is_same_v<decltype(x), stdx::fixed_size_simd<float, 8>&>);
        STATIC_REQUIRE(std::is_same_v<decltype(y), stdx::fixed_size_simd<float, 8>&>);
    }

    {
        llama::SimdN<Vec2F, 1, stdx::fixed_size_simd> v;
        auto& x = v(tag::X{});
        auto& y = v(tag::Y{});
        STATIC_REQUIRE(std::is_same_v<decltype(x), float&>);
        STATIC_REQUIRE(std::is_same_v<decltype(y), float&>);
    }

    {
        llama::SimdN<ParticleSimd, 4, stdx::fixed_size_simd> p;
        auto& x = p(tag::Pos{}, tag::X{});
        auto& m = p(tag::Mass{});
        auto& f0 = p(tag::Flags{}, llama::RecordCoord<0>{});
        STATIC_REQUIRE(std::is_same_v<decltype(x), stdx::fixed_size_simd<double, 4>&>);
        STATIC_REQUIRE(std::is_same_v<decltype(m), stdx::fixed_size_simd<float, 4>&>);
        STATIC_REQUIRE(std::is_same_v<decltype(f0), stdx::fixed_size_simd<uint8_t, 4>&>);
    }
}

TEST_CASE("simd.Simd.stdsimd")
{
    STATIC_REQUIRE(std::is_same_v<llama::Simd<float, stdx::native_simd>, stdx::native_simd<float>>);

    {
        llama::Simd<Vec2F, stdx::native_simd> v;
        auto& x = v(tag::X{});
        auto& y = v(tag::Y{});
        STATIC_REQUIRE(std::is_same_v<decltype(x), stdx::native_simd<float>&>);
        STATIC_REQUIRE(std::is_same_v<decltype(y), stdx::native_simd<float>&>);
    }

    {
        llama::Simd<ParticleSimd, stdx::native_simd> p;
        auto& x = p(tag::Pos{}, tag::X{});
        auto& m = p(tag::Mass{});
        auto& f0 = p(tag::Flags{}, llama::RecordCoord<0>{});
        STATIC_REQUIRE(std::is_same_v<decltype(x), stdx::native_simd<double>&>);
        STATIC_REQUIRE(std::is_same_v<decltype(m), stdx::native_simd<float>&>);
        STATIC_REQUIRE(std::is_same_v<decltype(f0), stdx::native_simd<uint8_t>&>);
    }
}

TEST_CASE("simd.simdLanes.stdsimd")
{
    STATIC_REQUIRE(llama::simdLanes<float> == 1);

    STATIC_REQUIRE(llama::simdLanes<llama::One<ParticleSimd>> == 1);

    STATIC_REQUIRE(llama::simdLanes<stdx::native_simd<float>> == stdx::native_simd<float>::size());
    STATIC_REQUIRE(llama::simdLanes<stdx::fixed_size_simd<float, 4>> == 4);

    STATIC_REQUIRE(llama::simdLanes<llama::Simd<float, stdx::native_simd>> == stdx::native_simd<float>::size());
    STATIC_REQUIRE(llama::simdLanes<llama::Simd<Vec2F, stdx::native_simd>> == stdx::native_simd<float>::size());

    STATIC_REQUIRE(llama::simdLanes<llama::SimdN<ParticleSimd, 8, stdx::fixed_size_simd>> == 8);
    STATIC_REQUIRE(llama::simdLanes<llama::SimdN<ParticleSimd, 1, stdx::fixed_size_simd>> == 1);
}

TEST_CASE("simd.loadSimd.scalar")
{
    const float mem = 3.14f;
    float s = 0;
    llama::loadSimd(mem, s);
    CHECK(s == mem);
}

TEST_CASE("simd.loadSimd.simd.stdsimd")
{
    std::array<float, 4> a{};
    std::iota(begin(a), end(a), 1.0f);

    stdx::fixed_size_simd<float, 4> s;
    llama::loadSimd(a[0], s);

    CHECK(s[0] == 1.0f);
    CHECK(s[1] == 2.0f);
    CHECK(s[2] == 3.0f);
    CHECK(s[3] == 4.0f);
}

TEMPLATE_TEST_CASE(
    "simd.loadSimd.record.scalar",
    "",
    llama::mapping::BindAoS<>,
    llama::mapping::BindSoA<>,
    llama::mapping::BindAoSoA<2>,
    llama::mapping::BindAoSoA<32>)
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    const auto mapping = typename TestType::template fn<ArrayExtents, ParticleSimd>(ArrayExtents{1});
    auto view = llama::allocViewUninitialized(mapping);
    iotaFillView(view);

    llama::SimdN<ParticleSimd, 1, stdx::fixed_size_simd> p;
    llama::loadSimd(view(0), p);

    CHECK(p(tag::Pos{}, tag::X{}) == 0);
    CHECK(p(tag::Pos{}, tag::Y{}) == 1);
    CHECK(p(tag::Pos{}, tag::Z{}) == 2);
    CHECK(p(tag::Mass{}) == 3);
    CHECK(p(tag::Vel{}, tag::X{}) == 4);
    CHECK(p(tag::Vel{}, tag::Y{}) == 5);
    CHECK(p(tag::Vel{}, tag::Z{}) == 6);
    CHECK(p(tag::Flags{}, llama::RecordCoord<0>{}) == 7);
    CHECK(p(tag::Flags{}, llama::RecordCoord<1>{}) == 8);
    CHECK(p(tag::Flags{}, llama::RecordCoord<2>{}) == 9);
    CHECK(p(tag::Flags{}, llama::RecordCoord<3>{}) == 10);

    llama::SimdN<Vec3D, 1, stdx::fixed_size_simd> v;
    llama::loadSimd(view(0)(tag::Pos{}), v);

    CHECK(v(tag::X{}) == 0);
    CHECK(v(tag::Y{}) == 1);
    CHECK(v(tag::Z{}) == 2);
}

TEMPLATE_TEST_CASE(
    "simd.loadSimd.record.stdsimd",
    "",
    llama::mapping::BindAoS<>,
    llama::mapping::BindSoA<>,
    llama::mapping::BindAoSoA<2>,
    llama::mapping::BindAoSoA<32>)
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    const auto mapping = typename TestType::template fn<ArrayExtents, ParticleSimd>(ArrayExtents{16});
    auto view = llama::allocViewUninitialized(mapping);
    iotaFillView(view);

    llama::SimdN<ParticleSimd, 4, stdx::fixed_size_simd> p;
    llama::loadSimd(view(0), p);
    CHECK(SimdRange{p(tag::Pos{}, tag::X{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 0.0 + ic * 11.0;
          }}});
    CHECK(SimdRange{p(tag::Pos{}, tag::Y{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 1.0 + ic * 11.0;
          }}});
    CHECK(SimdRange{p(tag::Pos{}, tag::Z{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 2.0 + ic * 11.0;
          }}});

    CHECK(SimdRange{p(tag::Mass{})} == SimdRange{stdx::fixed_size_simd<float, 4>{[](auto ic) {
              return 3.0f + ic * 11.0f;
          }}});

    CHECK(SimdRange{p(tag::Vel{}, tag::X{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 4.0 + ic * 11.0;
          }}});
    CHECK(SimdRange{p(tag::Vel{}, tag::Y{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 5.0 + ic * 11.0;
          }}});
    CHECK(SimdRange{p(tag::Vel{}, tag::Z{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 6.0 + ic * 11.0;
          }}});

    CHECK(
        SimdRange{p(tag::Flags{}, llama::RecordCoord<0>{})}
        == SimdRange{stdx::fixed_size_simd<std::uint8_t, 4>{[](auto ic) -> std::uint8_t { return 7 + ic * 11; }}});
    CHECK(
        SimdRange{p(tag::Flags{}, llama::RecordCoord<1>{})}
        == SimdRange{stdx::fixed_size_simd<std::uint8_t, 4>{[](auto ic) -> std::uint8_t { return 8 + ic * 11; }}});
    CHECK(
        SimdRange{p(tag::Flags{}, llama::RecordCoord<2>{})}
        == SimdRange{stdx::fixed_size_simd<std::uint8_t, 4>{[](auto ic) -> std::uint8_t { return 9 + ic * 11; }}});
    CHECK(
        SimdRange{p(tag::Flags{}, llama::RecordCoord<3>{})}
        == SimdRange{stdx::fixed_size_simd<std::uint8_t, 4>{[](auto ic) -> std::uint8_t { return 10 + ic * 11; }}});

    llama::SimdN<Vec3D, 4, stdx::fixed_size_simd> v;
    llama::loadSimd(view(0)(tag::Pos{}), v);

    CHECK(SimdRange{v(tag::X{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 0.0 + ic * 11.0;
          }}});
    CHECK(SimdRange{v(tag::Y{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 1.0 + ic * 11.0;
          }}});
    CHECK(SimdRange{v(tag::Z{})} == SimdRange{stdx::fixed_size_simd<double, 4>{[](auto ic) {
              return 2.0 + ic * 11.0;
          }}});
}

using Vec2I = llama::Record<llama::Field<tag::X, int>, llama::Field<tag::Y, int>>;
using Vec1I = llama::Record<llama::Field<tag::Y, int>>;

TEMPLATE_TEST_CASE(
    "simd.heterogeneousLoadStore.stdsimd",
    "",
    llama::mapping::BindAoS<>,
    llama::mapping::BindSoA<>,
    llama::mapping::BindAoSoA<2>,
    llama::mapping::BindAoSoA<32>)
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    const auto mapping = typename TestType::template fn<ArrayExtents, Vec2I>(ArrayExtents{2});
    auto view = llama::allocViewUninitialized(mapping);
    iotaFillView(view);

    SECTION("BiggerSimdRecord")
    {
        llama::SimdN<Vec3I, 2, stdx::fixed_size_simd> v{};
        llama::loadSimd(view(0), v);
        CHECK(SimdRange{v(tag::X{})} == SimdRange{stdx::fixed_size_simd<int, 2>{[](auto ic) {
                  return 0 + static_cast<int>(ic) * 2;
              }}});
        CHECK(SimdRange{v(tag::Y{})} == SimdRange{stdx::fixed_size_simd<int, 2>{[](auto ic) {
                  return 1 + static_cast<int>(ic) * 2;
              }}});
        CHECK(SimdRange{v(tag::Z{})} == SimdRange{stdx::fixed_size_simd<int, 2>{}});

        v(tag::X{}) = stdx::fixed_size_simd<int, 2>{[](auto ic) { return static_cast<int>(ic) + 100; }};
        v(tag::Y{}) = stdx::fixed_size_simd<int, 2>{[](auto ic) { return static_cast<int>(ic) + 200; }};
        v(tag::Z{}) = stdx::fixed_size_simd<int, 2>{[](auto ic) { return static_cast<int>(ic) + 300; }};
        llama::storeSimd(v, view(0));
        CHECK(view(0)(tag::X{}) == 100);
        CHECK(view(1)(tag::X{}) == 101);
        CHECK(view(0)(tag::Y{}) == 200);
        CHECK(view(1)(tag::Y{}) == 201);
    }

    SECTION("SmallerSimdRecord")
    {
        llama::SimdN<Vec1I, 2, stdx::fixed_size_simd> v{};
        llama::loadSimd(view(0), v);
        CHECK(SimdRange{v(tag::Y{})} == SimdRange{stdx::fixed_size_simd<int, 2>{[](auto ic) {
                  return 1 + static_cast<int>(ic) * 2;
              }}});

        v(tag::Y{}) = stdx::fixed_size_simd<int, 2>{[](auto ic) { return static_cast<int>(ic) + 1000; }};
        llama::storeSimd(v, view(0));
        CHECK(view(0)(tag::X{}) == 0);
        CHECK(view(1)(tag::X{}) == 2);
        CHECK(view(0)(tag::Y{}) == 1000);
        CHECK(view(1)(tag::Y{}) == 1001);
    }
}

TEST_CASE("simd.storeSimd.scalar")
{
    float mem = 0;
    const float s = 3.14f;
    llama::storeSimd(s, mem);
    CHECK(s == mem);
}

TEST_CASE("simd.storeSimd.simd.stdsimd")
{
    const stdx::fixed_size_simd<float, 4> s([](auto ic) -> float { return ic() + 1; });

    std::array<float, 4> a{};
    llama::storeSimd(s, a[0]);

    CHECK(a[0] == 1.0f);
    CHECK(a[1] == 2.0f);
    CHECK(a[2] == 3.0f);
    CHECK(a[3] == 4.0f);
}

TEMPLATE_TEST_CASE(
    "simd.storeSimd.record.scalar",
    "",
    llama::mapping::BindAoS<>,
    llama::mapping::BindSoA<>,
    llama::mapping::BindAoSoA<2>,
    llama::mapping::BindAoSoA<32>)
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    const auto mapping = typename TestType::template fn<ArrayExtents, ParticleSimd>(ArrayExtents{1});
    auto view = llama::allocViewUninitialized(mapping);

    llama::SimdN<ParticleSimd, 1, stdx::fixed_size_simd> p;
    p(tag::Pos{}, tag::X{}) = 0;
    p(tag::Pos{}, tag::Y{}) = 1;
    p(tag::Pos{}, tag::Z{}) = 2;
    p(tag::Mass{}) = 3;
    p(tag::Vel{}, tag::X{}) = 4;
    p(tag::Vel{}, tag::Y{}) = 5;
    p(tag::Vel{}, tag::Z{}) = 6;
    p(tag::Flags{}, llama::RecordCoord<0>{}) = 7;
    p(tag::Flags{}, llama::RecordCoord<1>{}) = 8;
    p(tag::Flags{}, llama::RecordCoord<2>{}) = 9;
    p(tag::Flags{}, llama::RecordCoord<3>{}) = 10;
    llama::storeSimd(p, view(0));

    CHECK(view(0)(tag::Pos{}, tag::X{}) == 0);
    CHECK(view(0)(tag::Pos{}, tag::Y{}) == 1);
    CHECK(view(0)(tag::Pos{}, tag::Z{}) == 2);
    CHECK(view(0)(tag::Mass{}) == 3);
    CHECK(view(0)(tag::Vel{}, tag::X{}) == 4);
    CHECK(view(0)(tag::Vel{}, tag::Y{}) == 5);
    CHECK(view(0)(tag::Vel{}, tag::Z{}) == 6);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<0>{}) == 7);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<1>{}) == 8);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<2>{}) == 9);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<3>{}) == 10);

    llama::SimdN<Vec3D, 1, stdx::fixed_size_simd> v;
    v(tag::X{}) = 100;
    v(tag::Y{}) = 101;
    v(tag::Z{}) = 102;
    llama::storeSimd(v, view(0)(tag::Vel{}));

    CHECK(view(0)(tag::Pos{}, tag::X{}) == 0);
    CHECK(view(0)(tag::Pos{}, tag::Y{}) == 1);
    CHECK(view(0)(tag::Pos{}, tag::Z{}) == 2);
    CHECK(view(0)(tag::Mass{}) == 3);
    CHECK(view(0)(tag::Vel{}, tag::X{}) == 100);
    CHECK(view(0)(tag::Vel{}, tag::Y{}) == 101);
    CHECK(view(0)(tag::Vel{}, tag::Z{}) == 102);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<0>{}) == 7);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<1>{}) == 8);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<2>{}) == 9);
    CHECK(view(0)(tag::Flags{}, llama::RecordCoord<3>{}) == 10);
}

TEMPLATE_TEST_CASE(
    "simd.storeSimd.record.stdsimd",
    "",
    llama::mapping::BindAoS<>,
    llama::mapping::BindSoA<>,
    llama::mapping::BindAoSoA<2>,
    llama::mapping::BindAoSoA<32>)
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    const auto mapping = typename TestType::template fn<ArrayExtents, ParticleSimd>(ArrayExtents{16});
    auto view = llama::allocViewUninitialized(mapping);

    llama::SimdN<ParticleSimd, 3, stdx::fixed_size_simd> p;
    {
        auto& x = p(tag::Pos{}, tag::X{});
        auto& y = p(tag::Pos{}, tag::Y{});
        auto& z = p(tag::Pos{}, tag::Z{});
        auto& m = p(tag::Mass{});
        x[0] = 1, x[1] = 2, x[2] = 3;
        y[0] = 4, y[1] = 5, y[2] = 6;
        z[0] = 7, z[1] = 8, z[2] = 9;
        m[0] = 80, m[1] = 81, m[2] = 82;
    }
    llama::storeSimd(p, view(0));

    CHECK(view(0)(tag::Pos{}, tag::X{}) == 1);
    CHECK(view(1)(tag::Pos{}, tag::X{}) == 2);
    CHECK(view(2)(tag::Pos{}, tag::X{}) == 3);
    CHECK(view(3)(tag::Pos{}, tag::X{}) == 0);
    CHECK(view(0)(tag::Pos{}, tag::Y{}) == 4);
    CHECK(view(1)(tag::Pos{}, tag::Y{}) == 5);
    CHECK(view(2)(tag::Pos{}, tag::Y{}) == 6);
    CHECK(view(3)(tag::Pos{}, tag::Y{}) == 0);
    CHECK(view(0)(tag::Pos{}, tag::Z{}) == 7);
    CHECK(view(1)(tag::Pos{}, tag::Z{}) == 8);
    CHECK(view(2)(tag::Pos{}, tag::Z{}) == 9);
    CHECK(view(3)(tag::Pos{}, tag::Z{}) == 0);
    CHECK(view(0)(tag::Mass{}) == 80);
    CHECK(view(1)(tag::Mass{}) == 81);
    CHECK(view(2)(tag::Mass{}) == 82);
    CHECK(view(3)(tag::Mass{}) == 0);

    llama::SimdN<Vec3D, 3, stdx::fixed_size_simd> v;
    {
        auto& x = v(tag::X{});
        auto& y = v(tag::Y{});
        auto& z = v(tag::Z{});
        x[0] = 101, x[1] = 102, x[2] = 103;
        y[0] = 104, y[1] = 105, y[2] = 106;
        z[0] = 107, z[1] = 108, z[2] = 109;
    }
    llama::storeSimd(v, view(0)(tag::Pos{}));
    CHECK(view(0)(tag::Pos{}, tag::X{}) == 101);
    CHECK(view(1)(tag::Pos{}, tag::X{}) == 102);
    CHECK(view(2)(tag::Pos{}, tag::X{}) == 103);
    CHECK(view(0)(tag::Pos{}, tag::Y{}) == 104);
    CHECK(view(1)(tag::Pos{}, tag::Y{}) == 105);
    CHECK(view(2)(tag::Pos{}, tag::Y{}) == 106);
    CHECK(view(0)(tag::Pos{}, tag::Z{}) == 107);
    CHECK(view(1)(tag::Pos{}, tag::Z{}) == 108);
    CHECK(view(2)(tag::Pos{}, tag::Z{}) == 109);
}

TEMPLATE_TEST_CASE(
    "simd.simdForEachN.stdsimd",
    "",
    llama::mapping::BindAoS<>,
    llama::mapping::BindSoA<>,
    llama::mapping::BindAoSoA<2>,
    llama::mapping::BindAoSoA<32>)
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 2>;
    for(auto extents : {ArrayExtents{16, 32}, ArrayExtents{11, 7}})
    {
        CAPTURE(extents);
        const auto mapping = typename TestType::template fn<ArrayExtents, Vec2F>(extents);
        auto view = llama::allocViewUninitialized(mapping);
        for(int i = 0; i < extents[0]; i++)
            for(int j = 0; j < extents[1]; j++)
                view(i, j)(tag::X{}) = static_cast<float>(i + j);

        llama::simdForEachN<8, stdx::fixed_size_simd>(
            view,
            [](auto simd)
            {
                using std::sqrt;
                simd(tag::Y{}) = sqrt(simd(tag::X{}));
                return simd; // TODO(bgruber): tag::X{} is redundantly stored
            });

        for(int i = 0; i < extents[0]; i++)
            for(int j = 0; j < extents[1]; j++)
            {
                CAPTURE(i, j);
                CHECK(view(i, j)(tag::Y{}) == std::sqrt(static_cast<float>(i + j)));
            }
    }
}

TEMPLATE_TEST_CASE(
    "simd.simdForEach.stdsimd",
    "",
    llama::mapping::BindAoS<>,
    llama::mapping::BindSoA<>,
    llama::mapping::BindAoSoA<2>,
    llama::mapping::BindAoSoA<32>)
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 2>;
    for(auto extents : {ArrayExtents{16, 32}, ArrayExtents{11, 7}})
    {
        CAPTURE(extents);
        const auto mapping = typename TestType::template fn<ArrayExtents, Vec2F>(extents);
        auto view = llama::allocViewUninitialized(mapping);
        for(int i = 0; i < extents[0]; i++)
            for(int j = 0; j < extents[1]; j++)
                view(i, j)(tag::X{}) = static_cast<float>(i + j);

        llama::simdForEach<stdx::native_simd, stdx::fixed_size_simd>(
            view,
            [](auto simd)
            {
                using std::sqrt;
                simd(tag::Y{}) = sqrt(simd(tag::X{}));
                return simd; // TODO(bgruber): tag::X{} is redundantly stored
            });

        for(int i = 0; i < extents[0]; i++)
            for(int j = 0; j < extents[1]; j++)
            {
                CAPTURE(i, j);
                CHECK(view(i, j)(tag::Y{}) == std::sqrt(static_cast<float>(i + j)));
            }
    }
}
#endif

// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

#include <catch2/catch_approx.hpp>
#include <cmath>
#include <limits>
#include <random>

#define STORE_LOAD_CHECK(in, out)                                                                                     \
    {                                                                                                                 \
        view() = (in);                                                                                                \
        CHECK(view() == (out));                                                                                       \
    }

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.Constant.Constant", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, llama::Constant<3>, llama::Constant<3>>{});
    CHECK(view.mapping().exponentBits() == 3);
    CHECK(view.mapping().mantissaBits() == 3);
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.Constant.Value", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, llama::Constant<3>, unsigned>{{}, {}, 3});
    CHECK(view.mapping().exponentBits() == 3);
    CHECK(view.mapping().mantissaBits() == 3);
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.Value.Constant", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, unsigned, llama::Constant<3>>{{}, 3, {}});
    CHECK(view.mapping().exponentBits() == 3);
    CHECK(view.mapping().mantissaBits() == 3);
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.Value.Value", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, unsigned, unsigned>{{}, 3, 3});
    CHECK(view.mapping().exponentBits() == 3);
    CHECK(view.mapping().mantissaBits() == 3);
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.exponent_only", "", float, double)
{
    // 3 bits for exponent allows range [-2^2-2;2^2-1]
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, unsigned, unsigned>{{}, 3, 0});
    CHECK(view.mapping().exponentBits() == 3);
    CHECK(view.mapping().mantissaBits() == 0);

    const auto inf = std::numeric_limits<TestType>::infinity();
    const auto nan = std::numeric_limits<TestType>::quiet_NaN();

    STORE_LOAD_CHECK(0.000f, 0.0f);
    STORE_LOAD_CHECK(0.222f, 0.0f);
    STORE_LOAD_CHECK(1.111f, 1.0f);
    STORE_LOAD_CHECK(3.333f, 2.0f);
    STORE_LOAD_CHECK(7.777f, 4.0f);
    STORE_LOAD_CHECK(15.555f, 8.0f);
    STORE_LOAD_CHECK(27.777f, inf);
    STORE_LOAD_CHECK(42.222f, inf);
    STORE_LOAD_CHECK(77.777f, inf);
    STORE_LOAD_CHECK(504.444f, inf);

    STORE_LOAD_CHECK(-0.000f, -0.0f);
    STORE_LOAD_CHECK(-0.222f, -0.0f);
    STORE_LOAD_CHECK(-1.111f, -1.0f);
    STORE_LOAD_CHECK(-3.333f, -2.0f);
    STORE_LOAD_CHECK(-7.777f, -4.0f);
    STORE_LOAD_CHECK(-15.555f, -8.0f);
    STORE_LOAD_CHECK(-27.777f, -inf);
    STORE_LOAD_CHECK(-42.222f, -inf);
    STORE_LOAD_CHECK(-77.777f, -inf);
    STORE_LOAD_CHECK(-504.444f, -inf);

    STORE_LOAD_CHECK(inf, inf);
    STORE_LOAD_CHECK(-inf, -inf);

    // nan's need mantissa bits, so they decay to infinities
    STORE_LOAD_CHECK(nan, inf);
#ifndef __NVCOMPILER
    // it is implementation defined whether nan retains the sign bit
    STORE_LOAD_CHECK(-nan, -inf);
#endif
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, llama::Constant<3>, llama::Constant<3>>{});

    const auto inf = std::numeric_limits<TestType>::infinity();
    const auto nan = std::numeric_limits<TestType>::quiet_NaN();

    STORE_LOAD_CHECK(0.000f, 0.0f);
    STORE_LOAD_CHECK(0.222f, 0.0f);
    STORE_LOAD_CHECK(1.111f, 1.0f);
    STORE_LOAD_CHECK(3.333f, 3.25f);
    STORE_LOAD_CHECK(7.777f, 7.5f);
    STORE_LOAD_CHECK(15.555f, 15.0f);
    STORE_LOAD_CHECK(27.777f, inf);
    STORE_LOAD_CHECK(42.222f, inf);
    STORE_LOAD_CHECK(77.777f, inf);
    STORE_LOAD_CHECK(504.444f, inf);

    STORE_LOAD_CHECK(-0.000f, -0.0f);
    STORE_LOAD_CHECK(-0.222f, -0.0f);
    STORE_LOAD_CHECK(-1.111f, -1.0f);
    STORE_LOAD_CHECK(-3.333f, -3.25f);
    STORE_LOAD_CHECK(-7.777f, -7.5f);
    STORE_LOAD_CHECK(-15.555f, -15.0f);
    STORE_LOAD_CHECK(-27.777f, -inf);
    STORE_LOAD_CHECK(-42.222f, -inf);
    STORE_LOAD_CHECK(-77.777f, -inf);
    STORE_LOAD_CHECK(-504.444f, -inf);

    STORE_LOAD_CHECK(inf, inf);
    STORE_LOAD_CHECK(-inf, -inf);

    {
        view() = nan;
        auto f = static_cast<TestType>(view());
        CAPTURE(f);

        CHECK(std::isnan(static_cast<TestType>(view())));
        CHECK(!std::signbit(static_cast<TestType>(view())));
        view() = -nan;
        CHECK(std::isnan(static_cast<TestType>(view())));
#ifndef __NVCOMPILER
        // it is implementation defined whether nan retains the sign bit
        CHECK(std::signbit(static_cast<TestType>(view())));
#endif
    }
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedFloatSoA.ReducedPrecisionComputation",
    "",
    (llama::mapping::
         BitPackedFloatAoS<llama::ArrayExtents<std::size_t, 1000>, Vec3D, llama::Constant<8>, llama::Constant<23>>),
    (llama::mapping::
         BitPackedFloatSoA<llama::ArrayExtents<std::size_t, 1000>, Vec3D, llama::Constant<8>, llama::Constant<23>>) )
{
    constexpr auto n = std::size_t{1000};
    auto view = llama::allocView(llama::mapping::AoS<llama::ArrayExtents<std::size_t, n>, Vec3D>{{}});
    std::default_random_engine engine;
    std::uniform_real_distribution dist{0.0f, 1e20f};
    for(std::size_t i = 0; i < n; i++)
    {
        auto v = dist(engine);
        view(i) = v;
        CHECK(view(i) == v);
    }

    // copy into packed representation
    auto packedView = llama::allocView(TestType{});
    llama::copy(view, packedView);

    // compute on original representation
    for(std::size_t i = 0; i < n; i++)
    {
        auto&& z = view(i)(tag::Z{});
        z = std::sqrt(z);
    }

    // compute on packed representation
    for(std::size_t i = 0; i < n; i++)
    {
        auto&& z = packedView(i)(tag::Z{});
        z = std::sqrt(z);
    }

    for(std::size_t i = 0; i < n; i++)
        CHECK(view(i)(tag::Z{}) == Catch::Approx(packedView(i)(tag::Z{})));
}

TEST_CASE("mapping.BitPackedFloatAoS.blobs")
{
    constexpr auto n = std::size_t{1000};
    using Mapping = llama::mapping::
        BitPackedFloatAoS<llama::ArrayExtents<std::size_t, n>, Vec3D, llama::Constant<3>, llama::Constant<5>>;
    STATIC_REQUIRE(Mapping::blobCount == 1);
    auto mapping = Mapping{};
    CHECK(
        mapping.blobSize(0)
        == llama::roundUpToMultiple(3 * n * (1 + 3 + 5), sizeof(Mapping::StoredIntegral) * CHAR_BIT) / CHAR_BIT);
}

TEST_CASE("mapping.BitPackedFloatSoA.blobs")
{
    constexpr auto n = std::size_t{1000};
    using Mapping = llama::mapping::
        BitPackedFloatSoA<llama::ArrayExtents<std::size_t, n>, Vec3D, llama::Constant<3>, llama::Constant<5>>;
    STATIC_REQUIRE(Mapping::blobCount == 3);
    auto mapping = Mapping{};
    for(auto i = 0u; i < 3; i++)
        CHECK(
            mapping.blobSize(i)
            == llama::roundUpToMultiple(n * (1 + 3 + 5), sizeof(Mapping::StoredIntegral) * CHAR_BIT) / CHAR_BIT);
}

TEST_CASE("mapping.BitPackedFloatAoS.Size")
{
    STATIC_REQUIRE(std::is_empty_v<llama::mapping::BitPackedFloatAoS<
                       llama::ArrayExtents<std::size_t, 16>,
                       float,
                       llama::Constant<7>,
                       llama::Constant<16>>>);
    STATIC_REQUIRE(
        sizeof(llama::mapping::
                   BitPackedFloatAoS<llama::ArrayExtents<std::size_t, 16>, float, llama::Constant<7>, unsigned>{
                       {},
                       {},
                       16})
        == sizeof(unsigned));
    STATIC_REQUIRE(
        sizeof(
            llama::mapping::
                BitPackedFloatAoS<llama::ArrayExtents<std::size_t, 16>, float, unsigned, llama::Constant<16>>{{}, 7})
        == sizeof(unsigned));
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedFloatAoS<llama::ArrayExtents<unsigned, 16>, float>{{}, 7, 16})
        == 2 * sizeof(unsigned));
}

TEST_CASE("mapping.BitPackedFloatSoA.Size")
{
    STATIC_REQUIRE(std::is_empty_v<llama::mapping::BitPackedFloatSoA<
                       llama::ArrayExtents<std::size_t, 16>,
                       float,
                       llama::Constant<7>,
                       llama::Constant<16>>>);
    STATIC_REQUIRE(
        sizeof(llama::mapping::
                   BitPackedFloatSoA<llama::ArrayExtents<std::size_t, 16>, float, llama::Constant<7>, unsigned>{
                       {},
                       {},
                       16})
        == sizeof(unsigned));
    STATIC_REQUIRE(
        sizeof(
            llama::mapping::
                BitPackedFloatSoA<llama::ArrayExtents<std::size_t, 16>, float, unsigned, llama::Constant<16>>{{}, 7})
        == sizeof(unsigned));
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<unsigned, 16>, float>{{}, 7, 16})
        == 2 * sizeof(unsigned));
}


TEST_CASE("mapping.BitPackedFloatSoA.ProxyRef.SwapAndAssign")
{
    auto view = llama::allocView(
        llama::mapping::
            BitPackedFloatSoA<llama::ArrayExtents<int, 2>, float, llama::Constant<5>, llama::Constant<5>>{});
    view(0) = 1.0;
    view(1) = 2.0;
    swap(view(0), view(1));
    CHECK(view(0) == 2.0);
    CHECK(view(1) == 1.0);

    view(0) = view(1);
    CHECK(view(0) == 1.0);
    CHECK(view(1) == 1.0);
}

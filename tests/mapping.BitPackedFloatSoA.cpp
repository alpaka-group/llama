#include "common.hpp"

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
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.Constant.Value", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, llama::Constant<3>, unsigned>{{}, {}, 3});
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.Value.Constant", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, unsigned, llama::Constant<3>>{{}, 3, {}});
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.Value.Value", "", float, double)
{
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, unsigned, unsigned>{{}, 3, 3});
    STORE_LOAD_CHECK(3.333f, 3.25f);
}

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.exponent_only", "", float, double)
{
    // 3 bits for exponent allows range [-2^2-2;2^2-1]
    auto view = llama::allocView(
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType, unsigned, unsigned>{{}, 3, 0});

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
    STORE_LOAD_CHECK(-nan, -inf);
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
        // we cannot call std::isnan here, because that is outside the float_control pragma
        auto isNan = [](TestType f)
        {
            return !(f == f); // NOLINT(misc-redundant-expression)
        };

        view() = nan;
        auto f = static_cast<TestType>(view());
        CAPTURE(f);

        CHECK(isNan(static_cast<TestType>(view())));
        CHECK(!std::signbit(static_cast<TestType>(view())));
        view() = -nan;
        CHECK(isNan(static_cast<TestType>(view())));
        CHECK(std::signbit(static_cast<TestType>(view())));
    }
}

TEST_CASE("mapping.BitPackedFloatSoA.ReducedPrecisionComputation")
{
    constexpr auto N = 1000;
    auto view = llama::allocView(llama::mapping::AoS<llama::ArrayExtents<N>, Vec3D>{{}});
    std::default_random_engine engine;
    std::uniform_real_distribution dist{0.0f, 1e20f};
    for(auto i = 0; i < N; i++)
    {
        auto v = dist(engine);
        view(i) = v;
        CHECK(view(i) == v);
    }

    // copy into packed representation
    auto packedView = llama::allocView(
        llama::mapping::
            BitPackedFloatSoA<llama::ArrayExtents<N>, Vec3D, llama::Constant<8>, llama::Constant<23>>{}); // basically
                                                                                                          // float
    llama::copy(view, packedView);

    // compute on original representation
    for(auto i = 0; i < N; i++)
    {
        auto&& z = view(i)(tag::Z{});
        z = std::sqrt(z);
    }

    // compute on packed representation
    for(auto i = 0; i < N; i++)
    {
        auto&& z = packedView(i)(tag::Z{});
        z = std::sqrt(z);
    }

    for(auto i = 0; i < N; i++)
        CHECK(view(i)(tag::Z{}) == Approx(packedView(i)(tag::Z{})));
}

TEST_CASE("mapping.BitPackedFloatSoA.Size")
{
    STATIC_REQUIRE(std::is_empty_v<
                   llama::mapping::
                       BitPackedFloatSoA<llama::ArrayExtents<16>, float, llama::Constant<7>, llama::Constant<16>>>);
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<16>, float, llama::Constant<7>, unsigned>{
            {},
            {},
            16})
        == sizeof(unsigned));
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<16>, float, unsigned, llama::Constant<16>>{{}, 7})
        == sizeof(unsigned));
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<16>, float>{{}, 7, 16}) == 2 * sizeof(unsigned));
}

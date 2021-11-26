#include "common.hpp"

#include <cmath>
#include <limits>
#include <random>

#define STORE_LOAD_CHECK(in, out)                                                                                     \
    {                                                                                                                 \
        view() = (in);                                                                                                \
        CHECK(view() == (out));                                                                                       \
    }

TEMPLATE_TEST_CASE("mapping.BitPackedFloatSoA.exponent_only", "", float, double)
{
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();

    // 3 bits for exponent allows range [-2^2-2;2^2-1]
    auto view = llama::allocView(llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType>{3, 0, {}});

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
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();

    auto view = llama::allocView(llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<>, TestType>{3, 3, {}});
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
#ifdef __INTEL_LLVM_COMPILER
#    pragma float_control(precise, on)
#endif
        view() = nan;
        CHECK(std::isnan(static_cast<TestType>(view())));
        CHECK(!std::signbit(static_cast<TestType>(view())));
        view() = -nan;
        CHECK(std::isnan(static_cast<TestType>(view())));
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
        llama::mapping::BitPackedFloatSoA<llama::ArrayExtents<N>, Vec3D>{8, 23, {}}); // basically float
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

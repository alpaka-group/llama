#include "common.hpp"

#include <random>

using SInts = llama::Record<
    llama::Field<std::int8_t, std::int8_t>,
    llama::Field<std::int16_t, std::int16_t>,
    llama::Field<std::int32_t, std::int32_t>,
    llama::Field<std::int64_t, std::int64_t>>;
using UInts = llama::Record<
    llama::Field<std::uint8_t, std::uint8_t>,
    llama::Field<std::uint16_t, std::uint16_t>,
    llama::Field<std::uint32_t, std::uint32_t>,
    llama::Field<std::uint64_t, std::uint64_t>>;

TEST_CASE("mapping.BitPackedIntSoA.Constant.SInts")
{
    // 16 elements * 4 fields = 64 integers, iota produces [0;63], which fits int8_t and into 7 bits
    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, SInts, llama::Constant<7>>{{16}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.BitPackedIntSoA.Value.SInts")
{
    // 16 elements * 4 fields = 64 integers, iota produces [0;63], which fits int8_t and into 7 bits
    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, SInts>{{16}, 7});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.BitPackedIntSoA.Constant.UInts")
{
    // 32 elements * 4 fields = 128 integers, iota produces [0;127], which fits uint8_t and into 7 bits
    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, UInts, llama::Constant<7>>{{32}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.BitPackedIntSoA.Value.UInts")
{
    // 32 elements * 4 fields = 128 integers, iota produces [0;127], which fits uint8_t and into 7 bits
    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, UInts>{{32}, 7});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.BitPackedIntSoA.UInts.Cutoff")
{
    auto view = llama::allocView(llama::mapping::BitPackedIntSoA<llama::ArrayExtents<>, UInts, llama::Constant<3>>{});

    for(auto i = 0; i < 8; i++)
    {
        CAPTURE(i);
        view() = i; // assigns to all fields
        CHECK(view() == i);
    }
    for(auto i = 8; i < 100; i++)
    {
        CAPTURE(i);
        view() = i;
        CHECK(view() == i % 8);
    }
}

TEST_CASE("mapping.BitPackedIntSoA.SInts.Cutoff")
{
    auto view = llama::allocView(llama::mapping::BitPackedIntSoA<llama::ArrayExtents<>, SInts, llama::Constant<4>>{});

    for(auto i = 0; i < 8; i++)
    {
        CAPTURE(i);
        view() = i; // assigns to all fields
        CHECK(view() == i);
    }
    for(auto i = 8; i < 100; i++)
    {
        CAPTURE(i);
        view() = i;
        CHECK(view() == i % 8);
    }

    for(auto i = 0; i > -8; i--)
    {
        CAPTURE(i);
        view() = i;
        CHECK(view() == i);
    }
    for(auto i = -8; i > -100; i--)
    {
        CAPTURE(i);
        view() = i;
        CHECK(view() == -((-i - 1) % 8) - 1); // [-1;-8]
    }
}

TEST_CASE("mapping.BitPackedIntSoA.SInts.Roundtrip")
{
    constexpr auto N = 1000;
    auto view = llama::allocView(llama::mapping::AoS<llama::ArrayExtents<std::size_t, N>, Vec3I>{});
    std::default_random_engine engine;
    std::uniform_int_distribution dist{-2000, 2000}; // fits into 12 bits
    for(auto i = 0; i < N; i++)
        view(i) = dist(engine);

    // copy into packed representation
    auto packedView = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, N>, Vec3I, llama::Constant<12>>{});
    llama::copy(view, packedView);

    // compute on packed representation
    for(auto i = 0; i < N; i++)
        packedView(i) = packedView(i) + 1;

    // copy into normal representation
    auto view2 = llama::allocView(llama::mapping::AoS<llama::ArrayExtents<std::size_t, N>, Vec3I>{});
    llama::copy(packedView, view2);

    // compute on normal representation
    for(auto i = 0; i < N; i++)
        view2(i) = view2(i) - 1;

    for(auto i = 0; i < N; i++)
        CHECK(view(i) == view2(i));
}

TEST_CASE("mapping.BitPackedIntSoA.bool")
{
    // pack 32 bools into 4 bytes
    const auto n = 32;
    const auto mapping
        = llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, bool, llama::Constant<1>>{{n}};
    CHECK(mapping.blobSize(0) == n / CHAR_BIT);
    auto view = llama::allocView(mapping);
    for(auto i = 0; i < n; i++)
        view(i) = i % 2 == 0;
    for(auto i = 0; i < n; i++)
        CHECK(view(i) == (i % 2 == 0));
}

namespace
{
    enum Grades
    {
        A,
        B,
        C,
        D,
        E,
        F
    };

    enum class GradesClass
    {
        A,
        B,
        C,
        D,
        E,
        F
    };
} // namespace

TEMPLATE_TEST_CASE("mapping.BitPackedIntSoA.Enum", "", Grades, GradesClass)
{
    using Enum = TestType;

    using StoredIntegral =
        typename llama::mapping::internal::MakeUnsigned<llama::mapping::internal::LargestIntegral<Enum>>::type;
    STATIC_REQUIRE(std::is_same_v<StoredIntegral, unsigned>);

    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, Enum, llama::Constant<3>>{{6}});
    view(0) = Enum::A;
    view(1) = Enum::B;
    view(2) = Enum::C;
    view(3) = Enum::D;
    view(4) = Enum::E;
    view(5) = Enum::F;

    CHECK(view(0) == Enum::A);
    CHECK(view(1) == Enum::B);
    CHECK(view(2) == Enum::C);
    CHECK(view(3) == Enum::D);
    CHECK(view(4) == Enum::E);
    CHECK(view(5) == Enum::F);
}

TEST_CASE("mapping.BitPackedIntSoA.Size")
{
    STATIC_REQUIRE(std::is_empty_v<
                   llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, 16>, SInts, llama::Constant<7>>>);
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, 16>, SInts>{{}, 7})
        == sizeof(unsigned));
}

// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

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

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.Constant.SInts",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, SInts, llama::Constant<7>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, SInts, llama::Constant<7>>) )
{
    // 16 elements * 4 fields = 64 integers, iota produces [0;63], which fits int8_t and into 7 bits
    auto view = llama::allocView(TestType{{16}});
    CHECK(view.mapping().bits() == 7);
    iotaFillView(view);
    iotaCheckView(view);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.Value.SInts",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, SInts>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, SInts>) )
{
    // 16 elements * 4 fields = 64 integers, iota produces [0;63], which fits int8_t and into 7 bits
    auto view = llama::allocView(TestType{{16}, 7});
    CHECK(view.mapping().bits() == 7);
    iotaFillView(view);
    iotaCheckView(view);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.Constant.UInts",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, UInts, llama::Constant<7>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, UInts, llama::Constant<7>>) )
{
    // 32 elements * 4 fields = 128 integers, iota produces [0;127], which fits uint8_t and into 7 bits
    auto view = llama::allocView(TestType{{32}});
    CHECK(view.mapping().bits() == 7);
    iotaFillView(view);
    iotaCheckView(view);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.Value.UInts",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, UInts>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, UInts>) )
{
    // 32 elements * 4 fields = 128 integers, iota produces [0;127], which fits uint8_t and into 7 bits
    auto view = llama::allocView(TestType{{32}, 7});
    CHECK(view.mapping().bits() == 7);
    iotaFillView(view);
    iotaCheckView(view);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.UInts.Cutoff",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<>, UInts, llama::Constant<3>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<>, UInts, llama::Constant<3>>) )
{
    auto view = llama::allocView(TestType{});

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

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.SInts.Cutoff",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<>, SInts, llama::Constant<4>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<>, SInts, llama::Constant<4>>) )
{
    auto view = llama::allocView(TestType{});

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

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.SInts.Roundtrip",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, 1000>, Vec3I, llama::Constant<12>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<std::size_t, 1000>, Vec3I, llama::Constant<12>>) )
{
    constexpr auto n = 1000;
    auto view = llama::allocView(llama::mapping::AoS<llama::ArrayExtents<std::size_t, n>, Vec3I>{});
    std::default_random_engine engine;
    std::uniform_int_distribution dist{-2000, 2000}; // fits into 12 bits
    for(auto i = 0; i < n; i++)
        view(i) = dist(engine);

    // copy into packed representation
    auto packedView = llama::allocView(TestType{});
    llama::copy(view, packedView);

    // compute on packed representation
    for(auto i = 0; i < n; i++)
        packedView(i) = packedView(i) + 1;

    // copy into normal representation
    auto view2 = llama::allocView(llama::mapping::AoS<llama::ArrayExtents<std::size_t, n>, Vec3I>{});
    llama::copy(packedView, view2);

    // compute on normal representation
    for(auto i = 0; i < n; i++)
        view2(i) = view2(i) - 1;

    for(auto i = 0; i < n; i++)
        CHECK(view(i) == view2(i));
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.bool",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, bool, llama::Constant<1>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, bool, llama::Constant<1>>) )
{
    // pack 32 bools into 4 bytes
    const auto n = 32;
    const auto mapping = TestType{{n}};
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

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.Enum",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, Grades, llama::Constant<3>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, Grades, llama::Constant<3>>),
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<std::size_t, 1>, GradesClass, llama::Constant<3>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtentsDynamic<std::size_t, 1>, GradesClass, llama::Constant<3>>) )
{
    using Enum = typename TestType::RecordDim;

    auto view = llama::allocView(TestType{{6}});
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

TEST_CASE("mapping.BitPackedIntAoS.Size")
{
    STATIC_REQUIRE(std::is_empty_v<
                   llama::mapping::BitPackedIntAoS<llama::ArrayExtents<std::size_t, 16>, SInts, llama::Constant<7>>>);
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedIntAoS<llama::ArrayExtents<unsigned, 16>, SInts>{{}, 7}) == sizeof(unsigned));
}

TEST_CASE("mapping.BitPackedIntSoA.Size")
{
    STATIC_REQUIRE(std::is_empty_v<
                   llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, 16>, SInts, llama::Constant<7>>>);
    STATIC_REQUIRE(
        sizeof(llama::mapping::BitPackedIntSoA<llama::ArrayExtents<unsigned, 16>, SInts>{{}, 7}) == sizeof(unsigned));
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.FullBitWidth.16",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<>, std::uint16_t, llama::Constant<16>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<>, std::uint16_t, llama::Constant<16>>) )
{
    // this could detect bugs when shifting integers by their bit-width
    auto view = llama::allocView(TestType{});

    constexpr std::uint16_t value = 0xAABB;
    view() = value;
    CHECK(view() == value);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.FullBitWidth.32",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<>, std::uint32_t, llama::Constant<32>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<>, std::uint32_t, llama::Constant<32>>) )
{
    // this could detect bugs when shifting integers by their bit-width
    auto view = llama::allocView(TestType{});

    constexpr std::uint32_t value = 0xAABBCCDD;
    view() = value;
    CHECK(view() == value);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.FullBitWidth.64",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<>, std::uint64_t, llama::Constant<64>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<>, std::uint64_t, llama::Constant<64>>) )
{
    // this could detect bugs when shifting integers by their bit-width
    auto view = llama::allocView(TestType{});

    constexpr std::uint64_t value = 0xAABBCCDDEEFF8899;
    view() = value;
    CHECK(view() == value);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedIntRef.SwapAndAssign",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<int, 2>, int, llama::Constant<3>>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<int, 2>, int, llama::Constant<3>>) )
{
    auto view = llama::allocView(TestType{});
    view(0) = 1;
    view(1) = 2;
    swap(view(0), view(1));
    CHECK(view(0) == 2);
    CHECK(view(1) == 1);

    view(0) = view(1);
    CHECK(view(0) == 1.0);
    CHECK(view(1) == 1.0);
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.ValidateBitsSmallerThanFieldType",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, 16>, UInts, unsigned>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<std::size_t, 16>, UInts, unsigned>) )
{
    // 11 bits are larger than the uint8_t field type
    CHECK_THROWS(TestType{{}, 11});
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.ValidateBitsSmallerThanStorageIntegral",
    "",
    (llama::mapping::BitPackedIntSoA<
        llama::ArrayExtents<std::size_t, 16>,
        std::uint32_t,
        unsigned,
        llama::mapping::SignBit::Keep,
        llama::mapping::LinearizeArrayDimsCpp,
        std::uint32_t>),
    (llama::mapping::BitPackedIntAoS<
        llama::ArrayExtents<std::size_t, 16>,
        std::uint32_t,
        unsigned,
        llama::mapping::SignBit::Keep,
        llama::mapping::LinearizeArrayDimsCpp,
        llama::mapping::FlattenRecordDimInOrder,
        std::uint32_t>) )
{
    CHECK_THROWS(TestType{{}, 40});
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.ValidateBitsNotZero",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, 16>, UInts, unsigned>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<std::size_t, 16>, UInts, unsigned>) )
{
    CHECK_THROWS(TestType{{}, 0});
}

TEMPLATE_TEST_CASE(
    "mapping.BitPackedInt.ValidateBitsAtLeast2WithSignBit",
    "",
    (llama::mapping::BitPackedIntSoA<llama::ArrayExtents<std::size_t, 16>, SInts, unsigned>),
    (llama::mapping::BitPackedIntAoS<llama::ArrayExtents<std::size_t, 16>, SInts, unsigned>) )
{
    CHECK_THROWS(TestType{{}, 1});
}

TEMPLATE_TEST_CASE(
    "mapping.bitpack",
    "",
    std::int8_t,
    std::int16_t,
    std::int32_t,
    std::int64_t,
    std::uint8_t,
    std::uint16_t,
    std::uint32_t,
    std::uint64_t)
{
    using Integral = TestType;
    boost::mp11::mp_for_each<boost::mp11::mp_list<std::uint32_t, std::uint64_t>>(
        [](auto si)
        {
            using StoredIntegral = decltype(si);
            if constexpr(sizeof(StoredIntegral) >= sizeof(TestType))
            {
                std::vector<StoredIntegral> blob(sizeof(Integral) * 32);

                // positive numbers: 5 bits are required to store values from 0..31, +1 for sign bit
                for(StoredIntegral bitCount = 5; bitCount <= sizeof(Integral) * CHAR_BIT; bitCount++)
                {
                    for(Integral i = 0; i < 32; i++)
                        llama::mapping::internal::bitpack<false>(
                            blob.data(),
                            static_cast<StoredIntegral>(i * bitCount),
                            bitCount,
                            i);

                    for(Integral i = 0; i < 32; i++)
                        CHECK(
                            llama::mapping::internal::bitunpack<false, Integral>(
                                blob.data(),
                                static_cast<StoredIntegral>(i * bitCount),
                                bitCount)
                            == i);
                }

                if constexpr(std::is_signed_v<Integral>)
                {
                    // negative numbers: 5 bits are required to store values from -32..-1, +1 for sign bit
                    for(StoredIntegral bitCount = 5 + 1; bitCount <= sizeof(Integral) * CHAR_BIT; bitCount++)
                    {
                        for(Integral i = 0; i < 32; i++)
                            llama::mapping::internal::bitpack<true>(
                                blob.data(),
                                static_cast<StoredIntegral>(i * bitCount),
                                bitCount,
                                i - 32);

                        for(Integral i = 0; i < 32; i++)
                            CHECK(
                                llama::mapping::internal::bitunpack<true, Integral>(
                                    blob.data(),
                                    static_cast<StoredIntegral>(i * bitCount),
                                    bitCount)
                                == i - 32);
                    }
                }
            }
        });
}

TEMPLATE_TEST_CASE(
    "mapping.bitpack.1bit",
    "",
    std::int8_t,
    std::int16_t,
    std::int32_t,
    std::int64_t,
    std::uint8_t,
    std::uint16_t,
    std::uint32_t,
    std::uint64_t)
{
    using Integral = TestType;
    boost::mp11::mp_for_each<boost::mp11::mp_list<std::uint32_t, std::uint64_t>>(
        [](auto si)
        {
            using StoredIntegral = decltype(si);
            if constexpr(sizeof(StoredIntegral) >= sizeof(TestType))
            {
                constexpr auto bitsToWrite = 127;
                std::vector<StoredIntegral> blob(
                    llama::divCeil(std::size_t{bitsToWrite}, sizeof(StoredIntegral) * CHAR_BIT));

                for(Integral i = 0; i < bitsToWrite; i++)
                    llama::mapping::internal::bitpack<false>(
                        blob.data(),
                        static_cast<StoredIntegral>(i),
                        StoredIntegral{1},
                        static_cast<Integral>(i % 2));

                for(Integral i = 0; i < bitsToWrite; i++)
                    CHECK(
                        llama::mapping::internal::bitunpack<false, Integral>(
                            blob.data(),
                            static_cast<StoredIntegral>(i),
                            StoredIntegral{1})
                        == static_cast<Integral>(i % 2));
            }
        });
}

TEMPLATE_TEST_CASE(
    "mapping.bitpack1",
    "",
    std::int8_t,
    std::int16_t,
    std::int32_t,
    std::int64_t,
    std::uint8_t,
    std::uint16_t,
    std::uint32_t,
    std::uint64_t)
{
    using Integral = TestType;
    boost::mp11::mp_for_each<boost::mp11::mp_list<std::uint32_t, std::uint64_t>>(
        []<typename StoredIntegral>(StoredIntegral)
        {
            if constexpr(sizeof(StoredIntegral) >= sizeof(TestType))
            {
                constexpr auto bitsToWrite = 127;
                std::vector<StoredIntegral> blob(
                    llama::divCeil(std::size_t{bitsToWrite}, sizeof(StoredIntegral) * CHAR_BIT));

                for(Integral i = 0; i < bitsToWrite; i++)
                    llama::mapping::internal::bitpack1(
                        blob.data(),
                        static_cast<StoredIntegral>(i),
                        static_cast<Integral>(i % 2));

                for(Integral i = 0; i < bitsToWrite; i++)
                {
                    CAPTURE(i);
                    [[maybe_unused]] auto r
                        = llama::mapping::internal::bitunpack1<Integral>(blob.data(), static_cast<StoredIntegral>(i));
                    assert(r == static_cast<Integral>(i % 2));
                    CHECK(
                        llama::mapping::internal::bitunpack1<Integral>(blob.data(), static_cast<StoredIntegral>(i))
                        == static_cast<Integral>(i % 2));
                }
            }
        });
}

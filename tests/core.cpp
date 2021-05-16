#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Flags {};
    struct Weight {};
} // namespace tag

using XYZ = llama::Record<
    llama::Field<tag::X, double>,
    llama::Field<tag::Y, double>,
    llama::Field<tag::Z, double>
>;
using Particle = llama::Record<
    llama::Field<tag::Pos, XYZ>,
    llama::Field<tag::Weight, float>,
    llama::Field<llama::NoName, int>,
    llama::Field<tag::Vel,llama::Record<
        llama::Field<tag::Z, double>,
        llama::Field<tag::X, double>
    >>,
    llama::Field<tag::Flags, bool[4]>
>;

using Other = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::Z, float>,
        llama::Field<tag::Y, float>
    >>
>;
// clang-format on

TEST_CASE("prettyPrintType")
{
    auto str = prettyPrintType(Particle());
#ifdef _WIN32
    replace_all(str, "__int64", "long");
#endif
    const auto ref = R"(llama::Record<
    llama::Field<
        tag::Pos,
        llama::Record<
            llama::Field<
                tag::X,
                double
            >,
            llama::Field<
                tag::Y,
                double
            >,
            llama::Field<
                tag::Z,
                double
            >
        >
    >,
    llama::Field<
        tag::Weight,
        float
    >,
    llama::Field<
        llama::NoName,
        int
    >,
    llama::Field<
        tag::Vel,
        llama::Record<
            llama::Field<
                tag::Z,
                double
            >,
            llama::Field<
                tag::X,
                double
            >
        >
    >,
    llama::Field<
        tag::Flags,
        bool [4]
    >
>)";
    CHECK(str == ref);
}

TEST_CASE("sizeOf")
{
    STATIC_REQUIRE(llama::sizeOf<float> == 4);
    STATIC_REQUIRE(llama::sizeOf<XYZ> == 24);
    STATIC_REQUIRE(llama::sizeOf<Particle> == 52);
}

TEST_CASE("sizeOf.Align")
{
    STATIC_REQUIRE(llama::sizeOf<float, true> == 4);
    STATIC_REQUIRE(llama::sizeOf<XYZ, true> == 24);
    STATIC_REQUIRE(llama::sizeOf<Particle, true> == 56);
}

TEST_CASE("offsetOf")
{
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 0>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 1>> == 8);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 2>> == 16);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<1>> == 24);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2>> == 28);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3>> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 0>> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 1>> == 40);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4>> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 0>> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 1>> == 49);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 2>> == 50);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 3>> == 51);
}

TEST_CASE("offsetOf.Align")
{
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 1>, true> == 8);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 2>, true> == 16);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<1>, true> == 24);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2>, true> == 28);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3>, true> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 0>, true> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 1>, true> == 40);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4>, true> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 0>, true> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 1>, true> == 49);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 2>, true> == 50);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<4, 3>, true> == 51);
}

template <int i>
struct S;

TEST_CASE("alignment")
{
    using RD = llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, bool>,
        llama::Field<tag::Weight, std::uint16_t>>;

    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<>, false> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<0>, false> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<1>, false> == 4);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<2>, false> == 12);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<3>, false> == 13);
    STATIC_REQUIRE(llama::sizeOf<RD, false> == 15);

    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<1>, true> == 8); // aligned
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<2>, true> == 16);
    STATIC_REQUIRE(llama::offsetOf<RD, llama::RecordCoord<3>, true> == 18); // aligned
    STATIC_REQUIRE(llama::sizeOf<RD, true> == 24);
}

TEST_CASE("GetCoordFromTags")
{
    using namespace llama::literals;
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle                                  >, llama::RecordCoord<    >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos                        >, llama::RecordCoord<0   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::X                >, llama::RecordCoord<0, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Y                >, llama::RecordCoord<0, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Z                >, llama::RecordCoord<0, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Weight                     >, llama::RecordCoord<1   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, llama::NoName                   >, llama::RecordCoord<2   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::Z                >, llama::RecordCoord<3, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::X                >, llama::RecordCoord<3, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags                      >, llama::RecordCoord<4   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<0>>, llama::RecordCoord<4, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<1>>, llama::RecordCoord<4, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<2>>, llama::RecordCoord<4, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<3>>, llama::RecordCoord<4, 3>>);
    // clang-format on
}

TEST_CASE("GetTags")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0, 0>>, boost::mp11::mp_list<llama::NoName, tag::Pos, tag::X >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0   >>, boost::mp11::mp_list<llama::NoName, tag::Pos         >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<    >>, boost::mp11::mp_list<llama::NoName                   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<3, 1>>, boost::mp11::mp_list<llama::NoName, tag::Vel, tag::X >>);
    // clang-format on
}

TEST_CASE("GetTag")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<0, 0>>, tag::X       >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<0   >>, tag::Pos     >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<    >>, llama::NoName>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<3, 1>>, tag::X       >);
    // clang-format on
}

TEST_CASE("hasSameTags")
{
    using PosRecord = llama::GetType<Particle, llama::GetCoordFromTags<Particle, tag::Pos>>;
    using VelRecord = llama::GetType<Particle, llama::GetCoordFromTags<Particle, tag::Vel>>;

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosRecord, // RD A
            llama::RecordCoord<0>, // Local A
            VelRecord, // RD B
            llama::RecordCoord<0> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosRecord, // RD A
            llama::RecordCoord<0>, // Local A
            VelRecord, // RD B
            llama::RecordCoord<1> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<0, 0>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<0, 2>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<3, 0>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            > == false);
}

TEST_CASE("FlatRecordDim")
{
    STATIC_REQUIRE(std::is_same_v<
                   llama::FlatRecordDim<Particle>,
                   boost::mp11::mp_list<double, double, double, float, int, double, double, bool, bool, bool, bool>>);
}

TEST_CASE("flatRecordCoord")
{
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<>> == 0);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0>> == 0);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0, 0>> == 0);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0, 1>> == 1);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<0, 2>> == 2);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<1>> == 3);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2>> == 4);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3>> == 5);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 0>> == 5);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 1>> == 6);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<4>> == 7);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<4, 0>> == 7);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<4, 1>> == 8);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<4, 2>> == 9);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<4, 3>> == 10);
}

// clang-format off
namespace tag
{
    struct A1{};
    struct A2{};
    struct A3{};
} // namespace tag

using Arrays = llama::Record<
    llama::Field<tag::A1, int[3]>,
    llama::Field<tag::A2, llama::Record<
        llama::Field<tag::X, float>
    >[3]>,
    llama::Field<tag::A3, int[2][2]>
>;
// clang-format on

TEST_CASE("arrays")
{
    using namespace llama::literals;

    auto v = llama::allocView(llama::mapping::AoS{llama::ArrayDims{1}, Arrays{}});
    v(0u)(tag::A1{}, 0_RC);
    v(0u)(tag::A1{}, 1_RC);
    v(0u)(tag::A1{}, 2_RC);
    v(0u)(tag::A2{}, 0_RC, tag::X{});
    v(0u)(tag::A2{}, 1_RC, tag::X{});
    v(0u)(tag::A2{}, 2_RC, tag::X{});
    v(0u)(tag::A3{}, 0_RC, 0_RC);
    v(0u)(tag::A3{}, 0_RC, 1_RC);
    v(0u)(tag::A3{}, 1_RC, 0_RC);
    v(0u)(tag::A3{}, 1_RC, 1_RC);
}

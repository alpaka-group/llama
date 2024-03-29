// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

TEST_CASE("flatSizeOf")
{
    STATIC_REQUIRE(llama::flatSizeOf<mp_list<std::int32_t>, true, true> == 4);
    STATIC_REQUIRE(llama::flatSizeOf<mp_list<char, std::int32_t, char>, true, true> == 12);
    STATIC_REQUIRE(llama::flatSizeOf<mp_list<char, std::int32_t, char>, false, true> == 6);
    STATIC_REQUIRE(llama::flatSizeOf<mp_list<char, std::int32_t, char>, true, false> == 9);
    STATIC_REQUIRE(llama::flatSizeOf<mp_list<char, std::int32_t, char>, false, false> == 6);
    STATIC_REQUIRE(llama::flatSizeOf<mp_list<>, true, true> == 0);
}

TEST_CASE("prettyPrintType")
{
    auto str = prettyPrintType<Particle>();
#ifdef _WIN32
    replaceAll(str, "__int64", "long");
#endif
    const auto* const ref = R"(llama::Record<
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
        tag::Mass,
        float
    >,
    llama::Field<
        tag::Vel,
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
        tag::Flags,
        bool [4]
    >
>)";
    CHECK(str == ref);
}

TEST_CASE("sizeOf")
{
    STATIC_REQUIRE(llama::sizeOf<float> == 4);
    STATIC_REQUIRE(llama::sizeOf<Vec3D> == 24);
    STATIC_REQUIRE(llama::sizeOf<Vec2F> == 8);
    STATIC_REQUIRE(llama::sizeOf<Particle> == 56);
}

TEST_CASE("sizeOf.Align")
{
    STATIC_REQUIRE(llama::sizeOf<float, true> == 4);
    STATIC_REQUIRE(llama::sizeOf<Vec3D, true> == 24);
    STATIC_REQUIRE(llama::sizeOf<Vec2F, true> == 8);
    STATIC_REQUIRE(llama::sizeOf<Particle, true> == 64);
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
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 0>> == 28);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 1>> == 36);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 2>> == 44);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3>> == 52);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 0>> == 52);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 1>> == 53);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 2>> == 54);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 3>> == 55);
}

TEST_CASE("offsetOf.Align")
{
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 0>, true> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 1>, true> == 8);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<0, 2>, true> == 16);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<1>, true> == 24);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2>, true> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 0>, true> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 1>, true> == 40);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<2, 2>, true> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3>, true> == 56);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 0>, true> == 56);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 1>, true> == 57);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 2>, true> == 58);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::RecordCoord<3, 3>, true> == 59);
}

TEST_CASE("alignOf")
{
    STATIC_REQUIRE(llama::alignOf<std::byte> == 1);
    STATIC_REQUIRE(llama::alignOf<unsigned short> == 2);
    STATIC_REQUIRE(llama::alignOf<float> == 4);
    STATIC_REQUIRE(llama::alignOf<Vec3D> == 8);
    STATIC_REQUIRE(llama::alignOf<Vec2F> == 4);
    STATIC_REQUIRE(llama::alignOf<Particle> == 8);

    struct alignas(32) Overaligned
    {
    };

    using OveralignedRD = llama::Record<llama::Field<int, Overaligned>>;

    STATIC_REQUIRE(llama::alignOf<Overaligned> == 32);
    STATIC_REQUIRE(llama::alignOf<OveralignedRD> == 32);
}

namespace
{
    // clang-format off
    using Other = llama::Record<
        llama::Field<tag::Pos, llama::Record<
            llama::Field<tag::Z, float>,
            llama::Field<tag::Y, float>
        >>
    >;
    // clang-format on
} // namespace

TEST_CASE("flatFieldCount")
{
    STATIC_REQUIRE(llama::flatFieldCount<int> == 1);
    STATIC_REQUIRE(llama::flatFieldCount<Vec3D> == 3);
    STATIC_REQUIRE(llama::flatFieldCount<Particle> == 11);
    STATIC_REQUIRE(llama::flatFieldCount<Other> == 2);
}

TEST_CASE("flatFieldCountBefore")
{
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<0, Particle> == 0);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<1, Particle> == 3);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<2, Particle> == 4);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<3, Particle> == 7);
    STATIC_REQUIRE(llama::internal::flatFieldCountBefore<4, Particle> == 11);
}

TEST_CASE("alignment")
{
    using RD = llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, bool>,
        llama::Field<tag::Mass, std::uint16_t>>;

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
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle                                   >, llama::RecordCoord<    >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos                         >, llama::RecordCoord<0   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::X                 >, llama::RecordCoord<0, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Y                 >, llama::RecordCoord<0, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Z                 >, llama::RecordCoord<0, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Mass                        >, llama::RecordCoord<1   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::X                 >, llama::RecordCoord<2, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::Y                 >, llama::RecordCoord<2, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::Z                 >, llama::RecordCoord<2, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags                       >, llama::RecordCoord<3   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<0>>, llama::RecordCoord<3, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<1>>, llama::RecordCoord<3, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<2>>, llama::RecordCoord<3, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::RecordCoord<3>>, llama::RecordCoord<3, 3>>);
    // clang-format on
}

TEST_CASE("GetCoordFromTags.List")
{
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, mp_list<>>, llama::RecordCoord<>>);
    STATIC_REQUIRE(
        std::is_same_v<llama::GetCoordFromTags<Particle, mp_list<tag::Vel, tag::Z>>, llama::RecordCoord<2, 2>>);
}

TEST_CASE("GetCoordFromTags.RecordCoord")
{
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, llama::RecordCoord<>>, llama::RecordCoord<>>);
    STATIC_REQUIRE(
        std::is_same_v<llama::GetCoordFromTags<Particle, llama::RecordCoord<2, 2>>, llama::RecordCoord<2, 2>>);
}

TEST_CASE("GetType")
{
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<>>, Particle>);

    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<0>>, Vec3D>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<0, 0>>, double>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<0, 1>>, double>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<0, 2>>, double>);

    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<1>>, float>);

    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<2>>, Vec3D>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<2, 0>>, double>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<2, 1>>, double>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<2, 2>>, double>);

    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<3>>, bool[4]>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<3, 0>>, bool>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<3, 1>>, bool>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<3, 2>>, bool>);
    STATIC_REQUIRE(std::is_same_v<llama::GetType<Particle, llama::RecordCoord<3, 3>>, bool>);
}

TEST_CASE("GetTags")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0, 0>>, mp_list<tag::Pos, tag::X >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0   >>, mp_list<tag::Pos         >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<    >>, mp_list<                 >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<2, 1>>, mp_list<tag::Vel, tag::Y >>);
    // clang-format on
}

TEST_CASE("GetTag")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<0, 0>>, tag::X       >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<0   >>, tag::Pos     >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<    >>, llama::NoName>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::RecordCoord<2, 1>>, tag::Y       >);
    // clang-format on
}

TEST_CASE("LeafRecordCoords")
{
    STATIC_REQUIRE(std::is_same_v<
                   llama::LeafRecordCoords<Particle>,
                   mp_list<
                       llama::RecordCoord<0, 0>,
                       llama::RecordCoord<0, 1>,
                       llama::RecordCoord<0, 2>,
                       llama::RecordCoord<1>,
                       llama::RecordCoord<2, 0>,
                       llama::RecordCoord<2, 1>,
                       llama::RecordCoord<2, 2>,
                       llama::RecordCoord<3, 0>,
                       llama::RecordCoord<3, 1>,
                       llama::RecordCoord<3, 2>,
                       llama::RecordCoord<3, 3>>>);
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
            llama::RecordCoord<1> // Local B
            >
        == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosRecord, // RD A
            llama::RecordCoord<0>, // Local A
            VelRecord, // RD B
            llama::RecordCoord<0> // Local B
            >
        == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<0, 0>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            >
        == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<0, 2>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            >
        == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // RD A
            llama::RecordCoord<3, 0>, // Local A
            Other, // RD B
            llama::RecordCoord<0, 0> // Local B
            >
        == false);
}

TEST_CASE("FlatRecordDim")
{
    STATIC_REQUIRE(std::is_same_v<
                   llama::FlatRecordDim<Particle>,
                   mp_list<double, double, double, float, double, double, double, bool, bool, bool, bool>>);
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
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2, 0>> == 4);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2, 1>> == 5);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<2, 2>> == 6);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3>> == 7);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 0>> == 7);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 1>> == 8);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 2>> == 9);
    STATIC_REQUIRE(llama::flatRecordCoord<Particle, llama::RecordCoord<3, 3>> == 10);
}

TEST_CASE("TransformLeaves")
{
    STATIC_REQUIRE(std::is_same_v<llama::TransformLeaves<int, std::add_pointer_t>, int*>);
    STATIC_REQUIRE(std::is_same_v<llama::TransformLeaves<int[3], std::add_pointer_t>, int* [3]>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::TransformLeaves<llama::Record<llama::Field<int, int>>, std::add_pointer_t>,
                   llama::Record<llama::Field<int, int*>>>);

    using Vec3DTransformed
        = llama::Record<llama::Field<tag::X, double*>, llama::Field<tag::Y, double*>, llama::Field<tag::Z, double*>>;
    using ParticleTransformed = llama::Record<
        llama::Field<tag::Pos, Vec3DTransformed>,
        llama::Field<tag::Mass, float*>,
        llama::Field<tag::Vel, Vec3DTransformed>,
        llama::Field<tag::Flags, bool* [4]>>;
    STATIC_REQUIRE(std::is_same_v<llama::TransformLeaves<Particle, std::add_pointer_t>, ParticleTransformed>);
}

TEST_CASE("TransformLeavesWithCoord")
{
    STATIC_REQUIRE(
        std::is_same_v<llama::TransformLeavesWithCoord<int, std::pair>, std::pair<llama::RecordCoord<>, int>>);
    STATIC_REQUIRE(
        std::is_same_v<llama::TransformLeavesWithCoord<int[3], std::pair>, std::pair<llama::RecordCoord<0>, int>[3]>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::TransformLeavesWithCoord<llama::Record<llama::Field<int, int>>, std::pair>,
                   llama::Record<llama::Field<int, std::pair<llama::RecordCoord<0>, int>>>>);

    using PosTransformed = llama::Record<
        llama::Field<tag::X, std::pair<llama::RecordCoord<0, 0>, double>>,
        llama::Field<tag::Y, std::pair<llama::RecordCoord<0, 1>, double>>,
        llama::Field<tag::Z, std::pair<llama::RecordCoord<0, 2>, double>>>;
    using VelTransformed = llama::Record<
        llama::Field<tag::X, std::pair<llama::RecordCoord<2, 0>, double>>,
        llama::Field<tag::Y, std::pair<llama::RecordCoord<2, 1>, double>>,
        llama::Field<tag::Z, std::pair<llama::RecordCoord<2, 2>, double>>>;
    using ParticleTransformed = llama::Record<
        llama::Field<tag::Pos, PosTransformed>,
        llama::Field<tag::Mass, std::pair<llama::RecordCoord<1>, float>>,
        llama::Field<tag::Vel, VelTransformed>,
        llama::Field<tag::Flags, std::pair<llama::RecordCoord<3, 0>, bool>[4]>>;
    STATIC_REQUIRE(std::is_same_v<llama::TransformLeavesWithCoord<Particle, std::pair>, ParticleTransformed>);
}

TEST_CASE("MergedRecordDims")
{
    STATIC_REQUIRE(std::is_same_v<llama::MergedRecordDims<llama::Record<>, llama::Record<>>, llama::Record<>>);

    using R1 = llama::Record<llama::Field<tag::X, int>>;
    using R2 = llama::Record<llama::Field<tag::Y, int>>;
    STATIC_REQUIRE(
        std::is_same_v<llama::MergedRecordDims<llama::Record<>, R2>, llama::Record<llama::Field<tag::Y, int>>>);
    STATIC_REQUIRE(
        std::is_same_v<llama::MergedRecordDims<R1, llama::Record<>>, llama::Record<llama::Field<tag::X, int>>>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::MergedRecordDims<R1, R2>,
                   llama::Record<llama::Field<tag::X, int>, llama::Field<tag::Y, int>>>);
    STATIC_REQUIRE(std::is_same_v<llama::MergedRecordDims<Vec3I, Vec3I>, Vec3I>);
    STATIC_REQUIRE(std::is_same_v<llama::MergedRecordDims<Particle, Particle>, Particle>);
    STATIC_REQUIRE(std::is_same_v<llama::MergedRecordDims<int[3], int[5]>, int[5]>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::MergedRecordDims<Particle, Vec3I>,
                   llama::Record<
                       llama::Field<tag::Pos, Vec3D>,
                       llama::Field<tag::Mass, float>,
                       llama::Field<tag::Vel, Vec3D>,
                       llama::Field<tag::Flags, bool[4]>,
                       llama::Field<tag::X, int>,
                       llama::Field<tag::Y, int>,
                       llama::Field<tag::Z, int>>>);
}

TEST_CASE("CopyConst")
{
    STATIC_REQUIRE(std::is_same_v<llama::CopyConst<int, float>, float>);
    STATIC_REQUIRE(std::is_same_v<llama::CopyConst<const int, float>, const float>);
    STATIC_REQUIRE(std::is_same_v<llama::CopyConst<int, const float>, const float>);
    STATIC_REQUIRE(std::is_same_v<llama::CopyConst<const int, const float>, const float>);
}

namespace
{
    struct AnonNs
    {
    };

    namespace ns
    {
        struct AnonNs2
        {
        };
    } // namespace ns
} // namespace

struct A
{
};

TEST_CASE("qualifiedTypeName")
{
    CHECK(llama::qualifiedTypeName<int> == "int");
    CHECK(llama::qualifiedTypeName<A> == "A");
    CHECK(llama::qualifiedTypeName<tag::A> == "tag::A");
    CHECK(llama::qualifiedTypeName<tag::Normal> == "tag::Normal");
    CHECK(llama::qualifiedTypeName<unsigned int> == "unsigned int");
    CHECK(llama::qualifiedTypeName<std::remove_const<int>> == "std::remove_const<int>");
    CHECK(
        llama::qualifiedTypeName<std::remove_const<std::add_const<std::common_type<int, char>>>>
        == "std::remove_const<std::add_const<std::common_type<int,char>>>");
    CHECK(
        llama::qualifiedTypeName<Vec3D>
        == "llama::Record<llama::Field<tag::X,double>,llama::Field<tag::Y,double>,llama::Field<tag::Z,double>>");

    CHECK(llama::qualifiedTypeName<pmacc::multiMask> == "pmacc::multiMask");
    CHECK(llama::qualifiedTypeName<pmacc::localCellIdx> == "pmacc::localCellIdx");
#if defined(__NVCOMPILER) || defined(_MSC_VER) || (defined(__clang__) && __clang_major__ < 12)
    CHECK(
        llama::qualifiedTypeName<picongpu::position<picongpu::position_pic>>
        == "picongpu::position<picongpu::position_pic,pmacc::pmacc_isAlias>");
#else
    CHECK(
        llama::qualifiedTypeName<picongpu::position<picongpu::position_pic>>
        == "picongpu::position<picongpu::position_pic>");
#endif
    CHECK(llama::qualifiedTypeName<picongpu::momentum> == "picongpu::momentum");
    CHECK(llama::qualifiedTypeName<picongpu::weighting> == "picongpu::weighting");

#if defined(__clang__)
    CHECK(llama::qualifiedTypeName<AnonNs> == "(anonymous namespace)::AnonNs");
    CHECK(llama::qualifiedTypeName<ns::AnonNs2> == "(anonymous namespace)::ns::AnonNs2");
#elif defined(__NVCOMPILER)
    CHECK(llama::qualifiedTypeName<AnonNs> == "<unnamed>::AnonNs");
    CHECK(llama::qualifiedTypeName<ns::AnonNs2> == "<unnamed>::ns::AnonNs2");
#elif defined(__GNUG__)
    CHECK(llama::qualifiedTypeName<AnonNs> == "{anonymous}::AnonNs");
    CHECK(llama::qualifiedTypeName<ns::AnonNs2> == "{anonymous}::ns::AnonNs2");
#elif defined(_MSC_VER)
    CHECK(llama::qualifiedTypeName<AnonNs> == "`anonymous-namespace'::AnonNs");
    CHECK(llama::qualifiedTypeName<ns::AnonNs2> == "`anonymous-namespace'::ns::AnonNs2");
#endif
}

TEST_CASE("structName")
{
    CHECK(llama::structName<int>() == "int");
    CHECK(llama::structName<A>() == "A");
    CHECK(llama::structName(int{}) == "int");
    CHECK(llama::structName<tag::A>() == "A");
    CHECK(llama::structName(tag::A{}) == "A");
    CHECK(llama::structName<tag::Normal>() == "Normal");
    CHECK(llama::structName(tag::Normal{}) == "Normal");
    CHECK(llama::structName<unsigned int>() == "unsigned int");
    CHECK(llama::structName(std::remove_const<int>{}) == "remove_const<int>");
    CHECK(
        llama::structName(std::remove_const<std::add_const<std::common_type<int, char>>>{})
        == "remove_const<add_const<common_type<int,char>>>");
    CHECK(llama::structName(Vec3D{}) == "Record<Field<X,double>,Field<Y,double>,Field<Z,double>>");

    CHECK(llama::structName<pmacc::multiMask>() == "multiMask");
    CHECK(llama::structName<pmacc::localCellIdx>() == "localCellIdx");
#if defined(__NVCOMPILER) || defined(_MSC_VER) || (defined(__clang__) && __clang_major__ < 12)
    CHECK(llama::structName<picongpu::position<picongpu::position_pic>>() == "position<position_pic,pmacc_isAlias>");
#else
    CHECK(llama::structName<picongpu::position<picongpu::position_pic>>() == "position<position_pic>");
#endif
    CHECK(llama::structName<picongpu::momentum>() == "momentum");
    CHECK(llama::structName<picongpu::weighting>() == "weighting");

    CHECK(llama::structName<AnonNs>() == "AnonNs");
    CHECK(llama::structName<ns::AnonNs2>() == "AnonNs2");
}

TEST_CASE("prettyRecordCoord.Particle")
{
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<>{}).empty());
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<0>{}) == "Pos");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<0, 0>{}) == "Pos.X");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<0, 1>{}) == "Pos.Y");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<0, 2>{}) == "Pos.Z");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<1>{}) == "Mass");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<2>{}) == "Vel");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<2, 0>{}) == "Vel.X");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<2, 1>{}) == "Vel.Y");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<2, 2>{}) == "Vel.Z");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<3>{}) == "Flags");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<3, 0>{}) == "Flags[0]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<3, 1>{}) == "Flags[1]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<3, 2>{}) == "Flags[2]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Particle>(llama::RecordCoord<3, 3>{}) == "Flags[3]");

    STATIC_REQUIRE(llama::prettyRecordCoord<Track>(llama::RecordCoord<2>{}) == "NumIALeft");
    STATIC_REQUIRE(llama::prettyRecordCoord<Track>(llama::RecordCoord<2, 0>{}) == "NumIALeft[0]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Track>(llama::RecordCoord<2, 1>{}) == "NumIALeft[1]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Track>(llama::RecordCoord<2, 2>{}) == "NumIALeft[2]");

    using Row = llama::Record<llama::Field<tag::A, double[3]>>;
    using Matrix = llama::Record<llama::Field<tag::B, Row[3]>>;

    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0>{}) == "B");
    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0, 0>{}) == "B[0]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0, 1>{}) == "B[1]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0, 2>{}) == "B[2]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0, 0, 0>{}) == "B[0].A");
    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0, 0, 0, 0>{}) == "B[0].A[0]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0, 0, 0, 1>{}) == "B[0].A[1]");
    STATIC_REQUIRE(llama::prettyRecordCoord<Matrix>(llama::RecordCoord<0, 2, 0, 2>{}) == "B[2].A[2]");
}

TEST_CASE("prettyRecordCoord.picongpu")
{
    CHECK(llama::prettyRecordCoord<picongpu::Frame>(llama::RecordCoord<0>{}) == "multiMask");
    CHECK(llama::prettyRecordCoord<picongpu::Frame>(llama::RecordCoord<1>{}) == "localCellIdx");
#if defined(__NVCOMPILER) || defined(_MSC_VER) || (defined(__clang__) && __clang_major__ < 12)
    CHECK(
        llama::prettyRecordCoord<picongpu::Frame>(llama::RecordCoord<2>{}) == "position<position_pic,pmacc_isAlias>");
#else
    CHECK(llama::prettyRecordCoord<picongpu::Frame>(llama::RecordCoord<2>{}) == "position<position_pic>");
#endif
    CHECK(llama::prettyRecordCoord<picongpu::Frame>(llama::RecordCoord<3>{}) == "momentum");
    CHECK(llama::prettyRecordCoord<picongpu::Frame>(llama::RecordCoord<4>{}) == "weighting");
}

namespace
{
    struct WithValue
    {
        llama::internal::BoxedValue<unsigned> v;
    };

    struct WithValueCtor
    {
        WithValueCtor(int, llama::internal::BoxedValue<double>, int)
        {
        }
    };
} // namespace

TEST_CASE("BoxedValue.implicit_ctor")
{
    [[maybe_unused]] const llama::internal::BoxedValue<unsigned> v1{42};
    [[maybe_unused]] const llama::internal::BoxedValue<unsigned> v2 = 42;
    [[maybe_unused]] const WithValue wv{42};
    [[maybe_unused]] const WithValueCtor wvc1{1, 2.4, 4};
    [[maybe_unused]] const WithValueCtor wvc2{1, 2, 4};
}

namespace
{
    template<typename Value>
    struct ValueConsumer : llama::internal::BoxedValue<Value>
    {
        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        ValueConsumer(Value v) : llama::internal::BoxedValue<Value>(v)
        {
        }

        constexpr auto operator()() const
        {
            return llama::internal::BoxedValue<Value>::value();
        }
    };
} // namespace

TEST_CASE("BoxedValue.Value")
{
    const ValueConsumer<unsigned> vc{1};
    CHECK(vc() == 1);
}

TEST_CASE("BoxedValue.Constant")
{
    const ValueConsumer<llama::Constant<1>> vc{{}};
    CHECK(vc() == 1);
    STATIC_REQUIRE(vc() == 1);
}

TEST_CASE("roundUpToMultiple")
{
    STATIC_REQUIRE(llama::roundUpToMultiple(0, 16) == 0);
    STATIC_REQUIRE(llama::roundUpToMultiple(5, 16) == 16);
    STATIC_REQUIRE(llama::roundUpToMultiple(16, 16) == 16);
    STATIC_REQUIRE(llama::roundUpToMultiple(17, 16) == 32);
    STATIC_REQUIRE(llama::roundUpToMultiple(300, 16) == 304);
}

TEST_CASE("divCeil")
{
    STATIC_REQUIRE(llama::divCeil(0, 16) == 0);
    STATIC_REQUIRE(llama::divCeil(5, 16) == 1);
    STATIC_REQUIRE(llama::divCeil(16, 16) == 1);
    STATIC_REQUIRE(llama::divCeil(17, 16) == 2);
    STATIC_REQUIRE(llama::divCeil(300, 16) == 19);
}

TEST_CASE("isProxyReference")
{
    STATIC_REQUIRE(!llama::internal::IsProxyReferenceImpl<int&>::value);
    STATIC_REQUIRE(!llama::isProxyReference<int&>);
    STATIC_REQUIRE(!llama::internal::IsProxyReferenceImpl<std::string>::value);
    STATIC_REQUIRE(!llama::isProxyReference<std::string>);
    STATIC_REQUIRE(!llama::internal::IsProxyReferenceImpl<std::vector<int>>::value);
    STATIC_REQUIRE(!llama::isProxyReference<std::vector<int>>);
    STATIC_REQUIRE(!llama::internal::IsProxyReferenceImpl<std::vector<bool>::reference>::value);
    STATIC_REQUIRE(!llama::isProxyReference<std::vector<bool>::reference>); // misses a value_type alias

    using One = llama::One<Vec3I>;
    STATIC_REQUIRE(!llama::internal::IsProxyReferenceImpl<One>::value);
    STATIC_REQUIRE(!llama::isProxyReference<One>);
    STATIC_REQUIRE(!llama::internal::IsProxyReferenceImpl<decltype(One{}())>::value);
    STATIC_REQUIRE(!llama::isProxyReference<decltype(One{}())>);

    auto mapping = llama::mapping::BitPackedIntSoA<llama::ArrayExtents<int, 4>, Vec3I>{{}, 17};
    auto v = llama::allocView(mapping);
    [[maybe_unused]] auto ref = v(1)(tag::X{});
    STATIC_REQUIRE(llama::internal::IsProxyReferenceImpl<decltype(ref)>::value);
    STATIC_REQUIRE(llama::isProxyReference<decltype(ref)>);
}

TEST_CASE("isConstant")
{
    STATIC_REQUIRE(!llama::isConstant<int>);
    STATIC_REQUIRE(llama::isConstant<std::true_type>);
    STATIC_REQUIRE(llama::isConstant<std::integral_constant<int, 34>>);
    STATIC_REQUIRE(llama::isConstant<llama::Constant<3u>>);
}

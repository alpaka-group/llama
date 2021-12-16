#include "common.hpp"

TEST_CASE("prettyPrintType")
{
    auto str = prettyPrintType<Particle>();
#ifdef _WIN32
    llama::mapping::tree::internal::replace_all(str, "__int64", "long");
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
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, boost::mp11::mp_list<>>, llama::RecordCoord<>>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::GetCoordFromTags<Particle, boost::mp11::mp_list<tag::Vel, tag::Z>>,
                   llama::RecordCoord<2, 2>>);
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
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0, 0>>, boost::mp11::mp_list<tag::Pos, tag::X >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<0   >>, boost::mp11::mp_list<tag::Pos         >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<    >>, boost::mp11::mp_list<                 >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::RecordCoord<2, 1>>, boost::mp11::mp_list<tag::Vel, tag::Y >>);
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
                   boost::mp11::mp_list<
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
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosRecord, // RD A
            llama::RecordCoord<0>, // Local A
            VelRecord, // RD B
            llama::RecordCoord<0> // Local B
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
    STATIC_REQUIRE(
        std::is_same_v<
            llama::FlatRecordDim<Particle>,
            boost::mp11::mp_list<double, double, double, float, double, double, double, bool, bool, bool, bool>>);
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

TEST_CASE("structName")
{
    CHECK(llama::structName<int>() == "int");
    CHECK(llama::structName(int{}) == "int");
    CHECK(llama::structName<tag::A>() == "A");
    CHECK(llama::structName(tag::A{}) == "A");
    CHECK(llama::structName<tag::Normal>() == "Normal");
    CHECK(llama::structName(tag::Normal{}) == "Normal");
    CHECK(llama::structName(std::string{}) == "basic_string<char,char_traits<char>,allocator<char>>");
    CHECK(llama::structName(Vec3D{}) == "Record<Field<X,double>,Field<Y,double>,Field<Z,double>>");
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
    [[maybe_unused]] llama::internal::BoxedValue<unsigned> v1{42};
    [[maybe_unused]] llama::internal::BoxedValue<unsigned> v2 = 42;
    [[maybe_unused]] WithValue wv{42};
    [[maybe_unused]] WithValueCtor wvc1{1, 2.4, 4};
    [[maybe_unused]] WithValueCtor wvc2{1, 2, 4};
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
    ValueConsumer<unsigned> vc{1};
    CHECK(vc() == 1);
}

TEST_CASE("BoxedValue.Constant")
{
    ValueConsumer<llama::Constant<1>> vc{{}};
    CHECK(vc() == 1);
    STATIC_REQUIRE(vc() == 1);
}

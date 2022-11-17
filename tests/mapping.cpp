#include "common.hpp"

#ifdef __cpp_lib_concepts
TEMPLATE_LIST_TEST_CASE("mapping.concepts", "", SizeTypes)
{
    STATIC_REQUIRE(
        llama::PhysicalMapping<llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<TestType, 2>, Particle>>);
    STATIC_REQUIRE(
        llama::PhysicalMapping<llama::mapping::PackedAoS<llama::ArrayExtentsDynamic<TestType, 2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<
                   llama::mapping::PackedSingleBlobSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<
                   llama::mapping::AlignedSingleBlobSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle>>);
    STATIC_REQUIRE(
        llama::PhysicalMapping<llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle>>);
    STATIC_REQUIRE(
        llama::PhysicalMapping<llama::mapping::AoSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle, 8>>);

    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::AlignedAoS<llama::ArrayExtents<TestType, 2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::PackedAoS<llama::ArrayExtents<TestType, 2>, Particle>>);
    STATIC_REQUIRE(
        llama::PhysicalMapping<llama::mapping::PackedSingleBlobSoA<llama::ArrayExtents<TestType, 2>, Particle>>);
    STATIC_REQUIRE(
        llama::PhysicalMapping<llama::mapping::AlignedSingleBlobSoA<llama::ArrayExtents<TestType, 2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::MultiBlobSoA<llama::ArrayExtents<TestType, 2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::AoSoA<llama::ArrayExtents<TestType, 2>, Particle, 8>>);

    using Inner = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;
    STATIC_REQUIRE(llama::FullyComputedMapping<llama::mapping::Trace<Inner>>);
    STATIC_REQUIRE(llama::FullyComputedMapping<llama::mapping::Trace<Inner, std::size_t, false>>);
#    ifndef _MSC_VER
    STATIC_REQUIRE(llama::FullyComputedMapping<llama::mapping::Heatmap<Inner>>);
#    endif

    STATIC_REQUIRE(
        llama::FullyComputedMapping<llama::mapping::Null<llama::ArrayExtentsDynamic<TestType, 2>, Particle>>);
#    ifndef __NVCOMPILER
    STATIC_REQUIRE(llama::FullyComputedMapping<
                   llama::mapping::
                       Bytesplit<llama::ArrayExtentsDynamic<TestType, 2>, Particle, llama::mapping::BindAoS<>::fn>>);
#    endif
    STATIC_REQUIRE(
        llama::FullyComputedMapping<llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<TestType, 2>, Vec3I>>);
    STATIC_REQUIRE(llama::FullyComputedMapping<
                   llama::mapping::BitPackedFloatSoA<llama::ArrayExtentsDynamic<TestType, 2>, Vec3D>>);
#    ifndef __NVCOMPILER
    STATIC_REQUIRE(llama::PartiallyComputedMapping<llama::mapping::ChangeType<
                       llama::ArrayExtentsDynamic<TestType, 2>,
                       Particle,
                       llama::mapping::BindAoS<>::fn,
                       boost::mp11::mp_list<boost::mp11::mp_list<bool, int>>>>);
#    endif
}
#endif

TEMPLATE_LIST_TEST_CASE("mapping.traits", "", SizeTypes)
{
    using AAoS = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;
    using PAoS = llama::mapping::PackedAoS<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;
    using ASBSoA = llama::mapping::AlignedSingleBlobSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;
    using PSBSoA = llama::mapping::PackedSingleBlobSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;
    using MBSoA = llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;
    using AoAoS = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<TestType, 2>, Particle, 8>;
    using One = llama::mapping::One<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;

    using Null = llama::mapping::Null<llama::ArrayExtentsDynamic<TestType, 2>, Particle>;
    using BS
        = llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<TestType, 2>, Particle, llama::mapping::BindAoS<>::fn>;
    using CT = llama::mapping::ChangeType<
        llama::ArrayExtentsDynamic<TestType, 2>,
        Particle,
        llama::mapping::BindAoS<>::fn,
        boost::mp11::mp_list<>>;
    using BPI = llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<TestType, 2>, Vec3I>;
    using BPF = llama::mapping::BitPackedFloatSoA<llama::ArrayExtentsDynamic<TestType, 2>, Vec3D>;
    using Trace = llama::mapping::Trace<llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<TestType, 2>, Particle>>;

    STATIC_REQUIRE(llama::mapping::isAoS<AAoS>);
    STATIC_REQUIRE(llama::mapping::isAoS<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoS<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoS<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoS<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<One>);
    STATIC_REQUIRE(!llama::mapping::isAoS<Null>);
    STATIC_REQUIRE(!llama::mapping::isAoS<BS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<CT>);
    STATIC_REQUIRE(!llama::mapping::isAoS<BPI>);
    STATIC_REQUIRE(!llama::mapping::isAoS<BPF>);
    STATIC_REQUIRE(!llama::mapping::isAoS<Trace>);

    STATIC_REQUIRE(!llama::mapping::isSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<PAoS>);
    STATIC_REQUIRE(llama::mapping::isSoA<ASBSoA>);
    STATIC_REQUIRE(llama::mapping::isSoA<PSBSoA>);
    STATIC_REQUIRE(llama::mapping::isSoA<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<CT>);
    STATIC_REQUIRE(!llama::mapping::isSoA<BPI>);
    STATIC_REQUIRE(!llama::mapping::isSoA<BPF>);
    STATIC_REQUIRE(!llama::mapping::isSoA<Trace>);

    STATIC_REQUIRE(!llama::mapping::isAoSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<MBSoA>);
    STATIC_REQUIRE(llama::mapping::isAoSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<CT>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<BPI>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<BPF>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<Trace>);

    STATIC_REQUIRE(!llama::mapping::isOne<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isOne<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isOne<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isOne<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isOne<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isOne<AoAoS>);
    STATIC_REQUIRE(llama::mapping::isOne<One>);
    STATIC_REQUIRE(!llama::mapping::isOne<Null>);
    STATIC_REQUIRE(!llama::mapping::isOne<BS>);
    STATIC_REQUIRE(!llama::mapping::isOne<CT>);
    STATIC_REQUIRE(!llama::mapping::isOne<BPI>);
    STATIC_REQUIRE(!llama::mapping::isOne<BPF>);
    STATIC_REQUIRE(!llama::mapping::isOne<Trace>);

    STATIC_REQUIRE(!llama::mapping::isNull<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isNull<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isNull<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isNull<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isNull<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isNull<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isNull<One>);
    STATIC_REQUIRE(llama::mapping::isNull<Null>);
    STATIC_REQUIRE(!llama::mapping::isNull<BS>);
    STATIC_REQUIRE(!llama::mapping::isNull<CT>);
    STATIC_REQUIRE(!llama::mapping::isNull<BPI>);
    STATIC_REQUIRE(!llama::mapping::isNull<BPF>);
    STATIC_REQUIRE(!llama::mapping::isNull<Trace>);

    STATIC_REQUIRE(!llama::mapping::isBytesplit<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<One>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<Null>);
    STATIC_REQUIRE(llama::mapping::isBytesplit<BS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<CT>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<BPI>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<BPF>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<Trace>);

    STATIC_REQUIRE(!llama::mapping::isChangeType<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<One>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<Null>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<BS>);
    STATIC_REQUIRE(llama::mapping::isChangeType<CT>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<BPI>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<BPF>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<Trace>);

    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<CT>);
    STATIC_REQUIRE(llama::mapping::isBitPackedIntSoA<BPI>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<BPF>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<Trace>);

    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<CT>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<BPI>);
    STATIC_REQUIRE(llama::mapping::isBitPackedFloatSoA<BPF>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<Trace>);

    STATIC_REQUIRE(!llama::mapping::isTrace<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isTrace<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isTrace<ASBSoA>);
    STATIC_REQUIRE(!llama::mapping::isTrace<PSBSoA>);
    STATIC_REQUIRE(!llama::mapping::isTrace<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isTrace<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isTrace<One>);
    STATIC_REQUIRE(!llama::mapping::isTrace<Null>);
    STATIC_REQUIRE(!llama::mapping::isTrace<BS>);
    STATIC_REQUIRE(!llama::mapping::isTrace<CT>);
    STATIC_REQUIRE(!llama::mapping::isTrace<BPI>);
    STATIC_REQUIRE(!llama::mapping::isTrace<BPF>);
    STATIC_REQUIRE(llama::mapping::isTrace<Trace>);
}

TEST_CASE("mapping.LinearizeArrayDimsCpp.size")
{
    llama::mapping::LinearizeArrayDimsCpp lin;
    CHECK(lin.size(llama::ArrayExtents{2, 3}) == 2 * 3);
    CHECK(lin.size(llama::ArrayExtents{2, 4}) == 2 * 4);
    CHECK(lin.size(llama::ArrayExtents{2, 5}) == 2 * 5);
    CHECK(lin.size(llama::ArrayExtents{8, 8}) == 8 * 8);
}

TEST_CASE("mapping.LinearizeArrayDimsCpp")
{
    const llama::mapping::LinearizeArrayDimsCpp lin;
    const auto extents = llama::ArrayExtents<int, 4, 4>{};
    CHECK(lin(llama::ArrayIndex{0, 0}, extents) == 0);
    CHECK(lin(llama::ArrayIndex{0, 1}, extents) == 1);
    CHECK(lin(llama::ArrayIndex{0, 2}, extents) == 2);
    CHECK(lin(llama::ArrayIndex{0, 3}, extents) == 3);
    CHECK(lin(llama::ArrayIndex{1, 0}, extents) == 4);
    CHECK(lin(llama::ArrayIndex{1, 1}, extents) == 5);
    CHECK(lin(llama::ArrayIndex{1, 2}, extents) == 6);
    CHECK(lin(llama::ArrayIndex{1, 3}, extents) == 7);
    CHECK(lin(llama::ArrayIndex{2, 0}, extents) == 8);
    CHECK(lin(llama::ArrayIndex{2, 1}, extents) == 9);
    CHECK(lin(llama::ArrayIndex{2, 2}, extents) == 10);
    CHECK(lin(llama::ArrayIndex{2, 3}, extents) == 11);
    CHECK(lin(llama::ArrayIndex{3, 0}, extents) == 12);
    CHECK(lin(llama::ArrayIndex{3, 1}, extents) == 13);
    CHECK(lin(llama::ArrayIndex{3, 2}, extents) == 14);
    CHECK(lin(llama::ArrayIndex{3, 3}, extents) == 15);
}

TEST_CASE("mapping.LinearizeArrayDimsFortran.size")
{
    llama::mapping::LinearizeArrayDimsFortran lin;
    CHECK(lin.size(llama::ArrayExtents{2, 3}) == 2 * 3);
    CHECK(lin.size(llama::ArrayExtents{2, 4}) == 2 * 4);
    CHECK(lin.size(llama::ArrayExtents{2, 5}) == 2 * 5);
    CHECK(lin.size(llama::ArrayExtents{8, 8}) == 8 * 8);
}

TEST_CASE("mapping.LinearizeArrayDimsFortran")
{
    const llama::mapping::LinearizeArrayDimsFortran lin;
    const auto extents = llama::ArrayExtents<int, 4, 4>{};
    CHECK(lin(llama::ArrayIndex{0, 0}, extents) == 0);
    CHECK(lin(llama::ArrayIndex{0, 1}, extents) == 4);
    CHECK(lin(llama::ArrayIndex{0, 2}, extents) == 8);
    CHECK(lin(llama::ArrayIndex{0, 3}, extents) == 12);
    CHECK(lin(llama::ArrayIndex{1, 0}, extents) == 1);
    CHECK(lin(llama::ArrayIndex{1, 1}, extents) == 5);
    CHECK(lin(llama::ArrayIndex{1, 2}, extents) == 9);
    CHECK(lin(llama::ArrayIndex{1, 3}, extents) == 13);
    CHECK(lin(llama::ArrayIndex{2, 0}, extents) == 2);
    CHECK(lin(llama::ArrayIndex{2, 1}, extents) == 6);
    CHECK(lin(llama::ArrayIndex{2, 2}, extents) == 10);
    CHECK(lin(llama::ArrayIndex{2, 3}, extents) == 14);
    CHECK(lin(llama::ArrayIndex{3, 0}, extents) == 3);
    CHECK(lin(llama::ArrayIndex{3, 1}, extents) == 7);
    CHECK(lin(llama::ArrayIndex{3, 2}, extents) == 11);
    CHECK(lin(llama::ArrayIndex{3, 3}, extents) == 15);
}

TEST_CASE("mapping.LinearizeArrayDimsMorton.size")
{
    const llama::mapping::LinearizeArrayDimsMorton lin;
    CHECK(lin.size(llama::ArrayExtents{2, 3}) == 4 * 4);
    CHECK(lin.size(llama::ArrayExtents{2, 4}) == 4 * 4);
    CHECK(lin.size(llama::ArrayExtents{2, 5}) == 8 * 8);
    CHECK(lin.size(llama::ArrayExtents{8, 8}) == 8 * 8);
}

TEST_CASE("mapping.LinearizeArrayDimsMorton")
{
    const llama::mapping::LinearizeArrayDimsMorton lin;
    const auto extents = llama::ArrayExtents<int, 4, 4>{};
    CHECK(lin(llama::ArrayIndex{0, 0}, extents) == 0);
    CHECK(lin(llama::ArrayIndex{0, 1}, extents) == 1);
    CHECK(lin(llama::ArrayIndex{0, 2}, extents) == 4);
    CHECK(lin(llama::ArrayIndex{0, 3}, extents) == 5);
    CHECK(lin(llama::ArrayIndex{1, 0}, extents) == 2);
    CHECK(lin(llama::ArrayIndex{1, 1}, extents) == 3);
    CHECK(lin(llama::ArrayIndex{1, 2}, extents) == 6);
    CHECK(lin(llama::ArrayIndex{1, 3}, extents) == 7);
    CHECK(lin(llama::ArrayIndex{2, 0}, extents) == 8);
    CHECK(lin(llama::ArrayIndex{2, 1}, extents) == 9);
    CHECK(lin(llama::ArrayIndex{2, 2}, extents) == 12);
    CHECK(lin(llama::ArrayIndex{2, 3}, extents) == 13);
    CHECK(lin(llama::ArrayIndex{3, 0}, extents) == 10);
    CHECK(lin(llama::ArrayIndex{3, 1}, extents) == 11);
    CHECK(lin(llama::ArrayIndex{3, 2}, extents) == 14);
    CHECK(lin(llama::ArrayIndex{3, 3}, extents) == 15);
}

TEST_CASE("mapping.FlattenRecordDimInOrder")
{
    using F = llama::mapping::FlattenRecordDimInOrder<Particle>;
    STATIC_REQUIRE(
        std::is_same_v<
            F::FlatRecordDim,
            boost::mp11::mp_list<double, double, double, float, double, double, double, bool, bool, bool, bool>>);
    STATIC_REQUIRE(F::flatIndex<0, 0> == 0);
    STATIC_REQUIRE(F::flatIndex<0, 1> == 1);
    STATIC_REQUIRE(F::flatIndex<0, 2> == 2);
    STATIC_REQUIRE(F::flatIndex<1> == 3);
    STATIC_REQUIRE(F::flatIndex<2, 0> == 4);
    STATIC_REQUIRE(F::flatIndex<2, 1> == 5);
    STATIC_REQUIRE(F::flatIndex<2, 2> == 6);
    STATIC_REQUIRE(F::flatIndex<3, 0> == 7);
    STATIC_REQUIRE(F::flatIndex<3, 1> == 8);
    STATIC_REQUIRE(F::flatIndex<3, 2> == 9);
    STATIC_REQUIRE(F::flatIndex<3, 3> == 10);
}

TEST_CASE("mapping.FlattenRecordDimIncreasingAlignment")
{
    using F = llama::mapping::FlattenRecordDimIncreasingAlignment<Particle>;
    STATIC_REQUIRE(
        std::is_same_v<
            F::FlatRecordDim,
            boost::mp11::mp_list<bool, bool, bool, bool, float, double, double, double, double, double, double>>);
    STATIC_REQUIRE(F::flatIndex<0, 0> == 5);
    STATIC_REQUIRE(F::flatIndex<0, 1> == 6);
    STATIC_REQUIRE(F::flatIndex<0, 2> == 7);
    STATIC_REQUIRE(F::flatIndex<1> == 4);
    STATIC_REQUIRE(F::flatIndex<2, 0> == 8);
    STATIC_REQUIRE(F::flatIndex<2, 1> == 9);
    STATIC_REQUIRE(F::flatIndex<2, 2> == 10);
    STATIC_REQUIRE(F::flatIndex<3, 0> == 0);
    STATIC_REQUIRE(F::flatIndex<3, 1> == 1);
    STATIC_REQUIRE(F::flatIndex<3, 2> == 2);
    STATIC_REQUIRE(F::flatIndex<3, 3> == 3);
}

TEST_CASE("mapping.FlattenRecordDimDecreasingAlignment")
{
    using F = llama::mapping::FlattenRecordDimDecreasingAlignment<Particle>;
    STATIC_REQUIRE(
        std::is_same_v<
            F::FlatRecordDim,
            boost::mp11::mp_list<double, double, double, double, double, double, float, bool, bool, bool, bool>>);
    STATIC_REQUIRE(F::flatIndex<0, 0> == 0);
    STATIC_REQUIRE(F::flatIndex<0, 1> == 1);
    STATIC_REQUIRE(F::flatIndex<0, 2> == 2);
    STATIC_REQUIRE(F::flatIndex<1> == 6);
    STATIC_REQUIRE(F::flatIndex<2, 0> == 3);
    STATIC_REQUIRE(F::flatIndex<2, 1> == 4);
    STATIC_REQUIRE(F::flatIndex<2, 2> == 5);
    STATIC_REQUIRE(F::flatIndex<3, 0> == 7);
    STATIC_REQUIRE(F::flatIndex<3, 1> == 8);
    STATIC_REQUIRE(F::flatIndex<3, 2> == 9);
    STATIC_REQUIRE(F::flatIndex<3, 3> == 10);
}

#include "common.hpp"

#ifdef __cpp_lib_concepts
TEST_CASE("mapping.concepts")
{
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::PackedAoS<llama::ArrayExtentsDynamic<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::SingleBlobSoA<llama::ArrayExtentsDynamic<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::AoSoA<llama::ArrayExtentsDynamic<2>, Particle, 8>>);

    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::AlignedAoS<llama::ArrayExtents<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::PackedAoS<llama::ArrayExtents<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::SingleBlobSoA<llama::ArrayExtents<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::MultiBlobSoA<llama::ArrayExtents<2>, Particle>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::AoSoA<llama::ArrayExtents<2>, Particle, 8>>);

    using Inner = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<2>, Particle>;
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::Trace<Inner>>);
    STATIC_REQUIRE(llama::PhysicalMapping<llama::mapping::Heatmap<Inner>>);

    STATIC_REQUIRE(llama::FullyComputedMapping<llama::mapping::Null<llama::ArrayExtentsDynamic<2>, Particle>>);
    STATIC_REQUIRE(llama::FullyComputedMapping<
                   llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<2>, Particle, llama::mapping::BindAoS<>::fn>>);
    STATIC_REQUIRE(llama::FullyComputedMapping<llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<2>, Vec3I>>);
    STATIC_REQUIRE(
        llama::FullyComputedMapping<llama::mapping::BitPackedFloatSoA<llama::ArrayExtentsDynamic<2>, Vec3D>>);

    STATIC_REQUIRE(llama::PartiallyComputedMapping<llama::mapping::ChangeType<
                       llama::ArrayExtentsDynamic<2>,
                       Particle,
                       llama::mapping::BindAoS<>::fn,
                       boost::mp11::mp_list<boost::mp11::mp_list<bool, int>>>>);
}
#endif

TEST_CASE("mapping.traits")
{
    using AAoS = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<2>, Particle>;
    using PAoS = llama::mapping::PackedAoS<llama::ArrayExtentsDynamic<2>, Particle>;
    using SBSoA = llama::mapping::SingleBlobSoA<llama::ArrayExtentsDynamic<2>, Particle>;
    using MBSoA = llama::mapping::MultiBlobSoA<llama::ArrayExtentsDynamic<2>, Particle>;
    using AoAoS = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<2>, Particle, 8>;
    using One = llama::mapping::One<llama::ArrayExtentsDynamic<2>, Particle>;

    using Null = llama::mapping::Null<llama::ArrayExtentsDynamic<2>, Particle>;
    using BS = llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<2>, Particle, llama::mapping::BindAoS<>::fn>;
    using CT = llama::mapping::
        ChangeType<llama::ArrayExtentsDynamic<2>, Particle, llama::mapping::BindAoS<>::fn, boost::mp11::mp_list<>>;
    using BPI = llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<2>, Vec3I>;
    using BPF = llama::mapping::BitPackedFloatSoA<llama::ArrayExtentsDynamic<2>, Vec3D>;

    STATIC_REQUIRE(llama::mapping::isAoS<AAoS>);
    STATIC_REQUIRE(llama::mapping::isAoS<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoS<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoS<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<One>);
    STATIC_REQUIRE(!llama::mapping::isAoS<Null>);
    STATIC_REQUIRE(!llama::mapping::isAoS<BS>);
    STATIC_REQUIRE(!llama::mapping::isAoS<CT>);
    STATIC_REQUIRE(!llama::mapping::isAoS<BPI>);
    STATIC_REQUIRE(!llama::mapping::isAoS<BPF>);

    STATIC_REQUIRE(!llama::mapping::isSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<PAoS>);
    STATIC_REQUIRE(llama::mapping::isSoA<SBSoA>);
    STATIC_REQUIRE(llama::mapping::isSoA<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isSoA<CT>);
    STATIC_REQUIRE(!llama::mapping::isSoA<BPI>);
    STATIC_REQUIRE(!llama::mapping::isSoA<BPF>);

    STATIC_REQUIRE(!llama::mapping::isAoSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<MBSoA>);
    STATIC_REQUIRE(llama::mapping::isAoSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<CT>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<BPI>);
    STATIC_REQUIRE(!llama::mapping::isAoSoA<BPF>);

    STATIC_REQUIRE(!llama::mapping::isOne<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isOne<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isOne<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isOne<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isOne<AoAoS>);
    STATIC_REQUIRE(llama::mapping::isOne<One>);
    STATIC_REQUIRE(!llama::mapping::isOne<Null>);
    STATIC_REQUIRE(!llama::mapping::isOne<BS>);
    STATIC_REQUIRE(!llama::mapping::isOne<CT>);
    STATIC_REQUIRE(!llama::mapping::isOne<BPI>);
    STATIC_REQUIRE(!llama::mapping::isOne<BPF>);

    STATIC_REQUIRE(!llama::mapping::isNull<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isNull<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isNull<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isNull<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isNull<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isNull<One>);
    STATIC_REQUIRE(llama::mapping::isNull<Null>);
    STATIC_REQUIRE(!llama::mapping::isNull<BS>);
    STATIC_REQUIRE(!llama::mapping::isNull<CT>);
    STATIC_REQUIRE(!llama::mapping::isNull<BPI>);
    STATIC_REQUIRE(!llama::mapping::isNull<BPF>);

    STATIC_REQUIRE(!llama::mapping::isBytesplit<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<One>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<Null>);
    STATIC_REQUIRE(llama::mapping::isBytesplit<BS>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<CT>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<BPI>);
    STATIC_REQUIRE(!llama::mapping::isBytesplit<BPF>);

    STATIC_REQUIRE(!llama::mapping::isChangeType<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<One>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<Null>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<BS>);
    STATIC_REQUIRE(llama::mapping::isChangeType<CT>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<BPI>);
    STATIC_REQUIRE(!llama::mapping::isChangeType<BPF>);

    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<CT>);
    STATIC_REQUIRE(llama::mapping::isBitPackedIntSoA<BPI>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedIntSoA<BPF>);

    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<AAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<PAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<SBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<MBSoA>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<AoAoS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<One>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<Null>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<BS>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<CT>);
    STATIC_REQUIRE(!llama::mapping::isBitPackedFloatSoA<BPI>);
    STATIC_REQUIRE(llama::mapping::isBitPackedFloatSoA<BPF>);
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
    llama::mapping::LinearizeArrayDimsCpp lin;
    const auto extents = llama::ArrayExtents<4, 4>{};
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
    llama::mapping::LinearizeArrayDimsFortran lin;
    const auto extents = llama::ArrayExtents<4, 4>{};
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
    llama::mapping::LinearizeArrayDimsMorton lin;
    CHECK(lin.size(llama::ArrayExtents{2, 3}) == 4 * 4);
    CHECK(lin.size(llama::ArrayExtents{2, 4}) == 4 * 4);
    CHECK(lin.size(llama::ArrayExtents{2, 5}) == 8 * 8);
    CHECK(lin.size(llama::ArrayExtents{8, 8}) == 8 * 8);
}

TEST_CASE("mapping.LinearizeArrayDimsMorton")
{
    llama::mapping::LinearizeArrayDimsMorton lin;
    const auto extents = llama::ArrayExtents<4, 4>{};
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

TEST_CASE("address.AoS.Packed")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::PackedAoS<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 44);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 52);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 53);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 54);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 55);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 84);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 92);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 108);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 109);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 110);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 111);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 904);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 912);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 920);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 924);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 932);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 940);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 948);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 949);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 950);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 951);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.AoS.Packed.fortran")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping
            = llama::mapping::PackedAoS<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsFortran>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 44);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 52);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 53);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 54);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 55);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 904);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 912);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 920);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 924);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 932);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 940);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 948);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 949);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 950);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 951);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 84);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 92);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 108);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 109);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 110);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 111);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.AoS.Packed.morton")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping
            = llama::mapping::PackedAoS<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsMorton>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 44);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 52);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 53);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 54);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 55);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 84);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 92);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 108);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 109);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 110);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 111);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 112);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 120);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 128);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 136);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 140);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 148);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 156);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 164);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 165);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 166);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 167);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.AoS.Aligned")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::AlignedAoS<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 48);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 57);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 58);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 59);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 88);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 96);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 104);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 112);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 120);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 121);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 122);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 123);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 1024);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 1032);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 1040);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 1048);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 1056);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 1064);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 1072);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 1080);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1081);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 1082);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 1083);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.AoS.aligned_min")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::MinAlignedAoS<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 48);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 2);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 3);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 80);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 60);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 88);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 96);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 104);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 57);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 58);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 59);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 904);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 912);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 920);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 900);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 928);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 936);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 944);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 897);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 898);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 899);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.SoA.SingleBlob")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::SingleBlobSoA<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2048);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4096);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6144);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7168);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9216);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11264);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13312);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13568);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13824);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14080);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2056);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4104);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6148);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7176);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9224);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11272);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13313);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13569);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13825);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14081);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 128);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2176);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4224);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6208);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7296);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9344);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11392);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13328);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13584);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13840);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14096);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.SoA.SingleBlob.fortran")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::
            SingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsFortran>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2048);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4096);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6144);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7168);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9216);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11264);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13312);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13568);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13824);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14080);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 128);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2176);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4224);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6208);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7296);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9344);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11392);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13328);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13584);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13840);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14096);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2056);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4104);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6148);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7176);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9224);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11272);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13313);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13569);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13825);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14081);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.SoA.SingleBlob.morton")
{
    struct Value
    {
    };

    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::
            SingleBlobSoA<decltype(arrayExtents), Particle, llama::mapping::LinearizeArrayDimsMorton>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2048);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4096);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6144);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7168);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9216);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11264);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13312);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13568);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13824);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14080);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2056);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4104);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6148);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7176);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9224);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11272);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13313);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13569);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13825);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14081);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 2064);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 4112);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 6152);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 7184);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 9232);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 11280);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 13314);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 13570);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 13826);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 14082);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.SoA.MultiBlob")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::MultiBlobSoA<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset{0, 0});
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset{1, 0});
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset{2, 0});
            CHECK(mapping.template blobNrAndOffset<1>(ai) == llama::NrAndOffset{3, 0});
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset{4, 0});
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset{5, 0});
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset{6, 0});
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset{7, 0});
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset{8, 0});
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset{9, 0});
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset{10, 0});
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset{0, 8});
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset{1, 8});
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset{2, 8});
            CHECK(mapping.template blobNrAndOffset<1>(ai) == llama::NrAndOffset{3, 4});
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset{4, 8});
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset{5, 8});
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset{6, 8});
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset{7, 1});
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset{8, 1});
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset{9, 1});
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset{10, 1});
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset{0, 128});
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset{1, 128});
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset{2, 128});
            CHECK(mapping.template blobNrAndOffset<1>(ai) == llama::NrAndOffset{3, 64});
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset{4, 128});
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset{5, 128});
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset{6, 128});
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset{7, 16});
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset{8, 16});
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset{9, 16});
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset{10, 16});
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.AoSoA.4")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::AoSoA<decltype(arrayExtents), Particle, 4>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        {
            const auto ai = ArrayIndex{0, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 64);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 96);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 112);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 144);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 176);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 208);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 212);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 216);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 220);
        }

        {
            const auto ai = ArrayIndex{0, 1};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 72);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 100);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 120);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 152);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 184);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 209);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 213);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 217);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 221);
        }

        {
            const auto ai = ArrayIndex{1, 0};
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 896);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 928);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 960);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 992);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 1008);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 1040);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 1072);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 1104);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1108);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 1112);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 1116);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.PackedOne")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::PackedOne<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobSize(0) == 56);
        for(const auto ai : {ArrayIndex{0, 0}, ArrayIndex{0, 1}, ArrayIndex{1, 0}})
        {
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 28);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 36);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 44);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 52);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 53);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 54);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 55);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.AlignedOne")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::AlignedOne<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobSize(0) == 60);
        for(const auto ai : {ArrayIndex{0, 0}, ArrayIndex{0, 1}, ArrayIndex{1, 0}})
        {
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 48);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 56);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 57);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 58);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 59);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}

TEST_CASE("address.MinAlignedOne")
{
    auto test = [](auto arrayExtents)
    {
        using Mapping = llama::mapping::MinAlignedOne<decltype(arrayExtents), Particle>;
        auto mapping = Mapping{arrayExtents};
        using ArrayIndex = typename Mapping::ArrayIndex;

        STATIC_REQUIRE(mapping.blobSize(0) == 56);
        for(const auto ai : {ArrayIndex{0, 0}, ArrayIndex{0, 1}, ArrayIndex{1, 0}})
        {
            CHECK(mapping.template blobNrAndOffset<0, 0>(ai).offset == 8);
            CHECK(mapping.template blobNrAndOffset<0, 1>(ai).offset == 16);
            CHECK(mapping.template blobNrAndOffset<0, 2>(ai).offset == 24);
            CHECK(mapping.template blobNrAndOffset<1>(ai).offset == 4);
            CHECK(mapping.template blobNrAndOffset<2, 0>(ai).offset == 32);
            CHECK(mapping.template blobNrAndOffset<2, 1>(ai).offset == 40);
            CHECK(mapping.template blobNrAndOffset<2, 2>(ai).offset == 48);
            CHECK(mapping.template blobNrAndOffset<3, 0>(ai).offset == 0);
            CHECK(mapping.template blobNrAndOffset<3, 1>(ai).offset == 1);
            CHECK(mapping.template blobNrAndOffset<3, 2>(ai).offset == 2);
            CHECK(mapping.template blobNrAndOffset<3, 3>(ai).offset == 3);
        }
    };
    test(llama::ArrayExtentsDynamic<2>{16, 16});
    test(llama::ArrayExtents<16, llama::dyn>{16});
    test(llama::ArrayExtents<llama::dyn, 16>{16});
    test(llama::ArrayExtents<16, 16>{});
}


TEST_CASE("maxLanes")
{
    STATIC_REQUIRE(llama::mapping::maxLanes<Particle, 128> == 2);
    STATIC_REQUIRE(llama::mapping::maxLanes<Particle, 256> == 4);
    STATIC_REQUIRE(llama::mapping::maxLanes<Particle, 512> == 8);

    STATIC_REQUIRE(llama::mapping::maxLanes<float, 128> == 4);
    STATIC_REQUIRE(llama::mapping::maxLanes<float, 256> == 8);
    STATIC_REQUIRE(llama::mapping::maxLanes<float, 512> == 16);

    using RecordDim1 = llama::Record<llama::Field<tag::X, std::int8_t>, llama::Field<tag::Y, std::uint8_t>>;
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim1, 128> == 16);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim1, 256> == 32);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim1, 512> == 64);

    using RecordDim2 = llama::Record<llama::Field<tag::X, std::int8_t>, llama::Field<tag::Y, std::int16_t>>;
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim2, 128> == 8);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim2, 256> == 16);
    STATIC_REQUIRE(llama::mapping::maxLanes<RecordDim2, 512> == 32);
}

TEST_CASE("AoSoA.size_round_up")
{
    using AoSoA = llama::mapping::AoSoA<llama::ArrayExtentsDynamic<1>, Particle, 4>;
    constexpr auto psize = llama::sizeOf<Particle>;

    CHECK(AoSoA{{0}}.blobSize(0) == 0 * psize);
    CHECK(AoSoA{{1}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{2}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{3}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{4}}.blobSize(0) == 4 * psize);
    CHECK(AoSoA{{5}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{6}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{7}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{8}}.blobSize(0) == 8 * psize);
    CHECK(AoSoA{{9}}.blobSize(0) == 12 * psize);
}

TEST_CASE("AoSoA.address_within_bounds")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<1>;
    using AoSoA = llama::mapping::AoSoA<ArrayExtents, Particle, 4>;

    const auto ad = ArrayExtents{3};
    auto mapping = AoSoA{ad};
    for(auto i : llama::ArrayIndexRange{ad})
        llama::forEachLeafCoord<Particle>([&](auto rc)
                                          { CHECK(mapping.blobNrAndOffset(i, rc).offset < mapping.blobSize(0)); });
}

TEST_CASE("FlattenRecordDimInOrder")
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

TEST_CASE("FlattenRecordDimIncreasingAlignment")
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

TEST_CASE("FlattenRecordDimDecreasingAlignment")
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

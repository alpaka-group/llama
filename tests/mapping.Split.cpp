#include "common.hpp"

#include <fstream>

TEST_CASE("mapping.Split.partitionRecordDim.OneMemberRecord")
{
    using RecordDim = llama::Record<llama::Field<int, int>>;
    using R = decltype(llama::mapping::internal::partitionRecordDim(RecordDim{}, llama::RecordCoord<0>{}));
    STATIC_REQUIRE(std::is_same_v<boost::mp11::mp_first<R>, RecordDim>);
    STATIC_REQUIRE(std::is_same_v<boost::mp11::mp_second<R>, llama::Record<>>);
}

TEST_CASE("mapping.Split.partitionRecordDim.Vec3I.RC")
{
    using R = decltype(llama::mapping::internal::partitionRecordDim(Vec3I{}, llama::RecordCoord<1>{}));
    STATIC_REQUIRE(std::is_same_v<boost::mp11::mp_first<R>, llama::Record<llama::Field<tag::Y, int>>>);
    STATIC_REQUIRE(
        std::
            is_same_v<boost::mp11::mp_second<R>, llama::Record<llama::Field<tag::X, int>, llama::Field<tag::Z, int>>>);
}

template<typename S>
using TestReplace = typename llama::mapping::internal::ReplaceTagListsByCoords<Particle, S>::type;

TEST_CASE("mapping.Split.ReplaceTagListsByCoords")
{
    // single selector
    STATIC_REQUIRE(std::is_same_v<TestReplace<llama::RecordCoord<2, 2>>, llama::RecordCoord<2, 2>>);
    STATIC_REQUIRE(std::is_same_v<TestReplace<boost::mp11::mp_list<tag::Vel, tag::Z>>, llama::RecordCoord<2, 2>>);

    // list of selectors with single item
    STATIC_REQUIRE(std::is_same_v<
                   TestReplace<boost::mp11::mp_list<llama::RecordCoord<2, 2>>>,
                   boost::mp11::mp_list<llama::RecordCoord<2, 2>>>);
    STATIC_REQUIRE(std::is_same_v<
                   TestReplace<boost::mp11::mp_list<boost::mp11::mp_list<tag::Vel, tag::Z>>>,
                   boost::mp11::mp_list<llama::RecordCoord<2, 2>>>);

    // list of selectors with multiple items
    STATIC_REQUIRE(std::is_same_v<
                   TestReplace<boost::mp11::mp_list<llama::RecordCoord<1>, llama::RecordCoord<2, 2>>>,
                   boost::mp11::mp_list<llama::RecordCoord<1>, llama::RecordCoord<2, 2>>>);
    STATIC_REQUIRE(
        std::is_same_v<
            TestReplace<boost::mp11::mp_list<boost::mp11::mp_list<tag::Mass>, boost::mp11::mp_list<tag::Vel, tag::Z>>>,
            boost::mp11::mp_list<llama::RecordCoord<1>, llama::RecordCoord<2, 2>>>);
    STATIC_REQUIRE(std::is_same_v<
                   TestReplace<boost::mp11::mp_list<llama::RecordCoord<1>, boost::mp11::mp_list<tag::Vel, tag::Z>>>,
                   boost::mp11::mp_list<llama::RecordCoord<1>, llama::RecordCoord<2, 2>>>);
}

TEST_CASE("mapping.Split.partitionRecordDim.Particle")
{
    using R = decltype(llama::mapping::internal::partitionRecordDim(Particle{}, llama::RecordCoord<2, 1>{}));
    STATIC_REQUIRE(std::is_same_v<
                   boost::mp11::mp_first<R>,
                   llama::Record<llama::Field<tag::Vel, llama::Record<llama::Field<tag::Y, double>>>>>);
    STATIC_REQUIRE(
        std::is_same_v<
            boost::mp11::mp_second<R>,
            llama::Record<
                llama::Field<tag::Pos, Vec3D>,
                llama::Field<tag::Mass, float>,
                llama::Field<tag::Vel, llama::Record<llama::Field<tag::X, double>, llama::Field<tag::Z, double>>>,
                llama::Field<tag::Flags, bool[4]>>>);
}

TEST_CASE("mapping.Split.partitionRecordDim.Particle.List")
{
    using R = decltype(llama::mapping::internal::partitionRecordDim(
        Particle{},
        boost::mp11::mp_list<llama::RecordCoord<0>, llama::RecordCoord<2>>{}));
    STATIC_REQUIRE(std::is_same_v<
                   boost::mp11::mp_first<R>,
                   llama::Record<llama::Field<tag::Pos, Vec3D>, llama::Field<tag::Vel, Vec3D>>>);
    STATIC_REQUIRE(std::is_same_v<
                   boost::mp11::mp_second<R>,
                   llama::Record<llama::Field<tag::Mass, float>, llama::Field<tag::Flags, bool[4]>>>);
}

TEST_CASE("mapping.Split.SoA_SingleBlob.AoS_Packed.1Buffer")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 2>;
    auto extents = ArrayExtents{16, 16};

    // we layout Pos as SoA, the rest as AoS
    auto mapping = llama::mapping::Split<
        ArrayExtents,
        Particle,
        llama::RecordCoord<0>,
        llama::mapping::PackedSingleBlobSoA,
        llama::mapping::PackedAoS>{extents};

    STATIC_REQUIRE(mapping.blobCount == 1);
    CHECK(mapping.blobSize(0) == 14336);

    constexpr auto mapping1Size = 6120;
    const auto ai = llama::ArrayIndex{0, 0};
    CHECK(mapping.blobNrAndOffset<0, 0>(ai) == llama::NrAndOffset{0, 0});
    CHECK(mapping.blobNrAndOffset<0, 1>(ai) == llama::NrAndOffset{0, 2048});
    CHECK(mapping.blobNrAndOffset<0, 2>(ai) == llama::NrAndOffset{0, 4096});
    CHECK(mapping.blobNrAndOffset<1>(ai) == llama::NrAndOffset{0, mapping1Size + 24});
    CHECK(mapping.blobNrAndOffset<2, 0>(ai) == llama::NrAndOffset{0, mapping1Size + 28});
    CHECK(mapping.blobNrAndOffset<2, 1>(ai) == llama::NrAndOffset{0, mapping1Size + 36});
    CHECK(mapping.blobNrAndOffset<2, 2>(ai) == llama::NrAndOffset{0, mapping1Size + 44});
    CHECK(mapping.blobNrAndOffset<3, 0>(ai) == llama::NrAndOffset{0, mapping1Size + 52});
    CHECK(mapping.blobNrAndOffset<3, 1>(ai) == llama::NrAndOffset{0, mapping1Size + 53});
    CHECK(mapping.blobNrAndOffset<3, 2>(ai) == llama::NrAndOffset{0, mapping1Size + 54});
    CHECK(mapping.blobNrAndOffset<3, 3>(ai) == llama::NrAndOffset{0, mapping1Size + 55});
}

TEST_CASE("mapping.Split.AoSoA8.AoS_Packed.One.SoA_SingleBlob.4Buffer")
{
    // split out velocity as AoSoA8, mass into a single value, position into AoS, and the flags into SoA, makes 4
    // buffers
    using ArrayExtents = llama::ArrayExtents<int, llama::dyn>;
    auto extents = ArrayExtents{32};
    auto mapping = llama::mapping::Split<
        ArrayExtents,
        Particle,
        boost::mp11::mp_list<tag::Vel>,
        llama::mapping::BindAoSoA<8>::fn,
        llama::mapping::BindSplit<
            boost::mp11::mp_list<tag::Mass>,
            llama::mapping::PackedOne,
            llama::mapping::BindSplit<
                boost::mp11::mp_list<tag::Pos>,
                llama::mapping::PackedAoS,
                llama::mapping::PackedSingleBlobSoA,
                true>::fn,
            true>::fn,
        true>{extents};

    STATIC_REQUIRE(mapping.blobCount == 4);
    CHECK(mapping.blobSize(0) == 768);
    CHECK(mapping.blobSize(1) == 4);
    CHECK(mapping.blobSize(2) == 768);
    CHECK(mapping.blobSize(3) == 128);

    CHECK(mapping.blobNrAndOffset<0, 0>({0}) == llama::NrAndOffset{2, 0});
    CHECK(mapping.blobNrAndOffset<0, 1>({0}) == llama::NrAndOffset{2, 8});
    CHECK(mapping.blobNrAndOffset<0, 2>({0}) == llama::NrAndOffset{2, 16});
    CHECK(mapping.blobNrAndOffset<1>({0}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({0}) == llama::NrAndOffset{0, 0});
    CHECK(mapping.blobNrAndOffset<2, 1>({0}) == llama::NrAndOffset{0, 64});
    CHECK(mapping.blobNrAndOffset<2, 2>({0}) == llama::NrAndOffset{0, 128});
    CHECK(mapping.blobNrAndOffset<3, 0>({0}) == llama::NrAndOffset{3, 0});
    CHECK(mapping.blobNrAndOffset<3, 1>({0}) == llama::NrAndOffset{3, 32});
    CHECK(mapping.blobNrAndOffset<3, 2>({0}) == llama::NrAndOffset{3, 64});
    CHECK(mapping.blobNrAndOffset<3, 3>({0}) == llama::NrAndOffset{3, 96});

    CHECK(mapping.blobNrAndOffset<0, 0>({31}) == llama::NrAndOffset{2, 744});
    CHECK(mapping.blobNrAndOffset<0, 1>({31}) == llama::NrAndOffset{2, 752});
    CHECK(mapping.blobNrAndOffset<0, 2>({31}) == llama::NrAndOffset{2, 760});
    CHECK(mapping.blobNrAndOffset<1>({31}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({31}) == llama::NrAndOffset{0, 632});
    CHECK(mapping.blobNrAndOffset<2, 1>({31}) == llama::NrAndOffset{0, 696});
    CHECK(mapping.blobNrAndOffset<2, 2>({31}) == llama::NrAndOffset{0, 760});
    CHECK(mapping.blobNrAndOffset<3, 0>({31}) == llama::NrAndOffset{3, 31});
    CHECK(mapping.blobNrAndOffset<3, 1>({31}) == llama::NrAndOffset{3, 63});
    CHECK(mapping.blobNrAndOffset<3, 2>({31}) == llama::NrAndOffset{3, 95});
    CHECK(mapping.blobNrAndOffset<3, 3>({31}) == llama::NrAndOffset{3, 127});

    // std::ofstream{"Split.AoSoA8.AoS.One.SoA.4Buffer.svg"} << llama::toSvg(mapping);
}


TEST_CASE("mapping.Split.Multilist.SoA.One")
{
    // split out Pos and Vel into SoA, the rest into One
    using ArrayExtents = llama::ArrayExtents<int, llama::dyn>;
    auto extents = ArrayExtents{32};
    auto mapping = llama::mapping::Split<
        ArrayExtents,
        Particle,
        boost::mp11::mp_list<llama::RecordCoord<0>, llama::RecordCoord<2>>,
        llama::mapping::BindSoA<>::fn,
        llama::mapping::AlignedOne,
        true>{extents};

    STATIC_REQUIRE(mapping.blobCount == 7);
    CHECK(mapping.blobSize(0) == 256);

    CHECK(mapping.blobNrAndOffset<0, 0>({0}) == llama::NrAndOffset{0, 0});
    CHECK(mapping.blobNrAndOffset<0, 1>({0}) == llama::NrAndOffset{1, 0});
    CHECK(mapping.blobNrAndOffset<0, 2>({0}) == llama::NrAndOffset{2, 0});
    CHECK(mapping.blobNrAndOffset<1>({0}) == llama::NrAndOffset{6, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({0}) == llama::NrAndOffset{3, 0});
    CHECK(mapping.blobNrAndOffset<2, 1>({0}) == llama::NrAndOffset{4, 0});
    CHECK(mapping.blobNrAndOffset<2, 2>({0}) == llama::NrAndOffset{5, 0});
    CHECK(mapping.blobNrAndOffset<3, 0>({0}) == llama::NrAndOffset{6, 4});
    CHECK(mapping.blobNrAndOffset<3, 1>({0}) == llama::NrAndOffset{6, 5});
    CHECK(mapping.blobNrAndOffset<3, 2>({0}) == llama::NrAndOffset{6, 6});
    CHECK(mapping.blobNrAndOffset<3, 3>({0}) == llama::NrAndOffset{6, 7});

    CHECK(mapping.blobNrAndOffset<0, 0>({31}) == llama::NrAndOffset{0, 31 * 8});
    CHECK(mapping.blobNrAndOffset<0, 1>({31}) == llama::NrAndOffset{1, 31 * 8});
    CHECK(mapping.blobNrAndOffset<0, 2>({31}) == llama::NrAndOffset{2, 31 * 8});
    CHECK(mapping.blobNrAndOffset<1>({31}) == llama::NrAndOffset{6, 0});
    CHECK(mapping.blobNrAndOffset<2, 0>({31}) == llama::NrAndOffset{3, 31 * 8});
    CHECK(mapping.blobNrAndOffset<2, 1>({31}) == llama::NrAndOffset{4, 31 * 8});
    CHECK(mapping.blobNrAndOffset<2, 2>({31}) == llama::NrAndOffset{5, 31 * 8});
    CHECK(mapping.blobNrAndOffset<3, 0>({31}) == llama::NrAndOffset{6, 4});
    CHECK(mapping.blobNrAndOffset<3, 1>({31}) == llama::NrAndOffset{6, 5});
    CHECK(mapping.blobNrAndOffset<3, 2>({31}) == llama::NrAndOffset{6, 6});
    CHECK(mapping.blobNrAndOffset<3, 3>({31}) == llama::NrAndOffset{6, 7});

    // std::ofstream{"Split.AoSoA8.AoS.One.SoA.4Buffer.svg"} << llama::toSvg(mapping);
}

TEST_CASE("mapping.Split.BitPacked")
{
    // split X and Y into bitpacked SoA, keep Z as AoS
    using ArrayExtents = llama::ArrayExtents<int, llama::dyn>;
    auto extents = ArrayExtents{32};
    auto mapping = llama::mapping::Split<
        ArrayExtents,
        Vec3I,
        llama::RecordCoord<0>,
        llama::mapping::BindBitPackedIntSoA<llama::Constant<3>>::fn,
        llama::mapping::BindSplit<
            llama::RecordCoord<0>,
            llama::mapping::BindBitPackedIntSoA<>::fn,
            llama::mapping::PackedAoS,
            true>::fn,
        true>{std::tuple{extents}, std::tuple{std::tuple{extents, 5}, std::tuple{extents}}};

    STATIC_REQUIRE(mapping.blobCount == 3);
    CHECK(mapping.blobSize(0) == 12);
    CHECK(mapping.blobSize(1) == 20);
    CHECK(mapping.blobSize(2) == 128);

    std::array<std::array<std::byte, 128>, 3> blobs{};

    STATIC_REQUIRE(mapping.isComputed(llama::RecordCoord<0>{}));
    mapping.compute({0}, llama::RecordCoord<0>{}, blobs) = 1;
    mapping.compute({1}, llama::RecordCoord<0>{}, blobs) = 2;
    mapping.compute({31}, llama::RecordCoord<0>{}, blobs) = 31;
    CHECK(mapping.compute({0}, llama::RecordCoord<0>{}, blobs) == 1);
    CHECK(mapping.compute({1}, llama::RecordCoord<0>{}, blobs) == 2);
    CHECK(mapping.compute({31}, llama::RecordCoord<0>{}, blobs) == 3); // bits cut off

    STATIC_REQUIRE(mapping.isComputed(llama::RecordCoord<1>{}));
    mapping.compute({0}, llama::RecordCoord<1>{}, blobs) = 1;
    mapping.compute({1}, llama::RecordCoord<1>{}, blobs) = 2;
    mapping.compute({31}, llama::RecordCoord<1>{}, blobs) = 31;
    CHECK(mapping.compute({0}, llama::RecordCoord<1>{}, blobs) == 1);
    CHECK(mapping.compute({1}, llama::RecordCoord<1>{}, blobs) == 2);
    CHECK(mapping.compute({31}, llama::RecordCoord<1>{}, blobs) == 15); // bits cut off

    CHECK(mapping.blobNrAndOffset<2>({0}) == llama::NrAndOffset{2, 0});
    CHECK(mapping.blobNrAndOffset<2>({1}) == llama::NrAndOffset{2, 4});
    CHECK(mapping.blobNrAndOffset<2>({31}) == llama::NrAndOffset{2, 124});
}
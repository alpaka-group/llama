#include "common.hpp"

//#include <bit>

TEST_CASE("mapping.ByteSplit.AoS")
{
    auto view = llama::allocView(
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<1>, Vec3I, llama::mapping::PreconfiguredAoS<>::type>{
            {128}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.SoA")
{
    auto view = llama::allocView(
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<1>, Vec3I, llama::mapping::PreconfiguredSoA<>::type>{
            {128}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.AoSoA")
{
    auto view = llama::allocView(
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<1>, Vec3I, llama::mapping::PreconfiguredAoSoA<16>::type>{
            {128}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.ChangeType.SoA")
{
    using Mapping = llama::mapping::Bytesplit<
        llama::ArrayExtentsDynamic<1>,
        Vec3I,
        llama::mapping::PreconfiguredChangeType<
            llama::mapping::PreconfiguredSoA<>::type,
            boost::mp11::mp_list<boost::mp11::mp_list<std::byte, unsigned char>>>::type>;

    STATIC_REQUIRE(std::is_same_v<Mapping::RecordDim, Vec3I>);
    STATIC_REQUIRE(std::is_same_v<
                   typename Mapping::Inner::RecordDim,
                   llama::Record<
                       llama::Field<tag::X, std::byte[4]>,
                       llama::Field<tag::Y, std::byte[4]>,
                       llama::Field<tag::Z, std::byte[4]>>>);
    STATIC_REQUIRE(std::is_same_v<
                   typename Mapping::Inner::Inner::RecordDim,
                   llama::Record<
                       llama::Field<tag::X, unsigned char[4]>,
                       llama::Field<tag::Y, unsigned char[4]>,
                       llama::Field<tag::Z, unsigned char[4]>>>);

    auto view = llama::allocView(Mapping{{128}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.Split.BitPackedIntSoA")
{
    auto view = llama::allocView(llama::mapping::Bytesplit<
                                 llama::ArrayExtentsDynamic<1>,
                                 Vec3I,
                                 llama::mapping::PreconfiguredSplit<
                                     llama::RecordCoord<1>,
                                     llama::mapping::BitPackedIntSoA,
                                     llama::mapping::PreconfiguredAoS<>::type,
                                     true>::type>{
        std::tuple{std::tuple{23, llama::ArrayExtents{128}}, std::tuple{llama::ArrayExtents{128}}}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.SoA.verify")
{
    using Mapping
        = llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<1>, Vec3I, llama::mapping::PreconfiguredSoA<>::type>;
    auto view = llama::allocView(Mapping{{128}});
    for(auto i = 0; i < 128; i++)
        view(i) = i;

    CHECK(Mapping::blobCount == 12);

    for(auto b = 0; b < 12; b++)
    {
        // TODO(bgruber): assumes little endian for now. Upgrade to C++20 and enable.
        // const auto isNonZero = std::endian::native == std::endian::little ? (b == 0 || b == 4 || b == 8)
        //                                                                  : (b == 3 || b == 7 || b == 11);
        const auto isNonZero = (b == 0 || b == 4 || b == 8);
        for(auto i = 0; i < 128; i++)
            CHECK(view.storageBlobs[b][i] == (isNonZero ? static_cast<std::byte>(i) : std::byte{0}));
    }
}

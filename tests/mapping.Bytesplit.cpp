#include "common.hpp"

// #include <bit>

TEST_CASE("mapping.ByteSplit.AoS")
{
    auto view = llama::allocView(
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<std::size_t, 1>, Vec3I, llama::mapping::BindAoS<>::fn>{
            {128}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.SoA")
{
    auto view = llama::allocView(
        llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<std::size_t, 1>, Vec3I, llama::mapping::BindSoA<>::fn>{
            {128}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.AoSoA")
{
    auto view = llama::allocView(
        llama::mapping::
            Bytesplit<llama::ArrayExtentsDynamic<std::size_t, 1>, Vec3I, llama::mapping::BindAoSoA<16>::fn>{{128}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.ChangeType.SoA")
{
    using Mapping = llama::mapping::Bytesplit<
        llama::ArrayExtentsDynamic<std::size_t, 1>,
        Vec3I,
        llama::mapping::BindChangeType<
            llama::mapping::BindSoA<>::fn,
            boost::mp11::mp_list<boost::mp11::mp_list<std::byte, unsigned char>>>::fn>;

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
    auto view = llama::allocView(
        llama::mapping::Bytesplit<
            llama::ArrayExtentsDynamic<int, 1>,
            Vec3I,
            llama::mapping::BindSplit<
                llama::RecordCoord<1>,
                llama::mapping::BindBitPackedIntSoA<>::fn,
                llama::mapping::BindAoS<>::fn,
                true>::fn>{std::tuple{std::tuple{llama::ArrayExtents{128}, 8}, std::tuple{llama::ArrayExtents{128}}}});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ByteSplit.SoA.verify")
{
    using Mapping
        = llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<std::size_t, 1>, Vec3I, llama::mapping::BindSoA<>::fn>;
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


TEST_CASE("mapping.ByteSplit.ProxyRef.SwapAndAssign")
{
    using Mapping
        = llama::mapping::Bytesplit<llama::ArrayExtentsDynamic<std::size_t, 1>, Vec3I, llama::mapping::BindSoA<>::fn>;
    auto view = llama::allocView(Mapping{{128}});

    view(0) = 1.0;
    view(1) = 2.0;
    swap(view(0), view(1));
    CHECK(view(0) == 2.0);
    CHECK(view(1) == 1.0);

    view(0) = view(1);
    CHECK(view(0) == 1.0);
    CHECK(view(1) == 1.0);
}
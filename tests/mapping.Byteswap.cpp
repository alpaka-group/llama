#include "common.hpp"

TEST_CASE("mapping.Byteswap.AoS")
{
    auto view = llama::allocView(
        llama::mapping::Byteswap<llama::ArrayExtents<std::size_t, 128>, Particle, llama::mapping::BindAoS<>::fn>{});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.Byteswap.SoA")
{
    auto view = llama::allocView(
        llama::mapping::Byteswap<llama::ArrayExtents<std::size_t, 128>, Particle, llama::mapping::BindSoA<>::fn>{});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.Byteswap.AoSoA")
{
    auto view = llama::allocView(
        llama::mapping::
            Byteswap<llama::ArrayExtents<std::size_t, 128>, Particle, llama::mapping::BindAoSoA<16>::fn>{});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.Byteswap.CheckBytes")
{
    using RecordDim
        = llama::Record<llama::Field<tag::X, uint8_t>, llama::Field<tag::Y, uint16_t>, llama::Field<tag::Z, uint32_t>>;

    auto view = llama::allocView(
        llama::mapping::Byteswap<llama::ArrayExtents<std::size_t, 128>, RecordDim, llama::mapping::BindAoS<>::fn>{});

    view(0)(tag::X{}) = 0x12;
    view(0)(tag::Y{}) = 0x1234;
    view(0)(tag::Z{}) = 0x12345678;

    std::byte* p = &view.storageBlobs[0][0]; // NOLINT(readability-container-data-pointer)
    CHECK(*reinterpret_cast<uint8_t*>(p) == 0x12);
    CHECK(*reinterpret_cast<uint16_t*>(p + 2) == 0x3412);
    CHECK(*reinterpret_cast<uint32_t*>(p + 4) == 0x78563412);
}

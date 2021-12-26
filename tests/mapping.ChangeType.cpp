#include "common.hpp"

#include <cstdint>

TEST_CASE("mapping.ChangeType.AoS")
{
    auto mapping = llama::mapping::
        ChangeType<llama::ArrayExtents<128>, Vec3D, llama::mapping::BindAoS<false>::fn, boost::mp11::mp_list<>>{{}};
    CHECK(mapping.blobSize(0) == 3072);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.AoS.doubleToFloat")
{
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<128>,
        Vec3D,
        llama::mapping::BindAoS<false>::fn,
        boost::mp11::mp_list<boost::mp11::mp_list<double, float>>>{{}};
    CHECK(mapping.blobSize(0) == 1536);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.SoA.particle")
{
    auto mappingAoS = llama::mapping::PackedAoS<llama::ArrayExtents<128>, Particle>{{}};
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<128>,
        Particle,
        llama::mapping::BindAoS<false>::fn,
        boost::mp11::mp_list<
            boost::mp11::mp_list<double, float>,
            boost::mp11::mp_list<float, std::int16_t>,
            boost::mp11::mp_list<bool, std::int32_t>>>{{}};
    CHECK(mappingAoS.blobSize(0) == 128 * (6 * sizeof(double) + 1 * sizeof(float) + 4 * sizeof(bool)));
    CHECK(mapping.blobSize(0) == 128 * (6 * sizeof(float) + 1 * sizeof(std::int16_t) + 4 * sizeof(std::int32_t)));
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

#include <cstdint>

TEST_CASE("mapping.ChangeType.AoS")
{
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<>>{{}};
    CHECK(mapping.blobSize(0) == 3072);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.AoS.doubleToFloat")
{
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<mp_list<double, float>>>{{}};
    CHECK(mapping.blobSize(0) == 1536);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.AoS.Coord1ToFloat")
{
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<mp_list<llama::RecordCoord<1>, float>>>{{}};
    CHECK(mapping.blobSize(0) == 2560);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.AoS.CoordsAndTypes")
{
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,

        mp_list<mp_list<double, float>, mp_list<llama::RecordCoord<1>, double>>>{{}};
    CHECK(mapping.blobSize(0) == 2048);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.SoA.particle.types")
{
    auto mappingAoS = llama::mapping::PackedAoS<llama::ArrayExtents<std::size_t, 128>, Particle>{{}};
    CHECK(mappingAoS.blobSize(0) == 128 * (6 * sizeof(double) + 1 * sizeof(float) + 4 * sizeof(bool)));

    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<std::size_t, 128>,
        Particle,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<mp_list<double, float>, mp_list<float, std::int16_t>, mp_list<bool, std::int32_t>>>{{}};
    CHECK(mapping.blobSize(0) == 128 * (6 * sizeof(float) + 1 * sizeof(std::int16_t) + 4 * sizeof(std::int32_t)));
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.SoA.particle.coords")
{
    using namespace boost::mp11;
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<std::size_t, 128>,
        Particle,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<
            mp_list<llama::RecordCoord<0, 1>, float>,
            mp_list<llama::RecordCoord<1>, std::int16_t>,
            mp_list<llama::RecordCoord<2, 0>, float>,
            mp_list<llama::RecordCoord<3, 0>, std::int32_t>,
            mp_list<llama::RecordCoord<3, 1>, std::int32_t>,
            mp_list<llama::RecordCoord<3, 2>, std::int32_t>,
            mp_list<llama::RecordCoord<3, 3>, std::int32_t>>>{{}};
    CHECK(
        mapping.blobSize(0)
        == 128 * (4 * sizeof(double) + 2 * sizeof(float) + 1 * sizeof(std::int16_t) + 4 * sizeof(std::int32_t)));
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.ChangeType.ProxyRef.SwapAndAssign")
{
    auto mapping = llama::mapping::ChangeType<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<>>{{}};
    auto view = llama::allocView(mapping);

    view(0) = 1.0;
    view(1) = 2.0;
    swap(view(0), view(1));
    CHECK(view(0) == 2.0);
    CHECK(view(1) == 1.0);

    view(0) = view(1);
    CHECK(view(0) == 1.0);
    CHECK(view(1) == 1.0);
}

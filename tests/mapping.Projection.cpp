// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "common.hpp"

#include <cstdint>

TEST_CASE("mapping.Projection.AoS")
{
    auto mapping = llama::mapping::Projection<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<>>{{}};
    CHECK(mapping.blobSize(0) == 3072);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

namespace
{
    struct DoubleToFloat
    {
        static auto load(float f) -> double
        {
            return f;
        }

        static auto store(double d) -> float
        {
            return static_cast<float>(d);
        }
    };
} // namespace

TEST_CASE("mapping.Projection.AoS.DoubleToFloat")
{
    auto mapping = llama::mapping::Projection<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<mp_list<double, DoubleToFloat>>>{{}};
    CHECK(mapping.blobSize(0) == 1536);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.Projection.ProxyRef.SwapAndAssign")
{
    auto mapping = llama::mapping::Projection<
        llama::ArrayExtents<int, 2>,
        double,
        llama::mapping::BindAoS<>::fn,
        mp_list<mp_list<double, DoubleToFloat>>>{{}};
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

namespace
{
    struct Sqrt
    {
        static auto load(float v) -> double
        {
            return std::sqrt(v);
        }

        static auto store(double d) -> float
        {
            return static_cast<float>(d * d);
        }
    };
} // namespace

TEST_CASE("mapping.Projection.AoS.Sqrt")
{
    auto mapping = llama::mapping::Projection<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<mp_list<double, Sqrt>>>{{}};
    CHECK(mapping.blobSize(0) == 1536);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.Projection.AoS.Coord1.Sqrt")
{
    auto mapping = llama::mapping::Projection<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,
        mp_list<mp_list<llama::RecordCoord<1>, Sqrt>>>{{}};
    CHECK(mapping.blobSize(0) == 2560);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("mapping.Projection.AoS.CoordsAndTypes")
{
    auto mapping = llama::mapping::Projection<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<llama::mapping::FieldAlignment::Pack>::fn,

        mp_list<mp_list<double, DoubleToFloat>, mp_list<llama::RecordCoord<1>, Sqrt>>>{{}};
    CHECK(mapping.blobSize(0) == 1536);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}

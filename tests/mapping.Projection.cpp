#include "common.hpp"

#include <cstdint>

TEST_CASE("mapping.Projection.AoS")
{
    auto mapping = llama::mapping::Projection<
        llama::ArrayExtents<std::size_t, 128>,
        Vec3D,
        llama::mapping::BindAoS<false>::fn,
        boost::mp11::mp_list<>>{{}};
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
        llama::mapping::BindAoS<false>::fn,
        boost::mp11::mp_list<boost::mp11::mp_list<double, DoubleToFloat>>>{{}};
    CHECK(mapping.blobSize(0) == 1536);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
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
        llama::mapping::BindAoS<false>::fn,
        boost::mp11::mp_list<boost::mp11::mp_list<double, Sqrt>>>{{}};
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
        llama::mapping::BindAoS<false>::fn,
        boost::mp11::mp_list<boost::mp11::mp_list<llama::RecordCoord<1>, Sqrt>>>{{}};
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
        llama::mapping::BindAoS<false>::fn,
        boost::mp11::
            mp_list<boost::mp11::mp_list<double, DoubleToFloat>, boost::mp11::mp_list<llama::RecordCoord<1>, Sqrt>>>{
        {}};
    CHECK(mapping.blobSize(0) == 1536);
    auto view = llama::allocView(mapping);
    iotaFillView(view);
    iotaCheckView(view);
}
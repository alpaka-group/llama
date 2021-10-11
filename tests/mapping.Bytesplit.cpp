#include "common.hpp"

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

#include "common.hpp"

using ArrayExtents = llama::ArrayExtents<16>;
using Mapping = llama::mapping::AoS<ArrayExtents, Vec3I>;

TEST_CASE("bloballocators.Stack")
{
    constexpr auto mapping = Mapping{{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Stack<mapping.blobSize(0)>{});
    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("bloballocators.Vector")
{
    auto view = llama::allocView(Mapping{{}}, llama::bloballoc::Vector{});
    iotaFillView(view);
    iotaCheckView(view);
}

#ifndef __clang_analyzer__
TEST_CASE("bloballocators.SharedPtr")
{
    auto view = llama::allocView(Mapping{{}}, llama::bloballoc::SharedPtr{});
    iotaFillView(view);
    iotaCheckView(view);
}
#endif

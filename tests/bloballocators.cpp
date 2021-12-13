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

TEST_CASE("bloballocators.AlignedAllocator")
{
    constexpr auto size = 50;

    auto aa = llama::bloballoc::AlignedAllocator<int, 1024>{};
    auto* p = aa.allocate(size);
    CHECK((reinterpret_cast<std::uintptr_t>(p) & std::uintptr_t{0x3FF}) == 0);
    aa.deallocate(p, size);
    CHECK(aa == llama::bloballoc::AlignedAllocator<int, 1024>{});
    CHECK(!(aa != llama::bloballoc::AlignedAllocator<int, 1024>{}));
}
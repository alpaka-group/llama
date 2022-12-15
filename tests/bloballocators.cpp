#include "common.hpp"

#include <memory>
#include <vector>

using ArrayExtents = llama::ArrayExtents<int, 16>;
using Mapping = llama::mapping::AoS<ArrayExtents, Vec3I>;

TEST_CASE("blob.concept")
{
#ifdef __cpp_lib_concepts
    STATIC_REQUIRE(llama::Blob<llama::Array<std::byte, 10>>);
    STATIC_REQUIRE(llama::Blob<llama::Array<unsigned char, 10>>);
    STATIC_REQUIRE(!llama::Blob<llama::Array<char, 10>>);
    STATIC_REQUIRE(!llama::Blob<llama::Array<double, 10>>);

    STATIC_REQUIRE(llama::Blob<std::vector<std::byte>>);
    STATIC_REQUIRE(llama::Blob<std::unique_ptr<std::byte[]>>);
    STATIC_REQUIRE(llama::Blob<std::shared_ptr<std::byte[]>>);
    STATIC_REQUIRE(llama::Blob<std::byte*>);
#endif
}

TEST_CASE("bloballocators.Array")
{
    constexpr auto mapping = Mapping{{}};
    auto view = llama::allocView(mapping, llama::bloballoc::Array<mapping.blobSize(0)>{});
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
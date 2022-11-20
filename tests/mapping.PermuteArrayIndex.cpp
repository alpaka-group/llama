#include "common.hpp"

TEST_CASE("mapping.PermuteArrayIndex.ctors")
{
    using AE = llama::ArrayExtentsDynamic<int, 3>;
    using InnerMapping = llama::mapping::AoS<AE, Vec3I>;

    auto inner = InnerMapping{AE{3, 4, 5}};
    [[maybe_unused]] auto perm1 = llama::mapping::PermuteArrayIndex<InnerMapping, 2, 0, 1>{};
    [[maybe_unused]] auto perm2 = llama::mapping::PermuteArrayIndex<InnerMapping, 2, 0, 1>{AE{3, 4, 5}};
    [[maybe_unused]] auto perm3 = llama::mapping::PermuteArrayIndex<InnerMapping, 2, 0, 1>{inner};
}

TEST_CASE("mapping.PermuteArrayIndex")
{
    using InnerMapping = llama::mapping::AoS<llama::ArrayExtents<int, 4, 5, 6>, Vec3I>;
    auto inner = InnerMapping{{}};
    auto view = llama::allocView(inner);

    for(int x = 0; x < 4; x++)
        for(int y = 0; y < 5; y++)
            for(int z = 0; z < 6; z++)
            {
                auto&& r = view(x, y, z);
                r(tag::X{}) = x;
                r(tag::Y{}) = y;
                r(tag::Z{}) = z;
            }

    auto view2 = llama::withMapping(std::move(view), llama::mapping::PermuteArrayIndex<InnerMapping, 2, 0, 1>{{}});

    for(int x = 0; x < 4; x++)
        for(int y = 0; y < 5; y++)
            for(int z = 0; z < 6; z++)
            {
                CAPTURE(x, y, z);
                auto&& r = view2(y, z, x); // permuted
                CHECK(r(tag::X{}) == x);
                CHECK(r(tag::Y{}) == y);
                CHECK(r(tag::Z{}) == z);
            }
}

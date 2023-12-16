// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

TEST_CASE("SubView.CTAD")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    constexpr ArrayExtents viewSize{10, 10};
    const auto mapping = llama::mapping::SoA<ArrayExtents, Vec3D>(viewSize);
    auto view = llama::allocViewUninitialized(mapping);

    using View = decltype(view);

    {
        const llama::SubView subView{view, {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::StoredParentView, View&>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::ParentView, View>);
    }

    {
        auto& viewRef = view;
        const llama::SubView subView{viewRef, {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::StoredParentView, View&>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::ParentView, View>);
    }

    {
        const auto& viewConstRef = view;
        const llama::SubView subView{viewConstRef, {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::StoredParentView, const View&>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::ParentView, View>);
    }

    {
        const llama::SubView subView{std::move(view), {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::StoredParentView, View>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(subView)::ParentView, View>);
    }
}

TEST_CASE("SubView.fast")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 2>;
    constexpr ArrayExtents viewSize{10, 10};

    using Mapping = llama::mapping::SoA<ArrayExtents, Vec3D>;
    auto view = llama::allocViewUninitialized(Mapping(viewSize));

    for(int x = 0; x < viewSize[0]; ++x)
        for(int y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    llama::SubView<decltype(view)&> subView{view, {2, 4}};

    CHECK(subView.offset == llama::ArrayIndex{2, 4});

    CHECK(view(subView.offset)(tag::X()) == 8.0);
    CHECK(subView({0, 0})(tag::X()) == 8.0);

    CHECK(view({subView.offset[0] + 2, subView.offset[1] + 3})(tag::Z()) == 28.0);
    CHECK(subView({2, 3})(tag::Z()) == 28.0);
}

namespace
{
    template<typename RecordRef>
    struct DoubleFunctor
    {
        template<typename RecordCoord>
        LLAMA_FN_HOST_ACC_INLINE void operator()(RecordCoord rc)
        {
            ref(rc) *= 2;
        }
        RecordRef ref;
    };
} // namespace

TEST_CASE("SubView")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    constexpr ArrayExtents viewSize{32, 32};
    constexpr ArrayExtents miniSize{8, 8};
    using Mapping = llama::mapping::SoA<ArrayExtents, Vec3D>;
    auto view = llama::allocViewUninitialized(Mapping(viewSize));

    for(std::size_t x = 0; x < viewSize[0]; ++x)
        for(std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    constexpr llama::ArrayIndex iterations{
        (viewSize[0] + miniSize[0] - 1) / miniSize[0],
        (viewSize[1] + miniSize[1] - 1) / miniSize[1]};

    for(std::size_t x = 0; x < iterations[0]; ++x)
        for(std::size_t y = 0; y < iterations[1]; ++y)
        {
            const llama::ArrayIndex validMiniSize{
                (x < iterations[0] - 1) ? miniSize[0] : (viewSize[0] - 1) % miniSize[0] + 1,
                (y < iterations[1] - 1) ? miniSize[1] : (viewSize[1] - 1) % miniSize[1] + 1};

            llama::SubView<decltype(view)&> subView(view, {x * miniSize[0], y * miniSize[1]});

            using MiniMapping = llama::mapping::SoA<ArrayExtents, Vec3D>;
            constexpr auto miniMapping = MiniMapping(miniSize);
            auto miniView
                = llama::allocViewUninitialized(miniMapping, llama::bloballoc::Array<miniMapping.blobSize(0)>{});

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                    miniView(a, b) = subView(a, b);

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                {
                    DoubleFunctor<decltype(miniView(a, b))> sqrtF{miniView(a, b)};
                    llama::forEachLeafCoord<Vec3D>(sqrtF);
                }

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                    subView(a, b) = miniView(a, b);
        }

    for(std::size_t x = 0; x < viewSize[0]; ++x)
        for(std::size_t y = 0; y < viewSize[1]; ++y)
            CHECK((view(x, y)) == x * y * 2);
}

TEST_CASE("SubView.negative_indices")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{10, 10}, int{}});
    auto shiftedView = llama::SubView{view, {2, 4}};

    int i = 0;
    for(int y = -2; y < 8; y++)
        for(int x = -4; x < 6; x++)
            shiftedView(y, x) = i++;

    i = 0;
    for(int y = 0; y < 10; y++)
        for(int x = 0; x < 10; x++)
            CHECK(view(y, x) == i++);
}


TEST_CASE("SubView.negative_offsets")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{10u, 10u}, int{}});
    auto shiftedView = llama::SubView{view, {2, 4}};
    auto shiftedView2 = llama::SubView{shiftedView, {static_cast<unsigned>(-2), static_cast<unsigned>(-4)}};

    int i = 0;
    for(int y = 0; y < 10; y++)
        for(int x = 0; x < 10; x++)
            shiftedView2(y, x) = i++;

    i = 0;
    for(int y = 0; y < 10; y++)
        for(int x = 0; x < 10; x++)
            CHECK(view(y, x) == i++);
}

TEST_CASE("SubView.stored_view")
{
    auto shiftedView = llama::SubView{
        llama::SubView{
            llama::SubView{llama::allocView(llama::mapping::AoS{llama::ArrayExtents{10, 10}, int{}}), {1, 2}},
            {0, 2}},
        {1, 0}};

    int i = 0;
    for(int y = -2; y < 8; y++)
        for(int x = -4; x < 6; x++)
            shiftedView(y, x) = i++;

    i = 0;
    for(int y = 0; y < 10; y++)
        for(int x = 0; x < 10; x++)
            CHECK(shiftedView.parentView.parentView.parentView(y, x) == i++);
}

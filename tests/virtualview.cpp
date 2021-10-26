#include "common.hpp"

TEST_CASE("VirtualView.CTAD")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<2>;
    constexpr ArrayExtents viewSize{10, 10};
    const auto mapping = llama::mapping::SoA<ArrayExtents, Vec3D>(viewSize);
    auto view = llama::allocViewUninitialized(mapping);

    using View = decltype(view);

    {
        llama::VirtualView virtualView{view, {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::StoredParentView, View&>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::ParentView, View>);
    }

    {
        auto& viewRef = view;
        llama::VirtualView virtualView{viewRef, {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::StoredParentView, View&>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::ParentView, View>);
    }

    {
        const auto& viewConstRef = view;
        llama::VirtualView virtualView{viewConstRef, {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::StoredParentView, const View&>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::ParentView, View>);
    }

    {
        llama::VirtualView virtualView{std::move(view), {2, 4}};
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::StoredParentView, View>);
        STATIC_REQUIRE(std::is_same_v<typename decltype(virtualView)::ParentView, View>);
    }
}

TEST_CASE("VirtualView.fast")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<2>;
    constexpr ArrayExtents viewSize{10, 10};

    using Mapping = llama::mapping::SoA<ArrayExtents, Vec3D>;
    auto view = llama::allocViewUninitialized(Mapping(viewSize));

    for(std::size_t x = 0; x < viewSize[0]; ++x)
        for(std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    llama::VirtualView<decltype(view)&> virtualView{view, {2, 4}};

    CHECK(virtualView.offset == llama::ArrayIndex{2, 4});

    CHECK(view(virtualView.offset)(tag::X()) == 8.0);
    CHECK(virtualView({0, 0})(tag::X()) == 8.0);

    CHECK(view({virtualView.offset[0] + 2, virtualView.offset[1] + 3})(tag::Z()) == 28.0);
    CHECK(virtualView({2, 3})(tag::Z()) == 28.0);
}

namespace
{
    template<typename VirtualRecord>
    struct DoubleFunctor
    {
        template<typename RecordCoord>
        void operator()(RecordCoord rc)
        {
            vd(rc) *= 2;
        }
        VirtualRecord vd;
    };
} // namespace

TEST_CASE("VirtualView")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<2>;
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

            llama::VirtualView<decltype(view)&> virtualView(view, {x * miniSize[0], y * miniSize[1]});

            using MiniMapping = llama::mapping::SoA<ArrayExtents, Vec3D>;
            auto miniView = llama::allocViewUninitialized(
                MiniMapping(miniSize),
                llama::bloballoc::Stack<miniSize[0] * miniSize[1] * llama::sizeOf<Vec3D>>{});

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                    miniView(a, b) = virtualView(a, b);

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                {
                    DoubleFunctor<decltype(miniView(a, b))> sqrtF{miniView(a, b)};
                    llama::forEachLeafCoord<Vec3D>(sqrtF);
                }

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                    virtualView(a, b) = miniView(a, b);
        }

    for(std::size_t x = 0; x < viewSize[0]; ++x)
        for(std::size_t y = 0; y < viewSize[1]; ++y)
            CHECK((view(x, y)) == x * y * 2);
}

TEST_CASE("VirtualView.negative_indices")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{10, 10}, int{}});
    auto shiftedView = llama::VirtualView{view, {2, 4}};

    int i = 0;
    for(int y = -2; y < 8; y++)
        for(int x = -4; x < 6; x++)
            shiftedView(y, x) = i++;

    i = 0;
    for(int y = 0; y < 10; y++)
        for(int x = 0; x < 10; x++)
            CHECK(view(y, x) == i++);
}


TEST_CASE("VirtualView.negative_offsets")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{10, 10}, int{}});
    auto shiftedView = llama::VirtualView{view, {2, 4}};
    auto shiftedView2 = llama::VirtualView{shiftedView, {static_cast<std::size_t>(-2), static_cast<std::size_t>(-4)}};

    int i = 0;
    for(int y = 0; y < 10; y++)
        for(int x = 0; x < 10; x++)
            shiftedView2(y, x) = i++;

    i = 0;
    for(int y = 0; y < 10; y++)
        for(int x = 0; x < 10; x++)
            CHECK(view(y, x) == i++);
}

TEST_CASE("VirtualView.stored_view")
{
    auto shiftedView = llama::VirtualView{
        llama::VirtualView{
            llama::VirtualView{llama::allocView(llama::mapping::AoS{llama::ArrayExtents{10, 10}, int{}}), {1, 2}},
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

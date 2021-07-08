#include "common.hpp"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

template<typename VirtualRecord>
struct DoubleFunctor
{
    template<typename Coord>
    void operator()(Coord coord)
    {
        vd(coord) *= 2;
    }
    VirtualRecord vd;
};

TEST_CASE("virtual view CTAD")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims viewSize{10, 10};
    auto view = allocView(llama::mapping::SoA<ArrayDims, Particle>(viewSize));

    llama::VirtualView virtualView{view, {2, 4}};
}

TEST_CASE("fast virtual view")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims viewSize{10, 10};

    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    auto view = allocView(Mapping(viewSize));

    for(std::size_t x = 0; x < viewSize[0]; ++x)
        for(std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    llama::VirtualView<decltype(view)> virtualView{view, {2, 4}};

    CHECK(virtualView.offset == ArrayDims{2, 4});

    CHECK(view(virtualView.offset)(tag::Pos(), tag::X()) == 8.0);
    CHECK(virtualView({0, 0})(tag::Pos(), tag::X()) == 8.0);

    CHECK(view({virtualView.offset[0] + 2, virtualView.offset[1] + 3})(tag::Vel(), tag::Z()) == 28.0);
    CHECK(virtualView({2, 3})(tag::Vel(), tag::Z()) == 28.0);
}

TEST_CASE("virtual view")
{
    using ArrayDims = llama::ArrayDims<2>;
    constexpr ArrayDims viewSize{32, 32};
    constexpr ArrayDims miniSize{8, 8};
    using Mapping = llama::mapping::SoA<ArrayDims, Particle>;
    auto view = allocView(Mapping(viewSize));

    for(std::size_t x = 0; x < viewSize[0]; ++x)
        for(std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    constexpr ArrayDims iterations{
        (viewSize[0] + miniSize[0] - 1) / miniSize[0],
        (viewSize[1] + miniSize[1] - 1) / miniSize[1]};

    for(std::size_t x = 0; x < iterations[0]; ++x)
        for(std::size_t y = 0; y < iterations[1]; ++y)
        {
            const ArrayDims validMiniSize{
                (x < iterations[0] - 1) ? miniSize[0] : (viewSize[0] - 1) % miniSize[0] + 1,
                (y < iterations[1] - 1) ? miniSize[1] : (viewSize[1] - 1) % miniSize[1] + 1};

            llama::VirtualView<decltype(view)> virtualView(view, {x * miniSize[0], y * miniSize[1]});

            using MiniMapping = llama::mapping::SoA<ArrayDims, Particle>;
            auto miniView = allocView(
                MiniMapping(miniSize),
                llama::bloballoc::Stack<miniSize[0] * miniSize[1] * llama::sizeOf<Particle>>{});

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                    miniView(a, b) = virtualView(a, b);

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                {
                    DoubleFunctor<decltype(miniView(a, b))> sqrtF{miniView(a, b)};
                    llama::forEachLeaf<Particle>(sqrtF);
                }

            for(std::size_t a = 0; a < validMiniSize[0]; ++a)
                for(std::size_t b = 0; b < validMiniSize[1]; ++b)
                    virtualView(a, b) = miniView(a, b);
        }

    for(std::size_t x = 0; x < viewSize[0]; ++x)
        for(std::size_t y = 0; y < viewSize[1]; ++y)
            CHECK((view(x, y)) == x * y * 2);
}

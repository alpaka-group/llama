#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct X {};
    struct Y {};
    struct Z {};
    struct Pos {};
    struct Vel {};
    struct Mom {};
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, int>,
        llama::DE<tag::Y, int>,
        llama::DE<tag::Z, int>
    >>,
    llama::DE<tag::Mom, int>,
    llama::DE<tag::Vel, llama::DS<
        llama::DE<tag::Z, int>,
        llama::DE<tag::Y, int>,
        llama::DE<tag::X, int>
    >>
>;
// clang-format on

template <typename VirtualDatum>
struct SqrtFunctor
{
    template <typename Coord>
    void operator()(Coord coord)
    {
        vd(coord) *= std::sqrt(vd(coord));
    }
    VirtualDatum vd;
};

TEST_CASE("fast virtual view")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{4096, 4096};

    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    auto view = allocView(Mapping(viewSize));

    for (std::size_t x = 0; x < viewSize[0]; ++x)
        for (std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    llama::VirtualView<decltype(view)> virtualView{
        view,
        {23, 42}, // offset
        {13, 37} // size
    };

    CHECK(virtualView.offset == ArrayDomain{23, 42});
    CHECK(virtualView.size == ArrayDomain{13, 37});

    CHECK(view(virtualView.offset)(tag::Pos(), tag::X()) == 966);
    CHECK(virtualView({0, 0})(tag::Pos(), tag::X()) == 966);

    CHECK(view({virtualView.offset[0] + 2, virtualView.offset[1] + 3})(tag::Vel(), tag::Z()) == 1125);
    CHECK(virtualView({2, 3})(tag::Vel(), tag::Z()) == 1125);
}

TEST_CASE("virtual view")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{256, 256};
    constexpr ArrayDomain miniSize{8, 8};
    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    auto view = allocView(Mapping(viewSize));

    for (std::size_t x = 0; x < viewSize[0]; ++x)
        for (std::size_t y = 0; y < viewSize[1]; ++y)
            view(x, y) = x * y;

    constexpr ArrayDomain iterations{
        (viewSize[0] + miniSize[0] - 1) / miniSize[0],
        (viewSize[1] + miniSize[1] - 1) / miniSize[1]};

    for (std::size_t x = 0; x < iterations[0]; ++x)
        for (std::size_t y = 0; y < iterations[1]; ++y)
        {
            const ArrayDomain validMiniSize{
                (x < iterations[0] - 1) ? miniSize[0] : (viewSize[0] - 1) % miniSize[0] + 1,
                (y < iterations[1] - 1) ? miniSize[1] : (viewSize[1] - 1) % miniSize[1] + 1};

            llama::VirtualView<decltype(view)> virtualView(view, {x * miniSize[0], y * miniSize[1]}, miniSize);

            using MiniMapping = llama::mapping::SoA<ArrayDomain, Particle>;
            auto miniView = allocView(
                MiniMapping(miniSize),
                llama::allocator::Stack<miniSize[0] * miniSize[1] * llama::sizeOf<Particle>>{});

            for (std::size_t a = 0; a < validMiniSize[0]; ++a)
                for (std::size_t b = 0; b < validMiniSize[1]; ++b)
                    miniView(a, b) = virtualView(a, b);

            for (std::size_t a = 0; a < validMiniSize[0]; ++a)
                for (std::size_t b = 0; b < validMiniSize[1]; ++b)
                {
                    SqrtFunctor<decltype(miniView(a, b))> sqrtF{miniView(a, b)};
                    llama::forEach<Particle>(sqrtF);
                }

            for (std::size_t a = 0; a < validMiniSize[0]; ++a)
                for (std::size_t b = 0; b < validMiniSize[1]; ++b)
                    virtualView(a, b) = miniView(a, b);
        }

    for (std::size_t x = 0; x < viewSize[0]; ++x)
        for (std::size_t y = 0; y < viewSize[1]; ++y)
            CHECK((view(x, y) == x * y * std::sqrt(x * y)));
}

#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Value {};
}

using DatumDomain = llama::DS<
    llama::DE<tag::Value, int>
>;
// clang-format on

TEST_CASE("view.default-ctor")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    [[maybe_unused]] llama::View<llama::mapping::SoA<ArrayDomain, DatumDomain>, std::byte*> view1;
    [[maybe_unused]] llama::View<llama::mapping::AoS<ArrayDomain, DatumDomain>, std::byte*> view2;
    [[maybe_unused]] llama::View<llama::mapping::One<ArrayDomain, DatumDomain>, std::byte*> view3;
    [[maybe_unused]] llama::View<llama::mapping::tree::Mapping<ArrayDomain, DatumDomain, llama::Tuple<>>, std::byte*>
        view4;
}

TEST_CASE("view.move")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, DatumDomain>;
    auto view1 = allocView(Mapping(viewSize));

    decltype(view1) view2;
    view1({3, 3}) = 1;
    view2 = std::move(view1);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view.swap")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, DatumDomain>;
    auto view1 = allocView(Mapping(viewSize));
    auto view2 = allocView(Mapping(viewSize));

    view1({3, 3}) = 1;
    view2({3, 3}) = 2;

    std::swap(view1, view2);

    CHECK(view1({3, 3}) == 2);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view.allocator.Vector")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::Vector{});

    for (auto i : llama::ArrayDomainIndexRange{viewSize})
        view(i) = 42;
}

TEST_CASE("view.allocator.SharedPtr")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::SharedPtr{});

    for (auto i : llama::ArrayDomainIndexRange{viewSize})
        view(i) = 42;
}

TEST_CASE("view.allocator.stack")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::Stack<16 * 16 * llama::sizeOf<DatumDomain>>{});

    for (auto i : llama::ArrayDomainIndexRange{viewSize})
        view(i) = 42;
}

TEST_CASE("view.non-memory-owning")
{
    using ArrayDomain = llama::ArrayDomain<1>;
    ArrayDomain arrayDomain{256};

    using Mapping = llama::mapping::SoA<ArrayDomain, DatumDomain>;
    Mapping mapping{arrayDomain};

    std::vector<std::byte> storage(mapping.getBlobSize(0));
    auto view = llama::View<Mapping, std::byte*>{mapping, {storage.data()}};

    for (auto i = 0u; i < 256u; i++)
    {
        auto* v = (std::byte*) &view(i)(tag::Value{});
        CHECK(&storage.front() <= v);
        CHECK(v <= &storage.back());
    }
}

// clang-format off
namespace tag {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Flags {};
}

// clang-format off
using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Weight, float>,
    llama::DE<tag::Momentum, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;
// clang-format on

TEST_CASE("view.access")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    auto l = [](auto& view) {
        const ArrayDomain pos{0, 0};
        CHECK((view(pos) == view[pos]));
        CHECK((view(pos) == view[{0, 0}]));
        CHECK((view(pos) == view({0, 0})));

        const auto& x = view(pos)(llama::DatumCoord<0, 0>{});
        CHECK(&x == &view(pos)(llama::DatumCoord<0>{})(llama::DatumCoord<0>{}));
        CHECK(&x == &view(pos)(llama::DatumCoord<0>{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{})(llama::DatumCoord<0>{}));
        CHECK(&x == &view(pos)(tag::Pos{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{}, tag::X{}));

        // also test arrays
        using namespace llama::literals;
        const bool& o0 = view(pos)(tag::Flags{})(llama::DatumCoord<0>{});
        CHECK(&o0 == &view(pos)(tag::Flags{})(0_DC));
    };
    l(view);
    l(std::as_const(view));
}

TEST_CASE("view.assign-one-datum")
{
    using namespace llama::literals;

    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    llama::One<Particle> datum;
    datum(tag::Pos{}, tag::X{}) = 14.0f;
    datum(tag::Pos{}, tag::Y{}) = 15.0f;
    datum(tag::Pos{}, tag::Z{}) = 16.0f;
    datum(tag::Momentum{}) = 0;
    datum(tag::Weight{}) = 500.0f;
    datum(tag::Flags{})(0_DC) = true;
    datum(tag::Flags{})(1_DC) = false;
    datum(tag::Flags{})(2_DC) = true;
    datum(tag::Flags{})(3_DC) = false;

    view({3, 4}) = datum;

    CHECK(datum(tag::Pos{}, tag::X{}) == 14.0f);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 15.0f);
    CHECK(datum(tag::Pos{}, tag::Z{}) == 16.0f);
    CHECK(datum(tag::Momentum{}, tag::X{}) == 0);
    CHECK(datum(tag::Momentum{}, tag::Y{}) == 0);
    CHECK(datum(tag::Momentum{}, tag::Z{}) == 0);
    CHECK(datum(tag::Weight{}) == 500.0f);
    CHECK(datum(tag::Flags{})(0_DC) == true);
    CHECK(datum(tag::Flags{})(1_DC) == false);
    CHECK(datum(tag::Flags{})(2_DC) == true);
    CHECK(datum(tag::Flags{})(3_DC) == false);
}

TEST_CASE("view.addresses")
{
    using namespace llama::literals;

    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    const ArrayDomain pos{0, 0};
    auto& x = view(pos)(tag::Pos{}, tag::X{});
    auto& y = view(pos)(tag::Pos{}, tag::Y{});
    auto& z = view(pos)(tag::Pos{}, tag::Z{});
    auto& w = view(pos)(tag::Weight{});
    auto& mx = view(pos)(tag::Momentum{}, tag::X{});
    auto& my = view(pos)(tag::Momentum{}, tag::Y{});
    auto& mz = view(pos)(tag::Momentum{}, tag::Z{});
    auto& o0 = view(pos)(tag::Flags{})(0_DC);
    auto& o1 = view(pos)(tag::Flags{})(1_DC);
    auto& o2 = view(pos)(tag::Flags{})(2_DC);
    auto& o3 = view(pos)(tag::Flags{})(3_DC);

    CHECK((size_t) &y - (size_t) &x == 2048);
    CHECK((size_t) &z - (size_t) &x == 4096);
    CHECK((size_t) &mx - (size_t) &x == 7168);
    CHECK((size_t) &my - (size_t) &x == 9216);
    CHECK((size_t) &mz - (size_t) &x == 11264);
    CHECK((size_t) &w - (size_t) &x == 6144);
    CHECK((size_t) &o0 - (size_t) &x == 13312);
    CHECK((size_t) &o1 - (size_t) &x == 13568);
    CHECK((size_t) &o2 - (size_t) &x == 13824);
    CHECK((size_t) &o3 - (size_t) &x == 14080);
}

template <typename VirtualDatum>
struct SetZeroFunctor
{
    template <typename Coord>
    void operator()(Coord coord)
    {
        vd(coord) = 0;
    }
    VirtualDatum vd;
};

TEST_CASE("view.iteration-and-access")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for (size_t x = 0; x < arrayDomain[0]; ++x)
        for (size_t y = 0; y < arrayDomain[1]; ++y)
        {
            SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
            llama::forEach<Particle>(szf, llama::DatumCoord<0, 0>{});
            llama::forEach<Particle>(szf, tag::Momentum{});
            view({x, y}) = double(x + y) / double(arrayDomain[0] + arrayDomain[1]);
        }

    double sum = 0.0;
    for (size_t x = 0; x < arrayDomain[0]; ++x)
        for (size_t y = 0; y < arrayDomain[1]; ++y)
            sum += view(x, y)(llama::DatumCoord<2, 0>{});
    CHECK(sum == 120.0);
}

TEST_CASE("view.datum-access")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for (size_t x = 0; x < arrayDomain[0]; ++x)
        for (size_t y = 0; y < arrayDomain[1]; ++y)
        {
            auto datum = view(x, y);
            datum(tag::Pos(), tag::X()) += datum(llama::DatumCoord<2, 0>{});
            datum(tag::Pos(), tag::Y()) += datum(llama::DatumCoord<2, 1>{});
            datum(tag::Pos(), tag::Z()) += datum(llama::DatumCoord<1>());
            datum(tag::Pos()) += datum(tag::Momentum());
        }

    double sum = 0.0;
    for (size_t x = 0; x < arrayDomain[0]; ++x)
        for (size_t y = 0; y < arrayDomain[1]; ++y)
            sum += view(x, y)(llama::DatumCoord<2, 0>{});

    CHECK(sum == 0.0);
}

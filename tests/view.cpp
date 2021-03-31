#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Value {};
} // namespace tag

using RecordDim = llama::Record<
    llama::Field<tag::Value, int>
>;
// clang-format on

TEST_CASE("view.default-ctor")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    [[maybe_unused]] llama::View<llama::mapping::SoA<ArrayDomain, RecordDim>, std::byte*> view1;
    [[maybe_unused]] llama::View<llama::mapping::AoS<ArrayDomain, RecordDim>, std::byte*> view2;
    [[maybe_unused]] llama::View<llama::mapping::One<ArrayDomain, RecordDim>, std::byte*> view3;
    [[maybe_unused]] llama::View<llama::mapping::tree::Mapping<ArrayDomain, RecordDim, llama::Tuple<>>, std::byte*>
        view4;
}

TEST_CASE("view.move")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, RecordDim>;
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

    using Mapping = llama::mapping::SoA<ArrayDomain, RecordDim>;
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

    using Mapping = llama::mapping::SoA<ArrayDomain, RecordDim>;
    auto view = allocView(Mapping(viewSize), llama::bloballoc::Vector{});

    for (auto i : llama::ArrayDomainIndexRange{viewSize})
        view(i) = 42;
}

TEST_CASE("view.allocator.SharedPtr")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, RecordDim>;
    auto view = allocView(Mapping(viewSize), llama::bloballoc::SharedPtr{});

    for (auto i : llama::ArrayDomainIndexRange{viewSize})
        view(i) = 42;
}

TEST_CASE("view.allocator.stack")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr ArrayDomain viewSize{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, RecordDim>;
    auto view = allocView(Mapping(viewSize), llama::bloballoc::Stack<16 * 16 * llama::sizeOf<RecordDim>>{});

    for (auto i : llama::ArrayDomainIndexRange{viewSize})
        view(i) = 42;
}

TEST_CASE("view.non-memory-owning")
{
    using ArrayDomain = llama::ArrayDomain<1>;
    ArrayDomain arrayDomain{256};

    using Mapping = llama::mapping::SoA<ArrayDomain, RecordDim>;
    Mapping mapping{arrayDomain};

    std::vector<std::byte> storage(mapping.blobSize(0));
    auto view = llama::View<Mapping, std::byte*>{mapping, {storage.data()}};

    for (auto i = 0u; i < 256u; i++)
    {
        auto* v = reinterpret_cast<std::byte*>(&view(i)(tag::Value{}));
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
} // namespace tag

// clang-format off
using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, double>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, double>
    >>,
    llama::Field<tag::Weight, float>,
    llama::Field<tag::Momentum, llama::Record<
        llama::Field<tag::X, double>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, double>
    >>,
    llama::Field<tag::Flags, bool[4]>
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

    auto l = [](auto& view)
    {
        const ArrayDomain pos{0, 0};
        CHECK((view(pos) == view[pos]));
        CHECK((view(pos) == view[{0, 0}]));
        CHECK((view(pos) == view({0, 0})));

        const auto& x = view(pos)(llama::RecordCoord<0, 0>{});
        CHECK(&x == &view(pos)(llama::RecordCoord<0>{})(llama::RecordCoord<0>{}));
        CHECK(&x == &view(pos)(llama::RecordCoord<0>{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{})(llama::RecordCoord<0>{}));
        CHECK(&x == &view(pos)(tag::Pos{})(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{}, tag::X{}));

        // also test arrays
        using namespace llama::literals;
        const bool& o0 = view(pos)(tag::Flags{})(llama::RecordCoord<0>{});
        CHECK(&o0 == &view(pos)(tag::Flags{})(0_RC));
    };
    l(view);
    l(std::as_const(view));
}

TEST_CASE("view.assign-one-record")
{
    using namespace llama::literals;

    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Particle>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    llama::One<Particle> record;
    record(tag::Pos{}, tag::X{}) = 14.0f;
    record(tag::Pos{}, tag::Y{}) = 15.0f;
    record(tag::Pos{}, tag::Z{}) = 16.0f;
    record(tag::Momentum{}) = 0;
    record(tag::Weight{}) = 500.0f;
    record(tag::Flags{})(0_RC) = true;
    record(tag::Flags{})(1_RC) = false;
    record(tag::Flags{})(2_RC) = true;
    record(tag::Flags{})(3_RC) = false;

    view({3, 4}) = record;

    CHECK(record(tag::Pos{}, tag::X{}) == 14.0f);
    CHECK(record(tag::Pos{}, tag::Y{}) == 15.0f);
    CHECK(record(tag::Pos{}, tag::Z{}) == 16.0f);
    CHECK(record(tag::Momentum{}, tag::X{}) == 0);
    CHECK(record(tag::Momentum{}, tag::Y{}) == 0);
    CHECK(record(tag::Momentum{}, tag::Z{}) == 0);
    CHECK(record(tag::Weight{}) == 500.0f);
    CHECK(record(tag::Flags{})(0_RC) == true);
    CHECK(record(tag::Flags{})(1_RC) == false);
    CHECK(record(tag::Flags{})(2_RC) == true);
    CHECK(record(tag::Flags{})(3_RC) == false);
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
    auto& o0 = view(pos)(tag::Flags{})(0_RC);
    auto& o1 = view(pos)(tag::Flags{})(1_RC);
    auto& o2 = view(pos)(tag::Flags{})(2_RC);
    auto& o3 = view(pos)(tag::Flags{})(3_RC);

    CHECK(reinterpret_cast<std::byte*>(&y) - reinterpret_cast<std::byte*>(&x) == 2048);
    CHECK(reinterpret_cast<std::byte*>(&z) - reinterpret_cast<std::byte*>(&x) == 4096);
    CHECK(reinterpret_cast<std::byte*>(&mx) - reinterpret_cast<std::byte*>(&x) == 7168);
    CHECK(reinterpret_cast<std::byte*>(&my) - reinterpret_cast<std::byte*>(&x) == 9216);
    CHECK(reinterpret_cast<std::byte*>(&mz) - reinterpret_cast<std::byte*>(&x) == 11264);
    CHECK(reinterpret_cast<std::byte*>(&w) - reinterpret_cast<std::byte*>(&x) == 6144);
    CHECK(reinterpret_cast<std::byte*>(&o0) - reinterpret_cast<std::byte*>(&x) == 13312);
    CHECK(reinterpret_cast<std::byte*>(&o1) - reinterpret_cast<std::byte*>(&x) == 13568);
    CHECK(reinterpret_cast<std::byte*>(&o2) - reinterpret_cast<std::byte*>(&x) == 13824);
    CHECK(reinterpret_cast<std::byte*>(&o3) - reinterpret_cast<std::byte*>(&x) == 14080);
}

template <typename VirtualRecord>
struct SetZeroFunctor
{
    template <typename Coord>
    void operator()(Coord coord)
    {
        vd(coord) = 0;
    }
    VirtualRecord vd;
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
            llama::forEachLeaf<Particle>(szf, llama::RecordCoord<0, 0>{});
            llama::forEachLeaf<Particle>(szf, tag::Momentum{});
            view({x, y}) = double(x + y) / double(arrayDomain[0] + arrayDomain[1]);
        }

    double sum = 0.0;
    for (size_t x = 0; x < arrayDomain[0]; ++x)
        for (size_t y = 0; y < arrayDomain[1]; ++y)
            sum += view(x, y)(llama::RecordCoord<2, 0>{});
    CHECK(sum == 120.0);
}

TEST_CASE("view.record-access")
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
            auto record = view(x, y);
            record(tag::Pos(), tag::X()) += record(llama::RecordCoord<2, 0>{});
            record(tag::Pos(), tag::Y()) += record(llama::RecordCoord<2, 1>{});
            record(tag::Pos(), tag::Z()) += record(llama::RecordCoord<1>());
            record(tag::Pos()) += record(tag::Momentum());
        }

    double sum = 0.0;
    for (size_t x = 0; x < arrayDomain[0]; ++x)
        for (size_t y = 0; y < arrayDomain[1]; ++y)
            sum += view(x, y)(llama::RecordCoord<2, 0>{});

    CHECK(sum == 0.0);
}

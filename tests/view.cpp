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
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize {16, 16};

    [[maybe_unused]] llama::View<llama::mapping::SoA<UserDomain, DatumDomain>, std::byte*> view1;
    [[maybe_unused]] llama::View<llama::mapping::AoS<UserDomain, DatumDomain>, std::byte*> view2;
    [[maybe_unused]] llama::View<llama::mapping::One<UserDomain, DatumDomain>, std::byte*> view3;
    [[maybe_unused]] llama::View<llama::mapping::tree::Mapping<UserDomain, DatumDomain, llama::Tuple<>>, std::byte*>
        view4;
}

TEST_CASE("view.move")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view1 = allocView(Mapping(viewSize));

    decltype(view1) view2;
    view1({3, 3}) = 1;
    view2 = std::move(view1);
    CHECK(view2({3, 3}) == 1);
}

TEST_CASE("view.swap")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
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
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::Vector {});

    for (auto i : llama::UserDomainCoordRange {viewSize})
        view(i) = 42;
}

TEST_CASE("view.allocator.SharedPtr")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::SharedPtr {});

    for (auto i : llama::UserDomainCoordRange {viewSize})
        view(i) = 42;
}

TEST_CASE("view.allocator.stack")
{
    using UserDomain = llama::UserDomain<2>;
    constexpr UserDomain viewSize {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    auto view = allocView(Mapping(viewSize), llama::allocator::Stack<16 * 16 * llama::sizeOf<DatumDomain>> {});

    for (auto i : llama::UserDomainCoordRange {viewSize})
        view(i) = 42;
}

TEST_CASE("view.non-memory-owning")
{
    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain {256};

    using Mapping = llama::mapping::SoA<UserDomain, DatumDomain>;
    Mapping mapping {userDomain};

    std::vector<std::byte> storage(mapping.getBlobSize(0));
    auto view = llama::View<Mapping, std::byte*> {mapping, {storage.data()}};

    for (auto i = 0u; i < 256u; i++)
    {
        auto* v = (std::byte*) &view(i)(tag::Value {});
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
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    auto l = [](auto& view) {
        const UserDomain pos {0, 0};
        CHECK((view(pos) == view[pos]));
        CHECK((view(pos) == view[{0, 0}]));
        CHECK((view(pos) == view({0, 0})));

        const auto& x = view(pos).template access<0, 0>();
        CHECK(&x == &view(pos).template access<0>().template access<0>());
        CHECK(&x == &view(pos).template access<0>().template access<tag::X>());
        CHECK(&x == &view(pos).template access<tag::Pos>().template access<0>());
        CHECK(&x == &view(pos).template access<tag::Pos, tag::X>());
        CHECK(&x == &view(pos).template access<tag::Pos>().template access<tag::X>());
        CHECK(&x == &view(pos)(tag::Pos {}, tag::X {}));
        CHECK(&x == &view(pos)(tag::Pos {})(tag::X {}));
        CHECK(&x == &view(pos).template access<tag::Pos>()(tag::X {}));
        CHECK(&x == &view(pos)(tag::Pos {}).template access<tag::X>());
        CHECK(&x == &view(pos)(llama::DatumCoord<0, 0> {}));
        CHECK(&x == &view(pos)(llama::DatumCoord<0> {})(llama::DatumCoord<0> {}));
        CHECK(&x == &view(pos).template access<llama::DatumCoord<0, 0>>());
        CHECK(&x == &view(pos).template access<llama::DatumCoord<0>>().template access<llama::DatumCoord<0>>());
        // there are even more combinations

        // also test arrays
        const bool& o0 = view(pos).template access<tag::Flags>().template access<0>();
        CHECK(&o0 == &view(pos).template access<tag::Flags>().template access<llama::Index<0>>());
        CHECK(&o0 == &view(pos).template access<tag::Flags>().access(llama::Index<0> {}));
        CHECK(&o0 == &view(pos).template access<tag::Flags>()(llama::Index<0> {}));
    };
    l(view);
    l(std::as_const(view));
}

TEST_CASE("view.assign-one-datum")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    auto datum = llama::allocVirtualDatumStack<Particle>();
    datum(tag::Pos {}, tag::X {}) = 14.0f;
    datum(tag::Pos {}, tag::Y {}) = 15.0f;
    datum(tag::Pos {}, tag::Z {}) = 16.0f;
    datum(tag::Momentum {}) = 0;
    datum(tag::Weight {}) = 500.0f;
    datum(tag::Flags {}).access<0>() = true;
    datum(tag::Flags {}).access<1>() = false;
    datum(tag::Flags {}).access<2>() = true;
    datum(tag::Flags {}).access<3>() = false;

    view({3, 4}) = datum;

    CHECK(datum(tag::Pos {}, tag::X {}) == 14.0f);
    CHECK(datum(tag::Pos {}, tag::Y {}) == 15.0f);
    CHECK(datum(tag::Pos {}, tag::Z {}) == 16.0f);
    CHECK(datum(tag::Momentum {}, tag::X {}) == 0);
    CHECK(datum(tag::Momentum {}, tag::Y {}) == 0);
    CHECK(datum(tag::Momentum {}, tag::Z {}) == 0);
    CHECK(datum(tag::Weight {}) == 500.0f);
    CHECK(datum(tag::Flags {}).access<0>() == true);
    CHECK(datum(tag::Flags {}).access<1>() == false);
    CHECK(datum(tag::Flags {}).access<2>() == true);
    CHECK(datum(tag::Flags {}).access<3>() == false);
}

TEST_CASE("view.addresses")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    const UserDomain pos {0, 0};
    auto& x = view(pos).access<tag::Pos, tag::X>();
    auto& y = view(pos).access<tag::Pos, tag::Y>();
    auto& z = view(pos).access<tag::Pos, tag::Z>();
    auto& w = view(pos).access<tag::Weight>();
    auto& mx = view(pos).access<tag::Momentum, tag::X>();
    auto& my = view(pos).access<tag::Momentum, tag::Y>();
    auto& mz = view(pos).access<tag::Momentum, tag::Z>();
    auto& o0 = view(pos).access<tag::Flags>().access<0>();
    auto& o1 = view(pos).access<tag::Flags>().access<1>();
    auto& o2 = view(pos).access<tag::Flags>().access<2>();
    auto& o3 = view(pos).access<tag::Flags>().access<3>();

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
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for (size_t x = 0; x < userDomain[0]; ++x)
        for (size_t y = 0; y < userDomain[1]; ++y)
        {
            SetZeroFunctor<decltype(view(x, y))> szf {view(x, y)};
            llama::forEach<Particle>(szf, llama::DatumCoord<0, 0> {});
            llama::forEach<Particle>(szf, tag::Momentum {});
            view({x, y}) = double(x + y) / double(userDomain[0] + userDomain[1]);
        }

    double sum = 0.0;
    for (size_t x = 0; x < userDomain[0]; ++x)
        for (size_t y = 0; y < userDomain[1]; ++y)
            sum += view(x, y).access<2, 0>();
    CHECK(sum == 120.0);
}

TEST_CASE("view.datum-access")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain {16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping {userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for (size_t x = 0; x < userDomain[0]; ++x)
        for (size_t y = 0; y < userDomain[1]; ++y)
        {
            auto datum = view(x, y);
            datum.access<tag::Pos, tag::X>() += datum.access<llama::DatumCoord<2, 0>>();
            datum.access(tag::Pos(), tag::Y()) += datum.access(llama::DatumCoord<2, 1>());
            datum(tag::Pos(), tag::Z()) += datum(llama::DatumCoord<1>());
            datum(tag::Pos()) += datum(tag::Momentum());
        }

    double sum = 0.0;
    for (size_t x = 0; x < userDomain[0]; ++x)
        for (size_t y = 0; y < userDomain[1]; ++y)
            sum += view(x, y).access<2, 0>();

    CHECK(sum == 0.0);
}

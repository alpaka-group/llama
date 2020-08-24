#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Options {};
}

// clang-format off
using Name = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>,
        llama::DE<tag::Z, float>
    >>,
    llama::DE<tag::Momentum, llama::DS<
        llama::DE<tag::Z, double>,
        llama::DE<tag::X, double>
    >>,
    llama::DE<tag::Weight, int>,
    llama::DE<tag::Options, llama::DA<bool, 4>>
>;
// clang-format on
// clang-format on

TEST_CASE("demangleType")
{
    const auto str = prettyPrintType(Name());
    CHECK(str == R"(boost::mp11::mp_list<
    boost::mp11::mp_list<
        tag::Pos,
        boost::mp11::mp_list<
            boost::mp11::mp_list<
                tag::X,
                float
            >,
            boost::mp11::mp_list<
                tag::Y,
                float
            >,
            boost::mp11::mp_list<
                tag::Z,
                float
            >
        >
    >,
    boost::mp11::mp_list<
        tag::Momentum,
        boost::mp11::mp_list<
            boost::mp11::mp_list<
                tag::Z,
                double
            >,
            boost::mp11::mp_list<
                tag::X,
                double
            >
        >
    >,
    boost::mp11::mp_list<
        tag::Weight,
        int
    >,
    boost::mp11::mp_list<
        tag::Options,
        boost::mp11::mp_list<
            boost::mp11::mp_list<
                llama::NoName,
                bool
            >,
            boost::mp11::mp_list<
                llama::NoName,
                bool
            >,
            boost::mp11::mp_list<
                llama::NoName,
                bool
            >,
            boost::mp11::mp_list<
                llama::NoName,
                bool
            >
        >
    >
>)");
}

TEST_CASE("AoS address")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};
    const auto address = llama::mapping::AoS<UserDomain, Name>(userDomain)
                             .getBlobNrAndOffset<0, 1>({0, 100})
                             .offset;
    CHECK(address == 3604);
}

TEST_CASE("AoS address fortran")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};
    const auto address
        = llama::mapping::AoS<
              UserDomain,
              Name,
              llama::LinearizeUserDomainAdressLikeFortran>(userDomain)
              .getBlobNrAndOffset<0, 1>({0, 100})
              .offset;
    CHECK(address == 57604);
}

TEST_CASE("SoA address")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};
    const auto address = llama::mapping::SoA<UserDomain, Name>(userDomain)
                             .getBlobNrAndOffset<0, 1>({0, 100})
                             .offset;
    CHECK(address == 1424);
}

TEST_CASE("SoA address fortran")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};
    const auto address
        = llama::mapping::SoA<
              UserDomain,
              Name,
              llama::LinearizeUserDomainAdressLikeFortran>(userDomain)
              .getBlobNrAndOffset<0, 1>({0, 100})
              .offset;
    CHECK(address == 7424);
}

TEST_CASE("SizeOf DatumDomain")
{
    const auto size = llama::SizeOf<Name>::value;
    CHECK(size == 36);
}

TEST_CASE("StubType")
{
    using NameStub = llama::StubType<Name>;
    static_assert(std::is_same_v<Name, NameStub::type>);
    const auto size = llama::SizeOf<Name>::value;
    const auto stubSize = sizeof(NameStub);
    CHECK(size == stubSize);
}

TEST_CASE("GetCoordFromUID")
{
    const auto str
        = prettyPrintType(llama::GetCoordFromUID<Name, tag::Pos, tag::X>());
    CHECK(str == R"(llama::DatumCoord<
    0,
    0
>)");
}

TEST_CASE("access")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    auto l = [](auto & view) {
        const UserDomain pos{0, 0};
        CHECK((view(pos) == view[pos]));
        CHECK((view(pos) == view[{0, 0}]));
        CHECK((view(pos) == view({0, 0})));

        const float & x = view(pos).template access<0, 0>();
        CHECK(&x == &view(pos).template access<0>().template access<0>());
        CHECK(&x == &view(pos).template access<0>().template access<tag::X>());
        CHECK(
            &x == &view(pos).template access<tag::Pos>().template access<0>());
        CHECK(&x == &view(pos).template access<tag::Pos, tag::X>());
        CHECK(
            &x
            == &view(pos)
                    .template access<tag::Pos>()
                    .template access<tag::X>());
        CHECK(&x == &view(pos)(tag::Pos{}, tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{})(tag::X{}));
        CHECK(&x == &view(pos).template access<tag::Pos>()(tag::X{}));
        CHECK(&x == &view(pos)(tag::Pos{}).template access<tag::X>());
        CHECK(&x == &view(pos)(llama::DatumCoord<0, 0>{}));
        CHECK(&x == &view(pos)(llama::DatumCoord<0>{})(llama::DatumCoord<0>{}));
        // there are even more combinations
    };
    l(view);
    l(std::as_const(view));
}

TEST_CASE("addresses")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    const UserDomain pos{0, 0};
    auto & x = view(pos).access<tag::Pos, tag::X>();
    auto & y = view(pos).access<tag::Pos, tag::Y>();
    auto & z = view(pos).access<tag::Pos, tag::Z>();
    auto & mz = view(pos).access<tag::Momentum, tag::Z>();
    auto & mx = view(pos).access<tag::Momentum, tag::X>();
    auto & w = view(pos)(llama::DatumCoord<2>());
    auto & o0 = view(pos).access<tag::Options>().access<0>();
    auto & o1 = view(pos).access<tag::Options>().access<1>();
    auto & o2 = view(pos).access<tag::Options>().access<2>();
    auto & o3 = view(pos).access<tag::Options>().access<3>();

    CHECK((size_t)&y - (size_t)&x == 1024);
    CHECK((size_t)&z - (size_t)&x == 2048);
    CHECK((size_t)&mz - (size_t)&x == 3072);
    CHECK((size_t)&mx - (size_t)&x == 5120);
    CHECK((size_t)&w - (size_t)&x == 7168);
    CHECK((size_t)&o0 - (size_t)&x == 8192);
    CHECK((size_t)&o1 - (size_t)&x == 8448);
    CHECK((size_t)&o2 - (size_t)&x == 8704);
    CHECK((size_t)&o3 - (size_t)&x == 8960);
}

template<typename T_VirtualDatum>
struct SetZeroFunctor
{
    template<typename T_OuterCoord, typename T_InnerCoord>
    void operator()(T_OuterCoord, T_InnerCoord)
    {
        vd(typename T_OuterCoord::template Cat<T_InnerCoord>()) = 0;
    }
    T_VirtualDatum vd;
};

TEST_CASE("iteration and access")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
        {
            SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
            llama::ForEach<Name, llama::DatumCoord<0, 0>>::apply(szf);
            llama::ForEach<Name, tag::Momentum>::apply(szf);
            view({x, y})
                = double(x + y) / double(userDomain[0] + userDomain[1]);
        }

    double sum = 0.0;
    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
            sum += view(x, y).access<1, 0>();
    CHECK(sum == 120.0);
}

TEST_CASE("Datum access")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
        {
            auto datum = view(x, y);
            datum.access<tag::Pos, tag::X>()
                += datum.access<llama::DatumCoord<1, 0>>();
            datum.access(tag::Pos(), tag::Y())
                += datum.access(llama::DatumCoord<1, 1>());
            datum(tag::Pos(), tag::Z()) += datum(llama::DatumCoord<2>());
            datum(tag::Pos()) += datum(tag::Momentum());
        }

    double sum = 0.0;
    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
            sum += view(x, y).access<1, 0>();

    CHECK(sum == 0.0);
}

TEST_CASE("AssignOneDatumIntoView")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    auto datum = llama::stackVirtualDatumAlloc<Name>();
    datum(tag::Pos{}, tag::X{}) = 14.0f;
    datum(tag::Pos{}, tag::Y{}) = 15.0f;
    datum(tag::Pos{}, tag::Z{}) = 16.0f;
    datum(tag::Momentum{}) = 0;
    datum(tag::Weight{}) = 500.0f;
    datum(tag::Options{}).access<0>() = true;
    datum(tag::Options{}).access<1>() = false;
    datum(tag::Options{}).access<2>() = true;
    datum(tag::Options{}).access<3>() = false;

    view({3, 4}) = datum;

    CHECK(datum(tag::Pos{}, tag::X{}) == 14.0f);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 15.0f);
    CHECK(datum(tag::Pos{}, tag::Z{}) == 16.0f);
    CHECK(datum(tag::Momentum{}, tag::Z{}) == 0);
    CHECK(datum(tag::Momentum{}, tag::X{}) == 0);
    CHECK(datum(tag::Weight{}) == 500.0f);
    CHECK(datum(tag::Options{}).access<0>() == true);
    CHECK(datum(tag::Options{}).access<1>() == false);
    CHECK(datum(tag::Options{}).access<2>() == true);
    CHECK(datum(tag::Options{}).access<3>() == false);
}

TEST_CASE("non-memory-owning view")
{
    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{256};

    using Mapping = llama::mapping::SoA<UserDomain, Name>;
    Mapping mapping{userDomain};

    std::vector<std::byte> storage(mapping.getBlobSize(0));
    auto view = llama::View<Mapping, std::byte *>{mapping, {storage.data()}};

    for(auto i = 0u; i < 256u; i++)
    {
        auto * x = (std::byte *)&view(i)(tag::Pos{}, tag::X{});
        auto * o3 = (std::byte *)&view(i)(tag::Options{}).access<3>();

        CHECK(&storage.front() <= x);
        CHECK(x <= &storage.back());
        CHECK(&storage.front() <= o3);
        CHECK(o3 <= &storage.back());
    }
}

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

TEST_CASE("demangleType")
{
    auto str = prettyPrintType(Particle());
#ifdef _WIN32
    boost::replace_all(str, "__int64", "long");
#endif
    CHECK(str == R"(boost::mp11::mp_list<
    boost::mp11::mp_list<
        tag::Pos,
        boost::mp11::mp_list<
            boost::mp11::mp_list<
                tag::X,
                double
            >,
            boost::mp11::mp_list<
                tag::Y,
                double
            >,
            boost::mp11::mp_list<
                tag::Z,
                double
            >
        >
    >,
    boost::mp11::mp_list<
        tag::Weight,
        float
    >,
    boost::mp11::mp_list<
        tag::Momentum,
        boost::mp11::mp_list<
            boost::mp11::mp_list<
                tag::X,
                double
            >,
            boost::mp11::mp_list<
                tag::Y,
                double
            >,
            boost::mp11::mp_list<
                tag::Z,
                double
            >
        >
    >,
    boost::mp11::mp_list<
        tag::Flags,
        boost::mp11::mp_list<
            boost::mp11::mp_list<
                std::integral_constant<
                    unsigned long,
                    0
                >,
                bool
            >,
            boost::mp11::mp_list<
                std::integral_constant<
                    unsigned long,
                    1
                >,
                bool
            >,
            boost::mp11::mp_list<
                std::integral_constant<
                    unsigned long,
                    2
                >,
                bool
            >,
            boost::mp11::mp_list<
                std::integral_constant<
                    unsigned long,
                    3
                >,
                bool
            >
        >
    >
>)");
}

TEST_CASE("address.AoS")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::AoS<UserDomain, Particle>{userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 111);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 951);
    }
}

TEST_CASE("address.AoS.fortran")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::
        AoS<UserDomain, Particle, llama::LinearizeUserDomainAdressLikeFortran>{
            userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 16);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 24);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 28);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 36);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 44);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 52);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 53);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 54);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 55);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 896);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 904);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 912);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 920);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 924);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 932);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 940);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 948);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 949);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 950);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 951);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 56);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 64);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 72);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 80);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 84);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 92);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 100);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 108);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 109);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 110);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 111);
    }
}

TEST_CASE("address.SoA")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::SoA<UserDomain, Particle>{userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14081);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14096);
    }
}

TEST_CASE("address.SoA.fortran")
{
    using UserDomain = llama::UserDomain<2>;
    auto userDomain = UserDomain{16, 16};
    auto mapping = llama::mapping::
        SoA<UserDomain, Particle, llama::LinearizeUserDomainAdressLikeFortran>{
            userDomain};

    {
        const auto coord = UserDomain{0, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 0);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2048);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4096);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6144);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7168);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9216);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11264);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13312);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13568);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13824);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14080);
    }

    {
        const auto coord = UserDomain{0, 1};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 128);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2176);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4224);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6208);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7296);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9344);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11392);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13328);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13584);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13840);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14096);
    }

    {
        const auto coord = UserDomain{1, 0};
        CHECK(mapping.getBlobNrAndOffset<0, 0>(coord).offset == 8);
        CHECK(mapping.getBlobNrAndOffset<0, 1>(coord).offset == 2056);
        CHECK(mapping.getBlobNrAndOffset<0, 2>(coord).offset == 4104);
        CHECK(mapping.getBlobNrAndOffset<1>(coord).offset == 6148);
        CHECK(mapping.getBlobNrAndOffset<2, 0>(coord).offset == 7176);
        CHECK(mapping.getBlobNrAndOffset<2, 1>(coord).offset == 9224);
        CHECK(mapping.getBlobNrAndOffset<2, 2>(coord).offset == 11272);
        CHECK(mapping.getBlobNrAndOffset<3, 0>(coord).offset == 13313);
        CHECK(mapping.getBlobNrAndOffset<3, 1>(coord).offset == 13569);
        CHECK(mapping.getBlobNrAndOffset<3, 2>(coord).offset == 13825);
        CHECK(mapping.getBlobNrAndOffset<3, 3>(coord).offset == 14081);
    }
}

TEST_CASE("sizeOf")
{
    CHECK(llama::sizeOf<Particle> == 56);
}

TEST_CASE("access")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    auto l = [](auto & view) {
        const UserDomain pos{0, 0};
        CHECK((view(pos) == view[pos]));
        CHECK((view(pos) == view[{0, 0}]));
        CHECK((view(pos) == view({0, 0})));

        const auto & x = view(pos).template access<0, 0>();
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

        // also test arrays
        const bool & o0
            = view(pos).template access<tag::Flags>().template access<0>();
        CHECK(
            &o0
            == &view(pos)
                    .template access<tag::Flags>()
                    .template access<llama::Index<0>>());
        CHECK(
            &o0
            == &view(pos).template access<tag::Flags>().access(
                llama::Index<0>{}));
        CHECK(
            &o0 == &view(pos).template access<tag::Flags>()(llama::Index<0>{}));
    };
    l(view);
    l(std::as_const(view));
}

TEST_CASE("offsetOf")
{
    static_assert(llama::offsetOf<Particle> == 0);
    static_assert(llama::offsetOf<Particle, 0> == 0);
    static_assert(llama::offsetOf<Particle, 0, 0> == 0);
    static_assert(llama::offsetOf<Particle, 0, 1> == 8);
    static_assert(llama::offsetOf<Particle, 0, 2> == 16);
    static_assert(llama::offsetOf<Particle, 1> == 24);
    static_assert(llama::offsetOf<Particle, 2> == 28);
    static_assert(llama::offsetOf<Particle, 2, 0> == 28);
    static_assert(llama::offsetOf<Particle, 2, 1> == 36);
    static_assert(llama::offsetOf<Particle, 2, 2> == 44);
    static_assert(llama::offsetOf<Particle, 3> == 52);
    static_assert(llama::offsetOf<Particle, 3, 0> == 52);
    static_assert(llama::offsetOf<Particle, 3, 1> == 53);
    static_assert(llama::offsetOf<Particle, 3, 2> == 54);
    static_assert(llama::offsetOf<Particle, 3, 3> == 55);
}

TEST_CASE("addresses")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    const UserDomain pos{0, 0};
    auto & x = view(pos).access<tag::Pos, tag::X>();
    auto & y = view(pos).access<tag::Pos, tag::Y>();
    auto & z = view(pos).access<tag::Pos, tag::Z>();
    auto & w = view(pos).access<tag::Weight>();
    auto & mx = view(pos).access<tag::Momentum, tag::X>();
    auto & my = view(pos).access<tag::Momentum, tag::Y>();
    auto & mz = view(pos).access<tag::Momentum, tag::Z>();
    auto & o0 = view(pos).access<tag::Flags>().access<0>();
    auto & o1 = view(pos).access<tag::Flags>().access<1>();
    auto & o2 = view(pos).access<tag::Flags>().access<2>();
    auto & o3 = view(pos).access<tag::Flags>().access<3>();

    CHECK((size_t)&y - (size_t)&x == 2048);
    CHECK((size_t)&z - (size_t)&x == 4096);
    CHECK((size_t)&mx - (size_t)&x == 7168);
    CHECK((size_t)&my - (size_t)&x == 9216);
    CHECK((size_t)&mz - (size_t)&x == 11264);
    CHECK((size_t)&w - (size_t)&x == 6144);
    CHECK((size_t)&o0 - (size_t)&x == 13312);
    CHECK((size_t)&o1 - (size_t)&x == 13568);
    CHECK((size_t)&o2 - (size_t)&x == 13824);
    CHECK((size_t)&o3 - (size_t)&x == 14080);
}

template<typename T_VirtualDatum>
struct SetZeroFunctor
{
    template<typename T_OuterCoord, typename T_InnerCoord>
    void operator()(T_OuterCoord, T_InnerCoord)
    {
        vd(llama::Cat<T_OuterCoord, T_InnerCoord>{}) = 0;
    }
    T_VirtualDatum vd;
};

TEST_CASE("iteration and access")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
        {
            SetZeroFunctor<decltype(view(x, y))> szf{view(x, y)};
            llama::forEach<Particle>(szf, llama::DatumCoord<0, 0>{});
            llama::forEach<Particle>(szf, tag::Momentum{});
            view({x, y})
                = double(x + y) / double(userDomain[0] + userDomain[1]);
        }

    double sum = 0.0;
    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
            sum += view(x, y).access<2, 0>();
    CHECK(sum == 120.0);
}

TEST_CASE("Datum access")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    zeroStorage(view);

    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
        {
            auto datum = view(x, y);
            datum.access<tag::Pos, tag::X>()
                += datum.access<llama::DatumCoord<2, 0>>();
            datum.access(tag::Pos(), tag::Y())
                += datum.access(llama::DatumCoord<2, 1>());
            datum(tag::Pos(), tag::Z()) += datum(llama::DatumCoord<1>());
            datum(tag::Pos()) += datum(tag::Momentum());
        }

    double sum = 0.0;
    for(size_t x = 0; x < userDomain[0]; ++x)
        for(size_t y = 0; y < userDomain[1]; ++y)
            sum += view(x, y).access<2, 0>();

    CHECK(sum == 0.0);
}

TEST_CASE("AssignOneDatumIntoView")
{
    using UserDomain = llama::UserDomain<2>;
    UserDomain userDomain{16, 16};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping{userDomain};
    auto view = allocView(mapping);

    auto datum = llama::stackVirtualDatumAlloc<Particle>();
    datum(tag::Pos{}, tag::X{}) = 14.0f;
    datum(tag::Pos{}, tag::Y{}) = 15.0f;
    datum(tag::Pos{}, tag::Z{}) = 16.0f;
    datum(tag::Momentum{}) = 0;
    datum(tag::Weight{}) = 500.0f;
    datum(tag::Flags{}).access<0>() = true;
    datum(tag::Flags{}).access<1>() = false;
    datum(tag::Flags{}).access<2>() = true;
    datum(tag::Flags{}).access<3>() = false;

    view({3, 4}) = datum;

    CHECK(datum(tag::Pos{}, tag::X{}) == 14.0f);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 15.0f);
    CHECK(datum(tag::Pos{}, tag::Z{}) == 16.0f);
    CHECK(datum(tag::Momentum{}, tag::X{}) == 0);
    CHECK(datum(tag::Momentum{}, tag::Y{}) == 0);
    CHECK(datum(tag::Momentum{}, tag::Z{}) == 0);
    CHECK(datum(tag::Weight{}) == 500.0f);
    CHECK(datum(tag::Flags{}).access<0>() == true);
    CHECK(datum(tag::Flags{}).access<1>() == false);
    CHECK(datum(tag::Flags{}).access<2>() == true);
    CHECK(datum(tag::Flags{}).access<3>() == false);
}

TEST_CASE("non-memory-owning view")
{
    using UserDomain = llama::UserDomain<1>;
    UserDomain userDomain{256};

    using Mapping = llama::mapping::SoA<UserDomain, Particle>;
    Mapping mapping{userDomain};

    std::vector<std::byte> storage(mapping.getBlobSize(0));
    auto view = llama::View<Mapping, std::byte *>{mapping, {storage.data()}};

    for(auto i = 0u; i < 256u; i++)
    {
        auto * x = (std::byte *)&view(i)(tag::Pos{}, tag::X{});
        auto * o3 = (std::byte *)&view(i)(tag::Flags{}).access<3>();

        CHECK(&storage.front() <= x);
        CHECK(x <= &storage.back());
        CHECK(&storage.front() <= o3);
        CHECK(o3 <= &storage.back());
    }
}

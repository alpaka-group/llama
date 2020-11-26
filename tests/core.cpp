#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Flags {};
    struct Weight {};
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Weight, float>,
    llama::DE<llama::NoName, int>,
    llama::DE<tag::Vel,llama::DS<
        llama::DE<tag::Z, double>,
        llama::DE<tag::X, double>
    >>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;

using Other = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::Z, float>,
        llama::DE<tag::Y, float>
    >>
>;
// clang-format on

TEST_CASE("prettyPrintType")
{
    auto str = prettyPrintType(Particle());
#ifdef _WIN32
    boost::replace_all(str, "__int64", "long");
#endif
    CHECK(str == R"(llama::DatumStruct<
    llama::DatumElement<
        tag::Pos,
        llama::DatumStruct<
            llama::DatumElement<
                tag::X,
                double
            >,
            llama::DatumElement<
                tag::Y,
                double
            >,
            llama::DatumElement<
                tag::Z,
                double
            >
        >
    >,
    llama::DatumElement<
        tag::Weight,
        float
    >,
    llama::DatumElement<
        llama::NoName,
        int
    >,
    llama::DatumElement<
        tag::Vel,
        llama::DatumStruct<
            llama::DatumElement<
                tag::Z,
                double
            >,
            llama::DatumElement<
                tag::X,
                double
            >
        >
    >,
    llama::DatumElement<
        tag::Flags,
        llama::DatumStruct<
            llama::DatumElement<
                std::integral_constant<
                    unsigned long,
                    0
                >,
                bool
            >,
            llama::DatumElement<
                std::integral_constant<
                    unsigned long,
                    1
                >,
                bool
            >,
            llama::DatumElement<
                std::integral_constant<
                    unsigned long,
                    2
                >,
                bool
            >,
            llama::DatumElement<
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

TEST_CASE("sizeOf")
{
    STATIC_REQUIRE(llama::sizeOf<Particle> == 52);
}
TEST_CASE("offsetOf")
{
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<0>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<0, 0>> == 0);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<0, 1>> == 8);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<0, 2>> == 16);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<1>> == 24);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<2>> == 28);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<3>> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<3, 0>> == 32);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<3, 1>> == 40);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<4>> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<4, 0>> == 48);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<4, 1>> == 49);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<4, 2>> == 50);
    STATIC_REQUIRE(llama::offsetOf<Particle, llama::DatumCoord<4, 3>> == 51);
}

TEST_CASE("GetCoordFromTags")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle                             >, llama::DatumCoord<    >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos                   >, llama::DatumCoord<0   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::X           >, llama::DatumCoord<0, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Y           >, llama::DatumCoord<0, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Pos, tag::Z           >, llama::DatumCoord<0, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Weight                >, llama::DatumCoord<1   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, llama::NoName              >, llama::DatumCoord<2   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::Z           >, llama::DatumCoord<3, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Vel, tag::X           >, llama::DatumCoord<3, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags                 >, llama::DatumCoord<4   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<0>>, llama::DatumCoord<4, 0>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<1>>, llama::DatumCoord<4, 1>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<2>>, llama::DatumCoord<4, 2>>);
    STATIC_REQUIRE(std::is_same_v<llama::GetCoordFromTags<Particle, tag::Flags, llama::Index<3>>, llama::DatumCoord<4, 3>>);
    // clang-format on
}

TEST_CASE("GetTags")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::DatumCoord<0, 0>>, boost::mp11::mp_list<llama::NoName, tag::Pos, tag::X >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::DatumCoord<0   >>, boost::mp11::mp_list<llama::NoName, tag::Pos         >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::DatumCoord<    >>, boost::mp11::mp_list<llama::NoName                   >>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTags<Particle, llama::DatumCoord<3, 1>>, boost::mp11::mp_list<llama::NoName, tag::Vel, tag::X >>);
    // clang-format on
}

TEST_CASE("GetTag")
{
    // clang-format off
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<0, 0>>, tag::X       >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<0   >>, tag::Pos     >);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<    >>, llama::NoName>);
    STATIC_REQUIRE(std::is_same_v<llama::GetTag<Particle, llama::DatumCoord<3, 1>>, tag::X       >);
    // clang-format on
}

TEST_CASE("hasSameTags")
{
    using PosDomain = llama::GetType<Particle, llama::GetCoordFromTags<Particle, tag::Pos>>;
    using VelDomain = llama::GetType<Particle, llama::GetCoordFromTags<Particle, tag::Vel>>;

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosDomain, // DD A
            llama::DatumCoord<0>, // Local A
            VelDomain, // DD B
            llama::DatumCoord<0> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            PosDomain, // DD A
            llama::DatumCoord<0>, // Local A
            VelDomain, // DD B
            llama::DatumCoord<1> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::DatumCoord<0, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<0, 0> // Local B
            > == false);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::DatumCoord<0, 2>, // Local A
            Other, // DD B
            llama::DatumCoord<0, 0> // Local B
            > == true);

    STATIC_REQUIRE(
        llama::hasSameTags<
            Particle, // DD A
            llama::DatumCoord<3, 0>, // Local A
            Other, // DD B
            llama::DatumCoord<0, 0> // Local B
            > == false);
}

TEST_CASE("FlattenDatumDomain")
{
    STATIC_REQUIRE(std::is_same_v<
                   llama::FlattenDatumDomain<Particle>,
                   boost::mp11::mp_list<double, double, double, float, int, double, double, bool, bool, bool, bool>>);
}

TEST_CASE("flatDatumCoord")
{
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<>> == 0);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<0>> == 0);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<0, 0>> == 0);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<0, 1>> == 1);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<0, 2>> == 2);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<1>> == 3);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<2>> == 4);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<3>> == 5);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<3, 0>> == 5);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<3, 1>> == 6);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<4>> == 7);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<4, 0>> == 7);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<4, 1>> == 8);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<4, 2>> == 9);
    STATIC_REQUIRE(llama::flatDatumCoord<Particle, llama::DatumCoord<4, 3>> == 10);
}

// clang-format off
namespace tag
{
    struct A1{};
    struct A2{};
    struct A3{};
}

using Arrays = llama::DS<
    llama::DE<tag::A1, llama::DA<int, 3>>,
    llama::DE<tag::A2, llama::DA<llama::DS<
        llama::DE<tag::X, float>
    >, 3>>,
    llama::DE<tag::A3, llama::DA<llama::DA<int, 2>, 2>>
>;
// clang-format on

TEST_CASE("arrays")
{
    auto v = llama::allocView(llama::mapping::AoS{llama::ArrayDomain{1}, Arrays{}});

    v(0u)(tag::A1{}, llama::Index<0>{});
    v(0u)(tag::A1{}, llama::Index<1>{});
    v(0u)(tag::A1{}, llama::Index<2>{});
    v(0u)(tag::A2{}, llama::Index<0>{}, tag::X{});
    v(0u)(tag::A2{}, llama::Index<1>{}, tag::X{});
    v(0u)(tag::A2{}, llama::Index<2>{}, tag::X{});
    v(0u)(tag::A3{}, llama::Index<0>{}, llama::Index<0>{});
    v(0u)(tag::A3{}, llama::Index<0>{}, llama::Index<1>{});
    v(0u)(tag::A3{}, llama::Index<1>{}, llama::Index<0>{});
    v(0u)(tag::A3{}, llama::Index<1>{}, llama::Index<1>{});
}

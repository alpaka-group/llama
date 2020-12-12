#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag {
    struct Pos {};
    struct Vel {};
    struct A {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Weight {};
    struct Part1 {};
    struct Part2 {};
}

using Name = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::A, int>,
        llama::DE<tag::Y, int>>>,
    llama::DE<tag::Vel, llama::DS<
        llama::DE<tag::X, int>,
        llama::DE<tag::Y, int>,
        llama::DE<tag::Z, int>>>,
    llama::DE<tag::Weight, int>>;
// clang-format on

TEST_CASE("VirtualDatum.operator=")
{
    llama::One<Name> datum;

    // scalar to multiple elements in virtual datum
    datum(tag::Pos{}) = 1;
    CHECK(datum(tag::Pos{}, tag::A{}) == 1);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::X{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
    CHECK(datum(tag::Weight{}) == 0);

    // scalar to multiple elements in virtual datum
    datum = 2;
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::X{}) == 2);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 2);
    CHECK(datum(tag::Weight{}) == 2);

    // smaller virtual datum to larger virtual datum
    datum(tag::Pos{}) = 3;
    datum(tag::Vel{}) = datum(tag::Pos{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 3);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::X{}) == 2);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 3); // only Y is propagated
    CHECK(datum(tag::Vel{}, tag::Z{}) == 2);
    CHECK(datum(tag::Weight{}) == 2);

    // larger virtual datum to smaller virtual datum
    datum(tag::Vel{}) = 4;
    datum(tag::Pos{}) = datum(tag::Vel{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 3);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 4); // only Y is propagated
    CHECK(datum(tag::Vel{}, tag::X{}) == 4);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 4);
    CHECK(datum(tag::Weight{}) == 2);

    // scalar virtual datum to larger virtual datum, full broadcast
    datum(tag::Weight{}) = 5;
    datum(tag::Vel{}) = datum(tag::Weight{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 3);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 4);
    CHECK(datum(tag::Vel{}, tag::X{}) == 5); // updated
    CHECK(datum(tag::Vel{}, tag::Y{}) == 5); // updated
    CHECK(datum(tag::Vel{}, tag::Z{}) == 5); // updated
    CHECK(datum(tag::Weight{}) == 5);
}

namespace
{
    auto allocVc()
    {
        llama::One<Name> datum;
        datum(tag::Pos{}, tag::A{}) = 1;
        datum(tag::Pos{}, tag::Y{}) = 2;
        datum(tag::Vel{}, tag::X{}) = 3;
        datum(tag::Vel{}, tag::Y{}) = 4;
        datum(tag::Vel{}, tag::Z{}) = 5;
        datum(tag::Weight{}) = 6;
        return datum;
    }
} // namespace

TEST_CASE("VirtualDatum.operator+=.scalar")
{
    {
        auto datum = allocVc();
        datum(tag::Pos{}) += 1;
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 6);
    }

    {
        auto datum = allocVc();
        datum += 1;
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 5);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 6);
        CHECK(datum(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualDatum.operator+=.VirtualDatum")
{
    {
        // smaller virtual datum to larger virtual datum
        auto datum = allocVc();
        datum(tag::Vel{}) += datum(tag::Pos{});
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 6);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 6);
    }

    {
        // larger virtual datum to smaller virtual datum
        auto datum = allocVc();
        datum(tag::Pos{}) += datum(tag::Vel{});
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 6);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 6);
    }

    {
        // scalar virtual datum to larger virtual datum, full broadcast
        auto datum = allocVc();
        datum(tag::Vel{}) += datum(tag::Weight{});
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 9);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 10);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 11);
        CHECK(datum(tag::Weight{}) == 6);
    }
}

TEST_CASE("VirtualDatum.operator+.scalar")
{
    {
        auto datum = allocVc();
        datum(tag::Pos{}) = datum(tag::Pos{}) + 1;
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 6);
    }
    {
        auto datum = allocVc();
        datum(tag::Pos{}) = 1 + datum(tag::Pos{});
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 6);
    }

    {
        auto datum = allocVc();
        datum = datum + 1;
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 5);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 6);
        CHECK(datum(tag::Weight{}) == 7);
    }
    {
        auto datum = allocVc();
        datum = 1 + datum;
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 5);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 6);
        CHECK(datum(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualDatum.operator+.VirtualDatum")
{
    {
        // smaller virtual datum to larger virtual datum
        auto datum = allocVc();
        datum(tag::Vel{}) = datum(tag::Vel{}) + datum(tag::Pos{});
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 6);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 6);
    }

    {
        // larger virtual datum to smaller virtual datum
        auto datum = allocVc();
        datum(tag::Pos{}) = datum(tag::Pos{}) + datum(tag::Vel{});
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 6);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 6);
    }

    {
        // scalar virtual datum to larger virtual datum, full broadcast
        auto datum = allocVc();
        datum(tag::Vel{}) = datum(tag::Vel{}) + datum(tag::Weight{});
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 9);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 10);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 11);
        CHECK(datum(tag::Weight{}) == 6);
    }
}

// clang-format off
using Name2 = llama::DS<
    llama::DE<tag::Part1, llama::DS<
        llama::DE<tag::Weight, int>,
        llama::DE<tag::Pos, llama::DS<
            llama::DE<tag::X, int>,
            llama::DE<tag::Y, int>,
            llama::DE<tag::Z, int>
        >>
    >>,
    llama::DE<tag::Part2, llama::DS<
        llama::DE<tag::Weight, int>,
        llama::DE<tag::Pos, llama::DS<
            llama::DE<tag::X, int>,
            llama::DE<tag::Y, int>,
            llama::DE<tag::A, int>
        >>,
        llama::DE<tag::Z, int>
    >>
>;
// clang-format on

TEST_CASE("VirtualDatum.operator=.propagation")
{
    llama::One<Name2> datum;

    datum(tag::Part1{}) = 1;
    datum(tag::Part2{}) = 2;
    CHECK(datum(tag::Part1{}, tag::Weight{}) == 1);
    CHECK(datum(tag::Part1{}, tag::Pos{}, tag::X{}) == 1);
    CHECK(datum(tag::Part1{}, tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Part1{}, tag::Pos{}, tag::Z{}) == 1);
    CHECK(datum(tag::Part2{}, tag::Weight{}) == 2);
    CHECK(datum(tag::Part2{}, tag::Pos{}, tag::X{}) == 2);
    CHECK(datum(tag::Part2{}, tag::Pos{}, tag::Y{}) == 2);
    CHECK(datum(tag::Part2{}, tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Part2{}, tag::Z{}) == 2);

    datum(tag::Part2{}) = datum(tag::Part1{});
    CHECK(datum(tag::Part1{}, tag::Weight{}) == 1);
    CHECK(datum(tag::Part1{}, tag::Pos{}, tag::X{}) == 1);
    CHECK(datum(tag::Part1{}, tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Part1{}, tag::Pos{}, tag::Z{}) == 1);
    CHECK(datum(tag::Part2{}, tag::Weight{}) == 1); // propagated
    CHECK(datum(tag::Part2{}, tag::Pos{}, tag::X{}) == 1); // propagated
    CHECK(datum(tag::Part2{}, tag::Pos{}, tag::Y{}) == 1); // propagated
    CHECK(datum(tag::Part2{}, tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Part2{}, tag::Z{}) == 2);
}

TEST_CASE("VirtualDatum.operator=.multiview")
{
    llama::One<Name> datum1;
    llama::One<Name2> datum2;

    datum2 = 1;
    datum1 = datum2;
    CHECK(datum1(tag::Pos{}, tag::A{}) == 0);
    CHECK(datum1(tag::Pos{}, tag::Y{}) == 0);
    CHECK(datum1(tag::Vel{}, tag::X{}) == 0);
    CHECK(datum1(tag::Vel{}, tag::Y{}) == 0);
    CHECK(datum1(tag::Vel{}, tag::Z{}) == 0);
    CHECK(datum1(tag::Weight{}) == 0);

    datum1 = datum2(tag::Part1{});
    CHECK(datum1(tag::Pos{}, tag::A{}) == 0);
    CHECK(datum1(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum1(tag::Vel{}, tag::X{}) == 0);
    CHECK(datum1(tag::Vel{}, tag::Y{}) == 0);
    CHECK(datum1(tag::Vel{}, tag::Z{}) == 0);
    CHECK(datum1(tag::Weight{}) == 1);
}

TEST_CASE("VirtualDatum.operator==")
{
    llama::One<Name> datum;

    datum = 1;

    CHECK((datum(tag::Pos{}, tag::Y{}) == datum(tag::Pos{}, tag::Y{})));
    CHECK((datum(tag::Pos{}) == datum(tag::Pos{})));
    CHECK((datum == datum));
    CHECK((datum(tag::Pos{}) == datum(tag::Vel{})));

    // scalar to multiple elements in virtual datum
    CHECK((datum(tag::Pos{}, tag::Y{}) == 1));
    CHECK((datum(tag::Pos{}) == 1));
    CHECK((datum == 1));

    datum(tag::Pos{}, tag::Y{}) = 2;

    CHECK((datum(tag::Pos{}, tag::Y{}) == 2));
    CHECK(!(datum(tag::Pos{}) == 1));
    CHECK(!(datum == 1));
    CHECK(!(datum(tag::Pos{}) == datum(tag::Vel{})));
}

TEST_CASE("VirtualDatum.operator<")
{
    llama::One<Name> datum;

    datum = 1;

    CHECK(!(datum(tag::Pos{}, tag::Y{}) < datum(tag::Pos{}, tag::Y{})));
    CHECK(!(datum(tag::Pos{}) < datum(tag::Pos{})));
    CHECK(!(datum < datum));
    CHECK(!(datum(tag::Pos{}) < datum(tag::Vel{})));

    // scalar to multiple elements in virtual datum
    CHECK((datum(tag::Pos{}, tag::Y{}) < 2));
    CHECK((datum(tag::Pos{}) < 2));
    CHECK((datum < 2));
    CHECK((2 > datum(tag::Pos{}, tag::Y{})));
    CHECK((2 > datum(tag::Pos{})));
    CHECK((2 > datum));

    CHECK(!(datum(tag::Pos{}, tag::Y{}) < 1));
    CHECK(!(datum(tag::Pos{}) < 1));
    CHECK(!(datum < 1));
    CHECK(!(1 > datum(tag::Pos{}, tag::Y{})));
    CHECK(!(1 > datum(tag::Pos{})));
    CHECK(!(1 > datum));

    datum(tag::Pos{}, tag::Y{}) = 2;

    CHECK((datum(tag::Pos{}, tag::Y{}) < 3));
    CHECK(!(datum(tag::Pos{}) < 2));
    CHECK(!(datum < 2));
    CHECK((3 > datum(tag::Pos{}, tag::Y{})));
    CHECK(!(2 > datum(tag::Pos{})));
    CHECK(!(2 > datum));
    CHECK(!(datum(tag::Pos{}) < datum(tag::Vel{})));
}

TEST_CASE("VirtualDatum.asTuple.types")
{
    {
        llama::One<Name> datum;

        std::tuple<int&, int&> pos = datum(tag::Pos{}).asTuple();
        std::tuple<int&, int&, int&> vel = datum(tag::Vel{}).asTuple();
        std::tuple<std::tuple<int&, int&>, std::tuple<int&, int&, int&>, int&> name = datum.asTuple();
    }
    {
        const llama::One<Name> datum;

        std::tuple<const int&, const int&> pos = datum(tag::Pos{}).asTuple();
        std::tuple<const int&, const int&, const int&> vel = datum(tag::Vel{}).asTuple();
        std::tuple<std::tuple<const int&, const int&>, std::tuple<const int&, const int&, const int&>, const int&> name
            = datum.asTuple();
    }
}

TEST_CASE("VirtualDatum.asTuple.assign")
{
    llama::One<Name> datum;

    datum(tag::Pos{}).asTuple() = std::tuple{1, 1};
    CHECK(datum(tag::Pos{}, tag::A{}) == 1);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::X{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
    CHECK(datum(tag::Weight{}) == 0);

    datum(tag::Vel{}).asTuple() = std::tuple{2, 2, 2};
    CHECK(datum(tag::Pos{}, tag::A{}) == 1);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::X{}) == 2);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 2);
    CHECK(datum(tag::Weight{}) == 0);

    datum.asTuple() = std::tuple{std::tuple{3, 3}, std::tuple{3, 3, 3}, 3};
    CHECK(datum(tag::Pos{}, tag::A{}) == 3);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::X{}) == 3);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 3);
    CHECK(datum(tag::Weight{}) == 3);
}

TEST_CASE("VirtualDatum.asTuple.structuredBindings")
{
    llama::One<Name> datum;

    {
        auto [a, y] = datum(tag::Pos{}).asTuple();
        a = 1;
        y = 2;
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 0);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
        CHECK(datum(tag::Weight{}) == 0);
    }

    {
        auto [x, y, z] = datum(tag::Vel{}).asTuple();
        x = 3;
        y = 4;
        z = 5;
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 0);
    }

    {
        auto [pos, vel, w] = datum.asTuple();
        auto [a, y1] = pos;
        auto [x, y2, z] = vel;
        a = 10;
        y1 = 20;
        x = 30;
        y2 = 40;
        z = 50;
        w = 60;
        CHECK(datum(tag::Pos{}, tag::A{}) == 10);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 20);
        CHECK(datum(tag::Vel{}, tag::X{}) == 30);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 40);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 50);
        CHECK(datum(tag::Weight{}) == 60);
    }
}


TEST_CASE("VirtualDatum.asFlatTuple.types")
{
    {
        llama::One<Name> datum;

        std::tuple<int&, int&> pos = datum(tag::Pos{}).asFlatTuple();
        std::tuple<int&, int&, int&> vel = datum(tag::Vel{}).asFlatTuple();
        std::tuple<int&, int&, int&, int&, int&, int&> name = datum.asFlatTuple();
    }
    {
        const llama::One<Name> datum;

        std::tuple<const int&, const int&> pos = datum(tag::Pos{}).asFlatTuple();
        std::tuple<const int&, const int&, const int&> vel = datum(tag::Vel{}).asFlatTuple();
        std::tuple<const int&, const int&, const int&, const int&, const int&, const int&> name = datum.asFlatTuple();
    }
}

TEST_CASE("VirtualDatum.asFlatTuple.assign")
{
    llama::One<Name> datum;

    datum(tag::Pos{}).asFlatTuple() = std::tuple{1, 1};
    CHECK(datum(tag::Pos{}, tag::A{}) == 1);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::X{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
    CHECK(datum(tag::Weight{}) == 0);

    datum(tag::Vel{}).asFlatTuple() = std::tuple{2, 2, 2};
    CHECK(datum(tag::Pos{}, tag::A{}) == 1);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::X{}) == 2);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 2);
    CHECK(datum(tag::Weight{}) == 0);

    datum.asFlatTuple() = std::tuple{3, 3, 3, 3, 3, 3};
    CHECK(datum(tag::Pos{}, tag::A{}) == 3);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::X{}) == 3);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 3);
    CHECK(datum(tag::Weight{}) == 3);
}

TEST_CASE("VirtualDatum.asFlatTuple.structuredBindings")
{
    llama::One<Name> datum;

    {
        auto [a, y] = datum(tag::Pos{}).asFlatTuple();
        a = 1;
        y = 2;
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 0);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
        CHECK(datum(tag::Weight{}) == 0);
    }

    {
        auto [x, y, z] = datum(tag::Vel{}).asFlatTuple();
        x = 3;
        y = 4;
        z = 5;
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 0);
    }

    {
        auto [a, y1, x, y2, z, w] = datum.asFlatTuple();
        a = 10;
        y1 = 20;
        x = 30;
        y2 = 40;
        z = 50;
        w = 60;
        CHECK(datum(tag::Pos{}, tag::A{}) == 10);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 20);
        CHECK(datum(tag::Vel{}, tag::X{}) == 30);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 40);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 50);
        CHECK(datum(tag::Weight{}) == 60);
    }
}

template <typename T>
struct S;

TEST_CASE("VirtualDatum.structuredBindings")
{
    llama::One<Name> datum;

    {
        auto&& [a, y] = datum(tag::Pos{});
        a = 1;
        y = 2;
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 0);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
        CHECK(datum(tag::Weight{}) == 0);
    }

    {
        auto&& [x, y, z] = datum(tag::Vel{});
        x = 3;
        y = 4;
        z = 5;
        CHECK(datum(tag::Pos{}, tag::A{}) == 1);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
        CHECK(datum(tag::Vel{}, tag::X{}) == 3);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 5);
        CHECK(datum(tag::Weight{}) == 0);
    }

    {
        auto&& [pos, vel, w] = datum;
        auto&& [a, y1] = pos;
        auto&& [x, y2, z] = vel;
        a = 10;
        y1 = 20;
        x = 30;
        y2 = 40;
        z = 50;
        w = 60;
        CHECK(datum(tag::Pos{}, tag::A{}) == 10);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 20);
        CHECK(datum(tag::Vel{}, tag::X{}) == 30);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 40);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 50);
        CHECK(datum(tag::Weight{}) == 60);
    }
}

namespace
{
    template <typename T>
    struct MyPos
    {
        T a;
        T y;
    };

    template <typename T>
    struct MyVel
    {
        T x;
        T y;
        T z;
    };

    template <typename T>
    struct MyDatum
    {
        MyPos<T> pos;
        MyVel<T> vel;
        T weight;
    };

    template <std::size_t I, typename T>
    auto get(const MyPos<T>& p)
    {
        if constexpr (I == 0)
            return p.a;
        if constexpr (I == 1)
            return p.y;
    }

    template <std::size_t I, typename T>
    auto get(const MyVel<T>& p)
    {
        if constexpr (I == 0)
            return p.x;
        if constexpr (I == 1)
            return p.y;
        if constexpr (I == 2)
            return p.z;
    }

    template <std::size_t I, typename T>
    auto get(const MyDatum<T>& p)
    {
        if constexpr (I == 0)
            return p.pos;
        if constexpr (I == 1)
            return p.vel;
        if constexpr (I == 2)
            return p.weight;
    }
} // namespace

template <typename T>
struct std::tuple_size<MyPos<T>>
{
    static constexpr std::size_t value = 2;
};

template <typename T>
struct std::tuple_size<MyVel<T>>
{
    static constexpr std::size_t value = 3;
};

template <typename T>
struct std::tuple_size<MyDatum<T>>
{
    static constexpr std::size_t value = 3;
};

TEST_CASE("VirtualDatum.load.value")
{
    llama::One<Name> datum;
    datum = 1;

    {
        MyPos<int> pos = datum(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        MyPos<int> pos = std::as_const(datum)(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }

    {
        MyDatum<int> d = datum.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
    {
        MyDatum<int> d = std::as_const(datum).load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
}

TEST_CASE("VirtualDatum.load.ref")
{
    llama::One<Name> datum;

    datum = 1;
    {
        MyPos<int&> pos = datum(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);

        pos.a = 2;
        pos.y = 3;
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 1);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 1);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
        CHECK(datum(tag::Weight{}) == 1);
    }

    datum = 1;
    {
        MyDatum<int&> d = datum.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);

        d.pos.a = 2;
        d.pos.y = 3;
        d.vel.x = 4;
        d.vel.y = 5;
        d.vel.z = 6;
        d.weight = 7;
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 5);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 6);
        CHECK(datum(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualDatum.load.constref")
{
    llama::One<Name> datum;
    datum = 1;

    {
        MyPos<const int&> pos = datum(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        MyPos<const int&> pos = std::as_const(datum)(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        MyDatum<const int&> d = datum.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
    }
    {
        MyDatum<const int&> d = std::as_const(datum).load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
    }
}

TEST_CASE("VirtualDatum.store")
{
    llama::One<Name> datum;

    datum = 1;
    {
        MyPos<int> pos{2, 3};
        datum(tag::Pos{}).store(pos);
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 1);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 1);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
        CHECK(datum(tag::Weight{}) == 1);
    }

    datum = 1;
    {
        MyDatum<int> d{{2, 3}, {4, 5, 6}, 7};
        datum.store(d);
        CHECK(datum(tag::Pos{}, tag::A{}) == 2);
        CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
        CHECK(datum(tag::Vel{}, tag::X{}) == 4);
        CHECK(datum(tag::Vel{}, tag::Y{}) == 5);
        CHECK(datum(tag::Vel{}, tag::Z{}) == 6);
        CHECK(datum(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualDatum.loadAs.value")
{
    llama::One<Name> datum;
    datum = 1;

    {
        auto pos = datum(tag::Pos{}).loadAs<MyPos<int>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        auto pos = std::as_const(datum)(tag::Pos{}).loadAs<MyPos<int>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
}

TEST_CASE("VirtualDatum.loadAs.ref")
{
    llama::One<Name> datum;
    datum = 1;

    auto pos = datum(tag::Pos{}).loadAs<MyPos<int&>>();
    CHECK(pos.a == 1);
    CHECK(pos.y == 1);

    pos.a = 2;
    pos.y = 3;
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::X{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
    CHECK(datum(tag::Weight{}) == 1);
}

TEST_CASE("VirtualDatum.loadAs.constref")
{
    llama::One<Name> datum;
    datum = 1;

    {
        auto pos = datum(tag::Pos{}).loadAs<MyPos<const int&>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        auto pos = std::as_const(datum)(tag::Pos{}).loadAs<MyPos<const int&>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
}

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

TEST_CASE("operator=")
{
    auto datum = llama::stackVirtualDatumAlloc<Name>();

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

TEST_CASE("operator+=")
{
    auto datum = llama::stackVirtualDatumAlloc<Name>();

    // scalar to multiple elements in virtual datum
    datum(tag::Pos{}) += 1;
    CHECK(datum(tag::Pos{}, tag::A{}) == 1);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::X{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
    CHECK(datum(tag::Weight{}) == 0);

    // scalar to multiple elements in virtual datum
    datum += 1;
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::X{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
    CHECK(datum(tag::Weight{}) == 1);

    // smaller virtual datum to larger virtual datum
    datum(tag::Vel{}) += datum(tag::Pos{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::X{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 3); // only Y is propagated
    CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
    CHECK(datum(tag::Weight{}) == 1);

    // larger virtual datum to smaller virtual datum
    datum(tag::Pos{}) += datum(tag::Vel{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 5); // only Y is propagated
    CHECK(datum(tag::Vel{}, tag::X{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
    CHECK(datum(tag::Weight{}) == 1);

    // scalar virtual datum to larger virtual datum, full broadcast
    datum(tag::Vel{}) += datum(tag::Weight{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 5);
    CHECK(datum(tag::Vel{}, tag::X{}) == 2); // updated
    CHECK(datum(tag::Vel{}, tag::Y{}) == 4); // updated
    CHECK(datum(tag::Vel{}, tag::Z{}) == 2); // updated
    CHECK(datum(tag::Weight{}) == 1);
}

TEST_CASE("operator+")
{
    auto datum = llama::stackVirtualDatumAlloc<Name>();

    // scalar to multiple elements in virtual datum
    datum(tag::Pos{}) = datum(tag::Pos{}) + 1;
    CHECK(datum(tag::Pos{}, tag::A{}) == 1);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::X{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 0);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 0);
    CHECK(datum(tag::Weight{}) == 0);

    // scalar to multiple elements in virtual datum
    datum = datum + 1;
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::X{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
    CHECK(datum(tag::Weight{}) == 1);

    // smaller virtual datum to larger virtual datum
    datum(tag::Vel{}) = datum(tag::Vel{}) + datum(tag::Pos{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 2);
    CHECK(datum(tag::Vel{}, tag::X{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 3); // only Y is propagated
    CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
    CHECK(datum(tag::Weight{}) == 1);

    // larger virtual datum to smaller virtual datum
    datum(tag::Pos{}) = datum(tag::Pos{}) + datum(tag::Vel{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 5); // only Y is propagated
    CHECK(datum(tag::Vel{}, tag::X{}) == 1);
    CHECK(datum(tag::Vel{}, tag::Y{}) == 3);
    CHECK(datum(tag::Vel{}, tag::Z{}) == 1);
    CHECK(datum(tag::Weight{}) == 1);

    // scalar virtual datum to larger virtual datum, full broadcast
    datum(tag::Vel{}) = datum(tag::Vel{}) + datum(tag::Weight{});
    CHECK(datum(tag::Pos{}, tag::A{}) == 2);
    CHECK(datum(tag::Pos{}, tag::Y{}) == 5);
    CHECK(datum(tag::Vel{}, tag::X{}) == 2); // updated
    CHECK(datum(tag::Vel{}, tag::Y{}) == 4); // updated
    CHECK(datum(tag::Vel{}, tag::Z{}) == 2); // updated
    CHECK(datum(tag::Weight{}) == 1);
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

TEST_CASE("operator=propagation")
{
    auto datum = llama::stackVirtualDatumAlloc<Name2>();

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

TEST_CASE("operator=multiview")
{
    auto datum1 = llama::stackVirtualDatumAlloc<Name>();
    auto datum2 = llama::stackVirtualDatumAlloc<Name2>();

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

TEST_CASE("operator==")
{
    auto datum = llama::stackVirtualDatumAlloc<Name>();

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

TEST_CASE("operator<")
{
    auto datum = llama::stackVirtualDatumAlloc<Name>();

    datum = 1;

    CHECK(!(datum(tag::Pos{}, tag::Y{}) < datum(tag::Pos{}, tag::Y{})));
    CHECK(!(datum(tag::Pos{}) < datum(tag::Pos{})));
    CHECK(!(datum < datum));
    CHECK(!(datum(tag::Pos{}) < datum(tag::Vel{})));

    // scalar to multiple elements in virtual datum
    CHECK((datum(tag::Pos{}, tag::Y{}) < 2));
    CHECK((datum(tag::Pos{}) < 2));
    CHECK((datum < 2));
    CHECK(!(datum(tag::Pos{}, tag::Y{}) < 1));
    CHECK(!(datum(tag::Pos{}) < 1));
    CHECK(!(datum < 1));

    datum(tag::Pos{}, tag::Y{}) = 2;

    CHECK((datum(tag::Pos{}, tag::Y{}) < 3));
    CHECK(!(datum(tag::Pos{}) < 2));
    CHECK(!(datum < 2));
    CHECK(!(datum(tag::Pos{}) < datum(tag::Vel{})));
}

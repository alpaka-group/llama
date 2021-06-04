#include "common.h"

#include <algorithm>
#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <sstream>
#include <vector>

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
} // namespace tag

using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::A, int>,
        llama::Field<tag::Y, int>>>,
    llama::Field<tag::Vel, llama::Record<
        llama::Field<tag::X, int>,
        llama::Field<tag::Y, int>,
        llama::Field<tag::Z, int>>>,
    llama::Field<tag::Weight, int>>;
// clang-format on

TEST_CASE("VirtualRecord.operator=")
{
    llama::One<Particle> record;

    // scalar to multiple elements in virtual record
    record(tag::Pos{}) = 1;
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Weight{}) == 0);

    // scalar to multiple elements in virtual record
    record = 2;
    CHECK(record(tag::Pos{}, tag::A{}) == 2);
    CHECK(record(tag::Pos{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Weight{}) == 2);

    // smaller virtual record to larger virtual record
    record(tag::Pos{}) = 3;
    record(tag::Vel{}) = record(tag::Pos{});
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 3); // only Y is propagated
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Weight{}) == 2);

    // larger virtual record to smaller virtual record
    record(tag::Vel{}) = 4;
    record(tag::Pos{}) = record(tag::Vel{});
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 4); // only Y is propagated
    CHECK(record(tag::Vel{}, tag::X{}) == 4);
    CHECK(record(tag::Vel{}, tag::Y{}) == 4);
    CHECK(record(tag::Vel{}, tag::Z{}) == 4);
    CHECK(record(tag::Weight{}) == 2);

    // scalar virtual record to larger virtual record, full broadcast
    record(tag::Weight{}) = 5;
    record(tag::Vel{}) = record(tag::Weight{});
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 4);
    CHECK(record(tag::Vel{}, tag::X{}) == 5); // updated
    CHECK(record(tag::Vel{}, tag::Y{}) == 5); // updated
    CHECK(record(tag::Vel{}, tag::Z{}) == 5); // updated
    CHECK(record(tag::Weight{}) == 5);
}

namespace
{
    auto oneParticle()
    {
        llama::One<Particle> record;
        record(tag::Pos{}, tag::A{}) = 1;
        record(tag::Pos{}, tag::Y{}) = 2;
        record(tag::Vel{}, tag::X{}) = 3;
        record(tag::Vel{}, tag::Y{}) = 4;
        record(tag::Vel{}, tag::Z{}) = 5;
        record(tag::Weight{}) = 6;
        return record;
    }
} // namespace

TEST_CASE("VirtualRecord.operator+=.scalar")
{
    {
        auto record = oneParticle();
        record(tag::Pos{}) += 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 6);
    }

    {
        auto record = oneParticle();
        record += 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualRecord.operator+=.VirtualRecord")
{
    {
        // smaller virtual record to larger virtual record
        auto record = oneParticle();
        record(tag::Vel{}) += record(tag::Pos{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 6);
    }

    {
        // larger virtual record to smaller virtual record
        auto record = oneParticle();
        record(tag::Pos{}) += record(tag::Vel{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 6);
    }

    {
        // scalar virtual record to larger virtual record, full broadcast
        auto record = oneParticle();
        record(tag::Vel{}) += record(tag::Weight{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 9);
        CHECK(record(tag::Vel{}, tag::Y{}) == 10);
        CHECK(record(tag::Vel{}, tag::Z{}) == 11);
        CHECK(record(tag::Weight{}) == 6);
    }
}

TEST_CASE("VirtualRecord.operator+.scalar")
{
    {
        auto record = oneParticle();
        record(tag::Pos{}) = record(tag::Pos{}) + 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 6);
    }
    {
        auto record = oneParticle();
        record(tag::Pos{}) = 1 + record(tag::Pos{});
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 6);
    }

    {
        auto record = oneParticle();
        record = record + 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Weight{}) == 7);
    }
    {
        auto record = oneParticle();
        record = 1 + record;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualRecord.operator+.VirtualRecord")
{
    {
        // smaller virtual record to larger virtual record
        auto record = oneParticle();
        record(tag::Vel{}) = record(tag::Vel{}) + record(tag::Pos{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 6);
    }

    {
        // larger virtual record to smaller virtual record
        auto record = oneParticle();
        record(tag::Pos{}) = record(tag::Pos{}) + record(tag::Vel{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 6);
    }

    {
        // scalar virtual record to larger virtual record, full broadcast
        auto record = oneParticle();
        record(tag::Vel{}) = record(tag::Vel{}) + record(tag::Weight{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 9);
        CHECK(record(tag::Vel{}, tag::Y{}) == 10);
        CHECK(record(tag::Vel{}, tag::Z{}) == 11);
        CHECK(record(tag::Weight{}) == 6);
    }
}

// clang-format off
using Name2 = llama::Record<
    llama::Field<tag::Part1, llama::Record<
        llama::Field<tag::Weight, int>,
        llama::Field<tag::Pos, llama::Record<
            llama::Field<tag::X, int>,
            llama::Field<tag::Y, int>,
            llama::Field<tag::Z, int>
        >>
    >>,
    llama::Field<tag::Part2, llama::Record<
        llama::Field<tag::Weight, int>,
        llama::Field<tag::Pos, llama::Record<
            llama::Field<tag::X, int>,
            llama::Field<tag::Y, int>,
            llama::Field<tag::A, int>
        >>,
        llama::Field<tag::Z, int>
    >>
>;
// clang-format on

TEST_CASE("VirtualRecord.operator=.propagation")
{
    llama::One<Name2> record;

    record(tag::Part1{}) = 1;
    record(tag::Part2{}) = 2;
    CHECK(record(tag::Part1{}, tag::Weight{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::X{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Z{}) == 1);
    CHECK(record(tag::Part2{}, tag::Weight{}) == 2);
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::X{}) == 2);
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::Y{}) == 2);
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::A{}) == 2);
    CHECK(record(tag::Part2{}, tag::Z{}) == 2);

    record(tag::Part2{}) = record(tag::Part1{});
    CHECK(record(tag::Part1{}, tag::Weight{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::X{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Z{}) == 1);
    CHECK(record(tag::Part2{}, tag::Weight{}) == 1); // propagated
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::X{}) == 1); // propagated
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::Y{}) == 1); // propagated
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::A{}) == 2);
    CHECK(record(tag::Part2{}, tag::Z{}) == 2);
}

TEST_CASE("VirtualRecord.operator=.multiview")
{
    llama::One<Particle> record1;
    llama::One<Name2> record2;

    record2 = 1;
    record1 = record2;
    CHECK(record1(tag::Pos{}, tag::A{}) == 0);
    CHECK(record1(tag::Pos{}, tag::Y{}) == 0);
    CHECK(record1(tag::Vel{}, tag::X{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record1(tag::Weight{}) == 0);

    record1 = record2(tag::Part1{});
    CHECK(record1(tag::Pos{}, tag::A{}) == 0);
    CHECK(record1(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record1(tag::Vel{}, tag::X{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record1(tag::Weight{}) == 1);
}

TEST_CASE("VirtualRecord.operator==")
{
    llama::One<Particle> record;

    record = 1;

    CHECK((record(tag::Pos{}, tag::Y{}) == record(tag::Pos{}, tag::Y{})));
    CHECK((record(tag::Pos{}) == record(tag::Pos{})));
    CHECK((record == record));
    CHECK((record(tag::Pos{}) == record(tag::Vel{})));

    // scalar to multiple elements in virtual record
    CHECK((record(tag::Pos{}, tag::Y{}) == 1));
    CHECK((record(tag::Pos{}) == 1));
    CHECK((record == 1));

    record(tag::Pos{}, tag::Y{}) = 2;

    CHECK((record(tag::Pos{}, tag::Y{}) == 2));
    CHECK(!(record(tag::Pos{}) == 1));
    CHECK(!(record == 1));
    CHECK(!(record(tag::Pos{}) == record(tag::Vel{})));
}

TEST_CASE("VirtualRecord.operator<")
{
    llama::One<Particle> record;

    record = 1;

    CHECK(!(record(tag::Pos{}, tag::Y{}) < record(tag::Pos{}, tag::Y{})));
    CHECK(!(record(tag::Pos{}) < record(tag::Pos{})));
    CHECK(!(record < record));
    CHECK(!(record(tag::Pos{}) < record(tag::Vel{})));

    // scalar to multiple elements in virtual record
    CHECK((record(tag::Pos{}, tag::Y{}) < 2));
    CHECK((record(tag::Pos{}) < 2));
    CHECK((record < 2));
    CHECK((2 > record(tag::Pos{}, tag::Y{})));
    CHECK((2 > record(tag::Pos{})));
    CHECK((2 > record));

    CHECK(!(record(tag::Pos{}, tag::Y{}) < 1));
    CHECK(!(record(tag::Pos{}) < 1));
    CHECK(!(record < 1));
    CHECK(!(1 > record(tag::Pos{}, tag::Y{})));
    CHECK(!(1 > record(tag::Pos{})));
    CHECK(!(1 > record));

    record(tag::Pos{}, tag::Y{}) = 2;

    CHECK((record(tag::Pos{}, tag::Y{}) < 3));
    CHECK(!(record(tag::Pos{}) < 2));
    CHECK(!(record < 2));
    CHECK((3 > record(tag::Pos{}, tag::Y{})));
    CHECK(!(2 > record(tag::Pos{})));
    CHECK(!(2 > record));
    CHECK(!(record(tag::Pos{}) < record(tag::Vel{})));
}

TEST_CASE("VirtualRecord.asTuple.types")
{
    {
        llama::One<Particle> record;

        std::tuple<int&, int&> pos = record(tag::Pos{}).asTuple();
        std::tuple<int&, int&, int&> vel = record(tag::Vel{}).asTuple();
        std::tuple<std::tuple<int&, int&>, std::tuple<int&, int&, int&>, int&> name = record.asTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
    {
        const llama::One<Particle> record;

        std::tuple<const int&, const int&> pos = record(tag::Pos{}).asTuple();
        std::tuple<const int&, const int&, const int&> vel = record(tag::Vel{}).asTuple();
        std::tuple<std::tuple<const int&, const int&>, std::tuple<const int&, const int&, const int&>, const int&> name
            = record.asTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
}

TEST_CASE("VirtualRecord.asTuple.assign")
{
    llama::One<Particle> record;

    record(tag::Pos{}).asTuple() = std::tuple{1, 1};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Weight{}) == 0);

    record(tag::Vel{}).asTuple() = std::tuple{2, 2, 2};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Weight{}) == 0);

    record.asTuple() = std::tuple{std::tuple{3, 3}, std::tuple{3, 3, 3}, 3};
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::X{}) == 3);
    CHECK(record(tag::Vel{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::Z{}) == 3);
    CHECK(record(tag::Weight{}) == 3);
}

TEST_CASE("VirtualRecord.asTuple.structuredBindings")
{
    llama::One<Particle> record;

    {
        auto [a, y] = record(tag::Pos{}).asTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        a = 1;
        y = 2;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 0);
        CHECK(record(tag::Vel{}, tag::Y{}) == 0);
        CHECK(record(tag::Vel{}, tag::Z{}) == 0);
        CHECK(record(tag::Weight{}) == 0);
    }

    {
        auto [x, y, z] = record(tag::Vel{}).asTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        x = 3;
        y = 4;
        z = 5;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 0);
    }

    {
        auto [pos, vel, w] = record.asTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        auto [a, y1] = pos; // NOLINT (clang-analyzer-deadcode.DeadStores)
        auto [x, y2, z] = vel; // NOLINT (clang-analyzer-deadcode.DeadStores)
        a = 10;
        y1 = 20;
        x = 30;
        y2 = 40;
        z = 50;
        w = 60;
        CHECK(record(tag::Pos{}, tag::A{}) == 10);
        CHECK(record(tag::Pos{}, tag::Y{}) == 20);
        CHECK(record(tag::Vel{}, tag::X{}) == 30);
        CHECK(record(tag::Vel{}, tag::Y{}) == 40);
        CHECK(record(tag::Vel{}, tag::Z{}) == 50);
        CHECK(record(tag::Weight{}) == 60);
    }
}


TEST_CASE("VirtualRecord.asFlatTuple.types")
{
    {
        llama::One<Particle> record;

        std::tuple<int&, int&> pos = record(tag::Pos{}).asFlatTuple();
        std::tuple<int&, int&, int&> vel = record(tag::Vel{}).asFlatTuple();
        std::tuple<int&, int&, int&, int&, int&, int&> name = record.asFlatTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
    {
        const llama::One<Particle> record;

        std::tuple<const int&, const int&> pos = record(tag::Pos{}).asFlatTuple();
        std::tuple<const int&, const int&, const int&> vel = record(tag::Vel{}).asFlatTuple();
        std::tuple<const int&, const int&, const int&, const int&, const int&, const int&> name = record.asFlatTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
}

TEST_CASE("VirtualRecord.asFlatTuple.assign")
{
    llama::One<Particle> record;

    record(tag::Pos{}).asFlatTuple() = std::tuple{1, 1};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Weight{}) == 0);

    record(tag::Vel{}).asFlatTuple() = std::tuple{2, 2, 2};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Weight{}) == 0);

    record.asFlatTuple() = std::tuple{3, 3, 3, 3, 3, 3};
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::X{}) == 3);
    CHECK(record(tag::Vel{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::Z{}) == 3);
    CHECK(record(tag::Weight{}) == 3);
}

TEST_CASE("VirtualRecord.asFlatTuple.structuredBindings")
{
    llama::One<Particle> record;

    {
        auto [a, y] = record(tag::Pos{}).asFlatTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        a = 1;
        y = 2;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 0);
        CHECK(record(tag::Vel{}, tag::Y{}) == 0);
        CHECK(record(tag::Vel{}, tag::Z{}) == 0);
        CHECK(record(tag::Weight{}) == 0);
    }

    {
        auto [x, y, z] = record(tag::Vel{}).asFlatTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        x = 3;
        y = 4;
        z = 5;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 0);
    }

    {
        auto [a, y1, x, y2, z, w] = record.asFlatTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        a = 10;
        y1 = 20;
        x = 30;
        y2 = 40;
        z = 50;
        w = 60;
        CHECK(record(tag::Pos{}, tag::A{}) == 10);
        CHECK(record(tag::Pos{}, tag::Y{}) == 20);
        CHECK(record(tag::Vel{}, tag::X{}) == 30);
        CHECK(record(tag::Vel{}, tag::Y{}) == 40);
        CHECK(record(tag::Vel{}, tag::Z{}) == 50);
        CHECK(record(tag::Weight{}) == 60);
    }
}

template <typename T>
struct S;

TEST_CASE("VirtualRecord.structuredBindings")
{
    llama::One<Particle> record;

    {
        auto&& [a, y] = record(tag::Pos{});
        a = 1;
        y = 2;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 0);
        CHECK(record(tag::Vel{}, tag::Y{}) == 0);
        CHECK(record(tag::Vel{}, tag::Z{}) == 0);
        CHECK(record(tag::Weight{}) == 0);
    }

    {
        auto&& [x, y, z] = record(tag::Vel{});
        x = 3;
        y = 4;
        z = 5;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Weight{}) == 0);
    }

    {
        auto&& [pos, vel, w] = record;
        auto&& [a, y1] = pos;
        auto&& [x, y2, z] = vel;
        a = 10;
        y1 = 20;
        x = 30;
        y2 = 40;
        z = 50;
        w = 60;
        CHECK(record(tag::Pos{}, tag::A{}) == 10);
        CHECK(record(tag::Pos{}, tag::Y{}) == 20);
        CHECK(record(tag::Vel{}, tag::X{}) == 30);
        CHECK(record(tag::Vel{}, tag::Y{}) == 40);
        CHECK(record(tag::Vel{}, tag::Z{}) == 50);
        CHECK(record(tag::Weight{}) == 60);
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
    struct MyStruct
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
    auto get(const MyStruct<T>& p)
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
struct std::tuple_size<MyStruct<T>>
{
    static constexpr std::size_t value = 3;
};

TEST_CASE("VirtualRecord.load.value")
{
    llama::One<Particle> record;
    record = 1;

    {
        MyPos<int> pos = record(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        MyPos<int> pos = std::as_const(record)(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }

    {
        MyStruct<int> d = record.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
    {
        MyStruct<int> d = std::as_const(record).load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
}

TEST_CASE("VirtualRecord.load.ref")
{
    llama::One<Particle> record;

    record = 1;
    {
        MyPos<int&> pos = record(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);

        pos.a = 2;
        pos.y = 3;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 1);
        CHECK(record(tag::Vel{}, tag::Y{}) == 1);
        CHECK(record(tag::Vel{}, tag::Z{}) == 1);
        CHECK(record(tag::Weight{}) == 1);
    }

    record = 1;
    {
        MyStruct<int&> d = record.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);

        d.pos.a = 2;
        d.pos.y = 3;
        d.vel.x = 4;
        d.vel.y = 5;
        d.vel.z = 6;
        d.weight = 7;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualRecord.load.constref")
{
    llama::One<Particle> record;
    record = 1;

    {
        MyPos<const int&> pos = record(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        MyPos<const int&> pos = std::as_const(record)(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        MyStruct<const int&> d = record.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
    }
    {
        MyStruct<const int&> d = std::as_const(record).load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
    }
}

TEST_CASE("VirtualRecord.store")
{
    llama::One<Particle> record;

    record = 1;
    {
        MyPos<int> pos{2, 3};
        record(tag::Pos{}).store(pos);
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 1);
        CHECK(record(tag::Vel{}, tag::Y{}) == 1);
        CHECK(record(tag::Vel{}, tag::Z{}) == 1);
        CHECK(record(tag::Weight{}) == 1);
    }

    record = 1;
    {
        MyStruct<int> d{{2, 3}, {4, 5, 6}, 7};
        record.store(d);
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Weight{}) == 7);
    }
}

TEST_CASE("VirtualRecord.loadAs.value")
{
    llama::One<Particle> record;
    record = 1;

    {
        auto pos = record(tag::Pos{}).loadAs<MyPos<int>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        auto pos = std::as_const(record)(tag::Pos{}).loadAs<MyPos<int>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
}

TEST_CASE("VirtualRecord.loadAs.ref")
{
    llama::One<Particle> record;
    record = 1;

    auto pos = record(tag::Pos{}).loadAs<MyPos<int&>>();
    CHECK(pos.a == 1);
    CHECK(pos.y == 1);

    pos.a = 2;
    pos.y = 3;
    CHECK(record(tag::Pos{}, tag::A{}) == 2);
    CHECK(record(tag::Pos{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::X{}) == 1);
    CHECK(record(tag::Vel{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::Z{}) == 1);
    CHECK(record(tag::Weight{}) == 1);
}

TEST_CASE("VirtualRecord.loadAs.constref")
{
    llama::One<Particle> record;
    record = 1;

    {
        auto pos = record(tag::Pos{}).loadAs<MyPos<const int&>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        auto pos = std::as_const(record)(tag::Pos{}).loadAs<MyPos<const int&>>();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
}

TEST_CASE("VirtualRecord.One_ctor_from_view")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayDims{5}, Particle{}});
    view(1u) = 1;

    llama::One<Particle> vr2 = view(1u);
    llama::One<Particle> vr3(view(1u));
    llama::One<Particle> vr4{view(1u)};

    vr2 = 2;
    vr3 = 3;
    vr4 = 4;

    CHECK(vr2 == 2);
    CHECK(vr3 == 3);
    CHECK(vr4 == 4);
    CHECK(view(1u) == 1);
}

TEST_CASE("VirtualRecord.One_range_for")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayDims{5}, Particle{}});
    for (auto p : view) // p is a reference object
        p = 1;
    CHECK(view(1u) == 1);

    for (llama::One<Particle> p : view) // p is a copy
        p = 2;
    CHECK(view(1u) == 1);
}

TEST_CASE("VirtualRecord.One_concepts")
{
#ifdef __cpp_concepts
    STATIC_REQUIRE(std::regular<llama::One<Particle>>);
#endif
}

TEST_CASE("VirtualRecord.One_inside_std::vector")
{
    std::vector<llama::One<Particle>> v(2); // create 2 One
    v.push_back(oneParticle()); // add 1 more
    v[0](tag::Weight{}) = 20;
    v[1](tag::Weight{}) = 30;
    v[2](tag::Weight{}) = 10;
    std::sort(
        std::begin(v),
        std::end(v),
        [](const llama::One<Particle>& a, const llama::One<Particle>& b)
        { return a(tag::Weight{}) < b(tag::Weight{}); });
    CHECK(v[0](tag::Weight{}) == 10);
    CHECK(v[1](tag::Weight{}) == 20);
    CHECK(v[2](tag::Weight{}) == 30);
}

TEST_CASE("VirtualRecord.One_from_scalar")
{
    llama::One<Particle> p{42};
    CHECK(p(tag::Pos{}, tag::A{}) == 42);
    CHECK(p(tag::Pos{}, tag::Y{}) == 42);
    CHECK(p(tag::Vel{}, tag::X{}) == 42);
    CHECK(p(tag::Vel{}, tag::Y{}) == 42);
    CHECK(p(tag::Vel{}, tag::Z{}) == 42);
    CHECK(p(tag::Weight{}) == 42);
}

TEST_CASE("VirtualRecord.operator<<")
{
    auto p = oneParticle();

    std::stringstream ss;
    ss << p;
    CHECK(ss.str() == "{Pos: {A: 1, Y: 2}, Vel: {X: 3, Y: 4, Z: 5}, Weight: 6}");

    ss = {};
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayDims{1}, Particle{}});
    view(0u) = p;
    ss << view(0u);
    CHECK(ss.str() == "{Pos: {A: 1, Y: 2}, Vel: {X: 3, Y: 4, Z: 5}, Weight: 6}");
}

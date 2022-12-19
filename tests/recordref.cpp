#include "common.hpp"

#include <algorithm>
#include <sstream>
#include <vector>

namespace
{
    // clang-format off
    using ParticleInt = llama::Record<
        llama::Field<tag::Pos, llama::Record<
            llama::Field<tag::A, int>,
            llama::Field<tag::Y, int>
        >>,
        llama::Field<tag::Vel, Vec3I>,
        llama::Field<tag::Mass, int>
    >;
    // clang-format on

    auto oneParticleInt()
    {
        llama::One<ParticleInt> record;
        record(tag::Pos{}, tag::A{}) = 1;
        record(tag::Pos{}, tag::Y{}) = 2;
        record(tag::Vel{}, tag::X{}) = 3;
        record(tag::Vel{}, tag::Y{}) = 4;
        record(tag::Vel{}, tag::Z{}) = 5;
        record(tag::Mass{}) = 6;
        return record;
    }
} // namespace

TEST_CASE("RecordRef.operator=")
{
    llama::One<ParticleInt> record;

    // scalar to multiple elements in record ref
    record(tag::Pos{}) = 1;
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Mass{}) == 0);

    // scalar to multiple elements in record ref
    record = 2;
    CHECK(record(tag::Pos{}, tag::A{}) == 2);
    CHECK(record(tag::Pos{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Mass{}) == 2);

    // smaller record ref to larger record ref
    record(tag::Pos{}) = 3;
    record(tag::Vel{}) = record(tag::Pos{});
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 3); // only Y is propagated
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Mass{}) == 2);

    // larger record ref to smaller record ref
    record(tag::Vel{}) = 4;
    record(tag::Pos{}) = record(tag::Vel{});
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 4); // only Y is propagated
    CHECK(record(tag::Vel{}, tag::X{}) == 4);
    CHECK(record(tag::Vel{}, tag::Y{}) == 4);
    CHECK(record(tag::Vel{}, tag::Z{}) == 4);
    CHECK(record(tag::Mass{}) == 2);

    // scalar record ref to larger record ref, full broadcast
    record(tag::Mass{}) = 5;
    record(tag::Vel{}) = record(tag::Mass{});
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 4);
    CHECK(record(tag::Vel{}, tag::X{}) == 5); // updated
    CHECK(record(tag::Vel{}, tag::Y{}) == 5); // updated
    CHECK(record(tag::Vel{}, tag::Z{}) == 5); // updated
    CHECK(record(tag::Mass{}) == 5);
}

TEST_CASE("RecordRef.operator+=.scalar")
{
    {
        auto record = oneParticleInt();
        record(tag::Pos{}) += 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Mass{}) == 6);
    }

    {
        auto record = oneParticleInt();
        record += 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Mass{}) == 7);
    }
}

TEST_CASE("RecordRef.operator+=.RecordRef")
{
    {
        // smaller record ref to larger record ref
        auto record = oneParticleInt();
        record(tag::Vel{}) += record(tag::Pos{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Mass{}) == 6);
    }

    {
        // larger record ref to smaller record ref
        auto record = oneParticleInt();
        record(tag::Pos{}) += record(tag::Vel{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Mass{}) == 6);
    }

    {
        // scalar record ref to larger record ref, full broadcast
        auto record = oneParticleInt();
        record(tag::Vel{}) += record(tag::Mass{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 9);
        CHECK(record(tag::Vel{}, tag::Y{}) == 10);
        CHECK(record(tag::Vel{}, tag::Z{}) == 11);
        CHECK(record(tag::Mass{}) == 6);
    }
}

TEST_CASE("RecordRef.operator+.scalar")
{
    {
        auto record = oneParticleInt();
        record(tag::Pos{}) = record(tag::Pos{}) + 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Mass{}) == 6);
    }
    {
        auto record = oneParticleInt();
        record(tag::Pos{}) = 1 + record(tag::Pos{});
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Mass{}) == 6);
    }

    {
        auto record = oneParticleInt();
        record = record + 1;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Mass{}) == 7);
    }
    {
        auto record = oneParticleInt();
        record = 1 + record;
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Mass{}) == 7);
    }
}

TEST_CASE("RecordRef.operator+.RecordRef")
{
    {
        // smaller record ref to larger record ref
        auto record = oneParticleInt();
        record(tag::Vel{}) = record(tag::Vel{}) + record(tag::Pos{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Mass{}) == 6);
    }

    {
        // larger record ref to smaller record ref
        auto record = oneParticleInt();
        record(tag::Pos{}) = record(tag::Pos{}) + record(tag::Vel{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 6);
        CHECK(record(tag::Vel{}, tag::X{}) == 3);
        CHECK(record(tag::Vel{}, tag::Y{}) == 4);
        CHECK(record(tag::Vel{}, tag::Z{}) == 5);
        CHECK(record(tag::Mass{}) == 6);
    }

    {
        // scalar record ref to larger record ref, full broadcast
        auto record = oneParticleInt();
        record(tag::Vel{}) = record(tag::Vel{}) + record(tag::Mass{});
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 9);
        CHECK(record(tag::Vel{}, tag::Y{}) == 10);
        CHECK(record(tag::Vel{}, tag::Z{}) == 11);
        CHECK(record(tag::Mass{}) == 6);
    }
}

namespace
{
    // clang-format off
    using RecordDim2 = llama::Record<
        llama::Field<tag::Part1, llama::Record<
            llama::Field<tag::Mass, int>,
            llama::Field<tag::Pos, llama::Record<
                llama::Field<tag::X, int>,
                llama::Field<tag::Y, int>,
                llama::Field<tag::Z, int>
            >>
        >>,
        llama::Field<tag::Part2, llama::Record<
            llama::Field<tag::Mass, int>,
            llama::Field<tag::Pos, llama::Record<
                llama::Field<tag::X, int>,
                llama::Field<tag::Y, int>,
                llama::Field<tag::A, int>
            >>,
            llama::Field<tag::Z, int>
        >>
    >;
    // clang-format on
} // namespace

TEST_CASE("RecordRef.operator=.propagation")
{
    llama::One<RecordDim2> record;

    record(tag::Part1{}) = 1;
    record(tag::Part2{}) = 2;
    CHECK(record(tag::Part1{}, tag::Mass{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::X{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Z{}) == 1);
    CHECK(record(tag::Part2{}, tag::Mass{}) == 2);
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::X{}) == 2);
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::Y{}) == 2);
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::A{}) == 2);
    CHECK(record(tag::Part2{}, tag::Z{}) == 2);

    record(tag::Part2{}) = record(tag::Part1{});
    CHECK(record(tag::Part1{}, tag::Mass{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::X{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Part1{}, tag::Pos{}, tag::Z{}) == 1);
    CHECK(record(tag::Part2{}, tag::Mass{}) == 1); // propagated
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::X{}) == 1); // propagated
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::Y{}) == 1); // propagated
    CHECK(record(tag::Part2{}, tag::Pos{}, tag::A{}) == 2);
    CHECK(record(tag::Part2{}, tag::Z{}) == 2);
}

TEST_CASE("RecordRef.operator=.multiview")
{
    llama::One<ParticleInt> record1;
    llama::One<RecordDim2> record2;

    record2 = 1;
    record1 = record2;
    CHECK(record1(tag::Pos{}, tag::A{}) == 0);
    CHECK(record1(tag::Pos{}, tag::Y{}) == 0);
    CHECK(record1(tag::Vel{}, tag::X{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record1(tag::Mass{}) == 0);

    record1 = record2(tag::Part1{});
    CHECK(record1(tag::Pos{}, tag::A{}) == 0);
    CHECK(record1(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record1(tag::Vel{}, tag::X{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record1(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record1(tag::Mass{}) == 1);
}

TEST_CASE("RecordRef.operator==")
{
    llama::One<ParticleInt> record;

    record = 1;

    CHECK((record(tag::Pos{}, tag::Y{}) == record(tag::Pos{}, tag::Y{})));
    CHECK((record(tag::Pos{}) == record(tag::Pos{})));
    CHECK((record == record));
    CHECK((record(tag::Pos{}) == record(tag::Vel{})));

    // scalar to multiple elements in record ref
    CHECK((record(tag::Pos{}, tag::Y{}) == 1));
    CHECK((record(tag::Pos{}) == 1));
    CHECK((record == 1));

    record(tag::Pos{}, tag::Y{}) = 2;

    CHECK((record(tag::Pos{}, tag::Y{}) == 2));
    CHECK(!(record(tag::Pos{}) == 1));
    CHECK(!(record == 1));
    CHECK(!(record(tag::Pos{}) == record(tag::Vel{})));
}

TEST_CASE("RecordRef.operator<")
{
    llama::One<ParticleInt> record;

    record = 1;

    CHECK(!(record(tag::Pos{}, tag::Y{}) < record(tag::Pos{}, tag::Y{})));
    CHECK(!(record(tag::Pos{}) < record(tag::Pos{})));
    CHECK(!(record < record));
    CHECK(!(record(tag::Pos{}) < record(tag::Vel{})));

    // scalar to multiple elements in record ref
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

TEST_CASE("RecordRef.asTuple.types")
{
    {
        llama::One<ParticleInt> record;

        const std::tuple<int&, int&> pos = record(tag::Pos{}).asTuple();
        const std::tuple<int&, int&, int&> vel = record(tag::Vel{}).asTuple();
        const std::tuple<std::tuple<int&, int&>, std::tuple<int&, int&, int&>, int&> name = record.asTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
    {
        const llama::One<ParticleInt> record;

        const std::tuple<const int&, const int&> pos = record(tag::Pos{}).asTuple();
        const std::tuple<const int&, const int&, const int&> vel = record(tag::Vel{}).asTuple();
        const std::
            tuple<std::tuple<const int&, const int&>, std::tuple<const int&, const int&, const int&>, const int&>
                name = record.asTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
}

TEST_CASE("RecordRef.asTuple.assign")
{
    llama::One<ParticleInt> record;

    record(tag::Pos{}).asTuple() = std::tuple{1, 1};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Mass{}) == 0);

    record(tag::Vel{}).asTuple() = std::tuple{2, 2, 2};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Mass{}) == 0);

    record.asTuple() = std::tuple{std::tuple{3, 3}, std::tuple{3, 3, 3}, 3};
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::X{}) == 3);
    CHECK(record(tag::Vel{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::Z{}) == 3);
    CHECK(record(tag::Mass{}) == 3);
}

TEST_CASE("RecordRef.asTuple.structuredBindings")
{
    llama::One<ParticleInt> record;

    {
        auto [a, y] = record(tag::Pos{}).asTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        a = 1;
        y = 2;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 0);
        CHECK(record(tag::Vel{}, tag::Y{}) == 0);
        CHECK(record(tag::Vel{}, tag::Z{}) == 0);
        CHECK(record(tag::Mass{}) == 0);
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
        CHECK(record(tag::Mass{}) == 0);
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
        CHECK(record(tag::Mass{}) == 60);
    }
}


TEST_CASE("RecordRef.asFlatTuple.types")
{
    {
        llama::One<ParticleInt> record;

        const std::tuple<int&, int&> pos = record(tag::Pos{}).asFlatTuple();
        const std::tuple<int&, int&, int&> vel = record(tag::Vel{}).asFlatTuple();
        const std::tuple<int&, int&, int&, int&, int&, int&> name = record.asFlatTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
    {
        const llama::One<ParticleInt> record;

        const std::tuple<const int&, const int&> pos = record(tag::Pos{}).asFlatTuple();
        const std::tuple<const int&, const int&, const int&> vel = record(tag::Vel{}).asFlatTuple();
        const std::tuple<const int&, const int&, const int&, const int&, const int&, const int&> name
            = record.asFlatTuple();
        static_cast<void>(pos);
        static_cast<void>(vel);
        static_cast<void>(name);
    }
}

TEST_CASE("RecordRef.asFlatTuple.assign")
{
    llama::One<ParticleInt> record;

    record(tag::Pos{}).asFlatTuple() = std::tuple{1, 1};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 0);
    CHECK(record(tag::Vel{}, tag::Y{}) == 0);
    CHECK(record(tag::Vel{}, tag::Z{}) == 0);
    CHECK(record(tag::Mass{}) == 0);

    record(tag::Vel{}).asFlatTuple() = std::tuple{2, 2, 2};
    CHECK(record(tag::Pos{}, tag::A{}) == 1);
    CHECK(record(tag::Pos{}, tag::Y{}) == 1);
    CHECK(record(tag::Vel{}, tag::X{}) == 2);
    CHECK(record(tag::Vel{}, tag::Y{}) == 2);
    CHECK(record(tag::Vel{}, tag::Z{}) == 2);
    CHECK(record(tag::Mass{}) == 0);

    record.asFlatTuple() = std::tuple{3, 3, 3, 3, 3, 3};
    CHECK(record(tag::Pos{}, tag::A{}) == 3);
    CHECK(record(tag::Pos{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::X{}) == 3);
    CHECK(record(tag::Vel{}, tag::Y{}) == 3);
    CHECK(record(tag::Vel{}, tag::Z{}) == 3);
    CHECK(record(tag::Mass{}) == 3);
}

TEST_CASE("RecordRef.asFlatTuple.structuredBindings")
{
    llama::One<ParticleInt> record;

    {
        auto [a, y] = record(tag::Pos{}).asFlatTuple(); // NOLINT (clang-analyzer-deadcode.DeadStores)
        a = 1;
        y = 2;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 0);
        CHECK(record(tag::Vel{}, tag::Y{}) == 0);
        CHECK(record(tag::Vel{}, tag::Z{}) == 0);
        CHECK(record(tag::Mass{}) == 0);
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
        CHECK(record(tag::Mass{}) == 0);
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
        CHECK(record(tag::Mass{}) == 60);
    }
}

template<typename T>
struct S;

TEST_CASE("RecordRef.structuredBindings")
{
    llama::One<ParticleInt> record;

    {
        auto&& [a, y] = record(tag::Pos{});
        a = 1;
        y = 2;
        CHECK(record(tag::Pos{}, tag::A{}) == 1);
        CHECK(record(tag::Pos{}, tag::Y{}) == 2);
        CHECK(record(tag::Vel{}, tag::X{}) == 0);
        CHECK(record(tag::Vel{}, tag::Y{}) == 0);
        CHECK(record(tag::Vel{}, tag::Z{}) == 0);
        CHECK(record(tag::Mass{}) == 0);
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
        CHECK(record(tag::Mass{}) == 0);
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
        CHECK(record(tag::Mass{}) == 60);
    }
}

namespace
{
    template<typename T>
    struct MyPos
    {
        T a;
        T y;
    };

    template<typename T>
    struct MyVel
    {
        T x;
        T y;
        T z;
    };

    template<typename T>
    struct MyStruct
    {
        MyPos<T> pos;
        MyVel<T> vel;
        T weight;
    };

    template<std::size_t I, typename T>
    LLAMA_FN_HOST_ACC_INLINE auto get(const MyPos<T>& p)
    {
        if constexpr(I == 0)
            return p.a;
        if constexpr(I == 1)
            return p.y;
    }

    template<std::size_t I, typename T>
    LLAMA_FN_HOST_ACC_INLINE auto get(const MyVel<T>& p)
    {
        if constexpr(I == 0)
            return p.x;
        if constexpr(I == 1)
            return p.y;
        if constexpr(I == 2)
            return p.z;
    }

    template<std::size_t I, typename T>
    LLAMA_FN_HOST_ACC_INLINE auto get(const MyStruct<T>& p)
    {
        if constexpr(I == 0)
            return p.pos;
        if constexpr(I == 1)
            return p.vel;
        if constexpr(I == 2)
            return p.weight;
    }
} // namespace

template<typename T>
struct std::tuple_size<MyPos<T>> // NOLINT(cert-dcl58-cpp)
{
    static constexpr std::size_t value = 2;
};

template<typename T>
struct std::tuple_size<MyVel<T>> // NOLINT(cert-dcl58-cpp)
{
    static constexpr std::size_t value = 3;
};

template<typename T>
struct std::tuple_size<MyStruct<T>> // NOLINT(cert-dcl58-cpp)
{
    static constexpr std::size_t value = 3;
};

TEST_CASE("RecordRef.load.value")
{
    llama::One<ParticleInt> record;
    record = 1;

    {
        const MyPos<int> pos = record(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        const MyPos<int> pos = std::as_const(record)(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }

    {
        const MyStruct<int> d = record.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
    {
        const MyStruct<int> d = std::as_const(record).load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
}

TEST_CASE("RecordRef.load.ref")
{
    llama::One<ParticleInt> record;

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
        CHECK(record(tag::Mass{}) == 1);
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
        CHECK(record(tag::Mass{}) == 7);
    }
}

TEST_CASE("RecordRef.load.constref")
{
    llama::One<ParticleInt> record;
    record = 1;

    {
        const MyPos<const int&> pos = record(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        const MyPos<const int&> pos = std::as_const(record)(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        const MyStruct<const int&> d = record.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
    }
    {
        const MyStruct<const int&> d = std::as_const(record).load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
    }
}

TEST_CASE("RecordRef.load.value.fromproxyref")
{
    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtents<int, 1>, ParticleInt, llama::Constant<14>>{});
    auto&& record = view(0);
    record = 1;

    {
        const MyPos<int> pos = record(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }
    {
        const MyPos<int> pos = std::as_const(record)(tag::Pos{}).load();
        CHECK(pos.a == 1);
        CHECK(pos.y == 1);
    }

    {
        const MyStruct<int> d = record.load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
    {
        const MyStruct<int> d = std::as_const(record).load();
        CHECK(d.pos.a == 1);
        CHECK(d.pos.y == 1);
        CHECK(d.vel.x == 1);
        CHECK(d.vel.y == 1);
        CHECK(d.vel.z == 1);
        CHECK(d.weight == 1);
    }
}

TEST_CASE("RecordRef.store")
{
    llama::One<ParticleInt> record;

    record = 1;
    {
        const MyPos<int> pos{2, 3};
        record(tag::Pos{}).store(pos);
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 1);
        CHECK(record(tag::Vel{}, tag::Y{}) == 1);
        CHECK(record(tag::Vel{}, tag::Z{}) == 1);
        CHECK(record(tag::Mass{}) == 1);
    }

    record = 1;
    {
        const MyStruct<int> d{{2, 3}, {4, 5, 6}, 7};
        record.store(d);
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Mass{}) == 7);
    }
}

TEST_CASE("RecordRef.store.toproxyref")
{
    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtents<int, 1>, ParticleInt, llama::Constant<14>>{});
    auto&& record = view(0);

    record = 1;
    {
        const MyPos<int> pos{2, 3};
        record(tag::Pos{}).store(pos);
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 1);
        CHECK(record(tag::Vel{}, tag::Y{}) == 1);
        CHECK(record(tag::Vel{}, tag::Z{}) == 1);
        CHECK(record(tag::Mass{}) == 1);
    }

    record = 1;
    {
        const MyStruct<int> d{{2, 3}, {4, 5, 6}, 7};
        record.store(d);
        CHECK(record(tag::Pos{}, tag::A{}) == 2);
        CHECK(record(tag::Pos{}, tag::Y{}) == 3);
        CHECK(record(tag::Vel{}, tag::X{}) == 4);
        CHECK(record(tag::Vel{}, tag::Y{}) == 5);
        CHECK(record(tag::Vel{}, tag::Z{}) == 6);
        CHECK(record(tag::Mass{}) == 7);
    }
}

TEST_CASE("RecordRef.loadAs.value")
{
    llama::One<ParticleInt> record;
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

TEST_CASE("RecordRef.loadAs.ref")
{
    llama::One<ParticleInt> record;
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
    CHECK(record(tag::Mass{}) == 1);
}

TEST_CASE("RecordRef.loadAs.constref")
{
    llama::One<ParticleInt> record;
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

TEST_CASE("RecordRef.loadAs.value.fromproxyref")
{
    auto view = llama::allocView(
        llama::mapping::BitPackedIntSoA<llama::ArrayExtents<int, 1>, ParticleInt, llama::Constant<14>>{});
    auto&& record = view(0);
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

TEST_CASE("RecordRef.One_ctor_from_view")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{5}, ParticleInt{}});
    view(1u) = 1;

    llama::One<ParticleInt> vr2 = view(1u);
    llama::One<ParticleInt> vr3(view(1u));
    llama::One<ParticleInt> vr4{view(1u)};

    vr2 = 2;
    vr3 = 3;
    vr4 = 4;

    CHECK(vr2 == 2);
    CHECK(vr3 == 3);
    CHECK(vr4 == 4);
    CHECK(view(1u) == 1);
}

TEST_CASE("RecordRef.One_range_for")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{5}, ParticleInt{}});
    for(auto p : view) // p is a reference object
        p = 1;
    CHECK(view(1u) == 1);

    for(llama::One<ParticleInt> p : view) // p is a copy
        p = 2;
    CHECK(view(1u) == 1);
}

TEST_CASE("RecordRef.One_concepts")
{
#ifdef __cpp_lib_concepts
    STATIC_REQUIRE(std::regular<llama::One<ParticleInt>>);
#endif
}

TEST_CASE("RecordRef.One_inside_std::vector")
{
    std::vector<llama::One<ParticleInt>> v(2); // create 2 One
    v.push_back(oneParticleInt()); // add 1 more
    v[0](tag::Mass{}) = 20;
    v[1](tag::Mass{}) = 30;
    v[2](tag::Mass{}) = 10;
    std::sort(
        std::begin(v),
        std::end(v),
        [](const llama::One<ParticleInt>& a, const llama::One<ParticleInt>& b)
        { return a(tag::Mass{}) < b(tag::Mass{}); });
    CHECK(v[0](tag::Mass{}) == 10);
    CHECK(v[1](tag::Mass{}) == 20);
    CHECK(v[2](tag::Mass{}) == 30);
}

TEST_CASE("RecordRef.One_from_scalar")
{
    llama::One<ParticleInt> p{42};
    CHECK(p(tag::Pos{}, tag::A{}) == 42);
    CHECK(p(tag::Pos{}, tag::Y{}) == 42);
    CHECK(p(tag::Vel{}, tag::X{}) == 42);
    CHECK(p(tag::Vel{}, tag::Y{}) == 42);
    CHECK(p(tag::Vel{}, tag::Z{}) == 42);
    CHECK(p(tag::Mass{}) == 42);
}

TEST_CASE("RecordRef.size.int")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{5}, ParticleInt{}});
    [[maybe_unused]] auto vr = view[0];
    STATIC_REQUIRE(
        sizeof(vr)
        == sizeof(llama::ArrayExtents<int, llama::dyn>::value_type) + 4 // padding
            + sizeof(&view)); // sizeof array dims and view reference // NOLINT
}

TEST_CASE("RecordRef.size.size_t")
{
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{std::size_t{5}}, ParticleInt{}});
    [[maybe_unused]] auto vr = view[0];
    STATIC_REQUIRE(
        sizeof(vr)
        == sizeof(llama::ArrayExtents<std::size_t, llama::dyn>::value_type)
            + sizeof(&view)); // sizeof array dims and view reference // NOLINT
}

TEST_CASE("RecordRef.One.size")
{
    using Mapping = llama::mapping::MinAlignedOne<llama::ArrayExtents<int>, Particle>;
    STATIC_REQUIRE(Mapping{}.blobSize(0) == 56);
    STATIC_REQUIRE(std::is_empty_v<Mapping::ArrayIndex>);

    [[maybe_unused]] const auto v = llama::allocViewStack<0, Particle>();
    STATIC_REQUIRE(std::is_same_v<typename decltype(v)::Mapping, Mapping>);
    STATIC_REQUIRE(sizeof(v) == 56);

    [[maybe_unused]] const auto p = llama::One<Particle>{};
    STATIC_REQUIRE(sizeof(p) == 56);
}

TEST_CASE("RecordRef.One.alignment")
{
    STATIC_REQUIRE(alignof(llama::One<Vec2F>) == 4);
    STATIC_REQUIRE(alignof(llama::One<Vec3I>) == 4);
    STATIC_REQUIRE(alignof(llama::One<Particle>) == 8);
}

TEST_CASE("RecordRef.operator<<")
{
    llama::One<Particle> p;
    llama::forEachLeafCoord<Particle>([&, i = 0](auto rc) mutable { p(rc) = ++i; });

    std::stringstream ss;
    ss << p;
    const auto* const expected
        = "{Pos: {X: 1, Y: 2, Z: 3}, Mass: 4, Vel: {X: 5, Y: 6, Z: 7}, Flags: {[0]: 1, [1]: 1, [2]: 1, [3]: 1}}";
    CHECK(ss.str() == expected);

    ss = std::stringstream{};
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, Particle{}});
    view(0u) = p;
    ss << view(0u);
    CHECK(ss.str() == expected);
}

TEST_CASE("RecordRef.swap")
{
    llama::One<Vec3I> p1{1};
    llama::One<Vec3I> p2{2};

    llama::forEachLeafCoord<Vec3I>(
        [&](auto rc)
        {
            CHECK(p1(rc) == 1);
            CHECK(p2(rc) == 2);
        });

    using std::swap; // to check that overload set (std::swap, llama::swap) is not ambigious
    swap(p1, p2);

    llama::forEachLeafCoord<Vec3I>(
        [&](auto rc)
        {
            CHECK(p1(rc) == 2);
            CHECK(p2(rc) == 1);
        });
}

TEST_CASE("RecordRef.forEachLeaf")
{
    llama::One<Particle> p;
    llama::forEachLeaf(p, [i = 0](auto& field) mutable { field = ++i; });
    CHECK(p(tag::Pos{}, tag::X{}) == 1);
    CHECK(p(tag::Pos{}, tag::Y{}) == 2);
    CHECK(p(tag::Pos{}, tag::Z{}) == 3);
    CHECK(p(tag::Mass{}) == 4);
    CHECK(p(tag::Vel{}, tag::X{}) == 5);
    CHECK(p(tag::Vel{}, tag::Y{}) == 6);
    CHECK(p(tag::Vel{}, tag::Z{}) == 7);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);

    llama::forEachLeaf(p(tag::Pos{}), [i = 10](auto& field) mutable { field = ++i; });
    CHECK(p(tag::Pos{}, tag::X{}) == 11);
    CHECK(p(tag::Pos{}, tag::Y{}) == 12);
    CHECK(p(tag::Pos{}, tag::Z{}) == 13);
    CHECK(p(tag::Mass{}) == 4);
    CHECK(p(tag::Vel{}, tag::X{}) == 5);
    CHECK(p(tag::Vel{}, tag::Y{}) == 6);
    CHECK(p(tag::Vel{}, tag::Z{}) == 7);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);
    CHECK(p(tag::Flags{})(llama::RecordCoord<0>{}) == true);
}

TEST_CASE("RecordRef.reference_to_One")
{
    llama::One<Vec3I> v;
    llama::forEachLeaf(v, [i = 0](auto& field) mutable { field = ++i; });
    CHECK(v(tag::X{}) == 1);
    CHECK(v(tag::Y{}) == 2);
    CHECK(v(tag::Z{}) == 3);

    STATIC_REQUIRE(std::is_same_v<decltype(v(llama::RecordCoord<>{})), decltype(v())>);
    auto r = v();
    CHECK(r(tag::X{}) == 1);
    CHECK(r(tag::Y{}) == 2);
    CHECK(r(tag::Z{}) == 3);

    r(tag::Y{}) = 22;
    CHECK(r(tag::X{}) == 1);
    CHECK(r(tag::Y{}) == 22);
    CHECK(r(tag::Z{}) == 3);

    CHECK(v(tag::X{}) == 1);
    CHECK(v(tag::Y{}) == 22);
    CHECK(v(tag::Z{}) == 3);
}

TEST_CASE("ValueOf")
{
    STATIC_REQUIRE(std::is_same_v<llama::internal::ValueOf<int&>::type, int>);

    using One = llama::One<Vec3I>;
    STATIC_REQUIRE(std::is_same_v<llama::internal::ValueOf<decltype(One{}())>::type, One>);

    auto mapping = llama::mapping::BitPackedIntSoA<llama::ArrayExtents<int, 4>, Vec3I>{{}, 17};
    auto v = llama::allocView(mapping);
    [[maybe_unused]] auto ref = v(1)(tag::X{});
#ifdef __cpp_lib_concepts
    STATIC_REQUIRE(llama::ProxyReference<decltype(ref)>);
#endif
    STATIC_REQUIRE(std::is_same_v<llama::internal::ValueOf<decltype(ref)>::type, int>);
}
TEST_CASE("ScopedUpdate.Fundamental")
{
    int i = 1;
    {
        llama::ScopedUpdate u(i);
        STATIC_REQUIRE(std::is_same_v<decltype(u), llama::ScopedUpdate<int&>>);
        u = 23;
        CHECK(u == 23);
        CHECK(u.get() == 23);
        CHECK(i == 1);
        u = 24;
        CHECK(u == 24);
        CHECK(u.get() == 24);
        CHECK(i == 1);
    }
    CHECK(i == 24);
}

TEST_CASE("ScopedUpdate.Object")
{
    std::vector v = {1};
    {
        llama::ScopedUpdate u(v);
        STATIC_REQUIRE(std::is_same_v<decltype(u), llama::ScopedUpdate<std::vector<int>&>>);
        u.push_back(2);
        CHECK(u == std::vector{1, 2});
        CHECK(u.get() == std::vector{1, 2});
        CHECK(v == std::vector{1});
        u = std::vector{3, 4, 5};
        CHECK(u == std::vector{3, 4, 5});
        CHECK(u.get() == std::vector{3, 4, 5});
        CHECK(v == std::vector{1});
    }
    CHECK(v == std::vector{3, 4, 5});
}

TEST_CASE("ScopedUpdate.ProxyRef")
{
    auto mapping = llama::mapping::BitPackedIntSoA<llama::ArrayExtents<int, 4>, Vec3I>{{}, 17};
    auto v = llama::allocView(mapping);
    auto i = v(1)(tag::X{});
    i = 1;
    {
        llama::ScopedUpdate u(i);
        STATIC_REQUIRE(std::is_same_v<decltype(u), llama::ScopedUpdate<decltype(i)>>);
        u = 23;
        CHECK(u == 23);
        CHECK(u.get() == 23);
        CHECK(i == 1);
        u = 24;
        CHECK(u == 24);
        CHECK(u.get() == 24);
        CHECK(i == 1);
    }
    CHECK(i == 24);
}

TEST_CASE("ScopedUpdate.RecordRef")
{
    auto test = [](auto&& v)
    {
        llama::forEachLeaf(v, [i = 0](auto& field) mutable { field = ++i; });
        {
            llama::ScopedUpdate u(v);
            if constexpr(llama::isOne<std::remove_reference_t<decltype(v)>>)
            {
                STATIC_REQUIRE(
                    std::is_same_v<decltype(u), llama::ScopedUpdate<std::remove_reference_t<decltype(v)>&>>);
            }
            else
            {
                STATIC_REQUIRE(std::is_same_v<decltype(u), llama::ScopedUpdate<decltype(v())>>);
            }
            u(tag::X{}) = 11;
            CHECK(u(tag::X{}) == 11);
            CHECK(u(tag::Y{}) == 2);
            CHECK(u(tag::Z{}) == 3);
            CHECK(u.get()(tag::Z{}) == 3);
            CHECK(v(tag::X{}) == 1);
            CHECK(v(tag::Y{}) == 2);
            CHECK(v(tag::Z{}) == 3);
            CHECK(v(tag::Z{}) == 3);
            u = 24;
            CHECK(u(tag::X{}) == 24);
            CHECK(u(tag::Y{}) == 24);
            CHECK(u(tag::Z{}) == 24);
            CHECK(u.get()(tag::Z{}) == 24);
            CHECK(v(tag::X{}) == 1);
            CHECK(v(tag::Y{}) == 2);
            CHECK(v(tag::Z{}) == 3);
        }
        CHECK(v(tag::X{}) == 24);
        CHECK(v(tag::Y{}) == 24);
        CHECK(v(tag::Z{}) == 24);
    };
    llama::One<Vec3I> v;
    test(v);
    test(v());
}

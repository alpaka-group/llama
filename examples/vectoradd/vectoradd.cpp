// Copyright 2021 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "../common/Stats.hpp"
#include "../common/Stopwatch.hpp"
#include "../common/env.hpp"

#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>

constexpr auto problemSize = 64 * 1024 * 1024 + 3; ///< problem size
constexpr auto steps = 20; ///< number of vector adds to perform, excluding 1 warmup run
constexpr auto aosoaLanes = 16;

// use different types for various struct members to alignment/padding plays a role
using X_t = float;
using Y_t = double;
using Z_t = std::uint16_t;

namespace usellama
{
    // clang-format off
    namespace tag
    {
        struct X{} x;
        struct Y{} y;
        struct Z{} z;
    } // namespace tag

    using Vector = llama::Record<
        llama::Field<tag::X, X_t>,
        llama::Field<tag::Y, Y_t>,
        llama::Field<tag::Z, Z_t>
    >;
    // clang-format on

    template<typename View>
    [[gnu::noinline]] void compute(const View& a, const View& b, View& c)
    {
        const auto [n] = c.extents();

        for(std::size_t i = 0; i < n; i++)
        {
            c(i)(tag::x) = a(i)(tag::x) + b(i)(tag::x);
            c(i)(tag::y) = a(i)(tag::y) - b(i)(tag::y);
            c(i)(tag::z) = a(i)(tag::z) * b(i)(tag::z);
        }
    }

    template<int MappingIndex>
    auto main(std::ofstream& plotFile) -> int
    {
        const auto mappingname = [&]
        {
            if constexpr(MappingIndex == 0)
                return "AoS";
            if constexpr(MappingIndex == 1)
                return "SoA SB packed";
            if constexpr(MappingIndex == 2)
                return "SoA SB aligned";
            if constexpr(MappingIndex == 3)
                return "SoA SB aligned CT size";
            if constexpr(MappingIndex == 4)
                return "SoA MB";
            if constexpr(MappingIndex == 5)
                return "AoSoA" + std::to_string(aosoaLanes);
        }();

        std::cout << "\nLLAMA " << mappingname << "\n";
        Stopwatch watch;

        const auto mapping = [&]
        {
            using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 1>;
            const auto extents = ArrayExtents{problemSize};
            if constexpr(MappingIndex == 0)
                return llama::mapping::AoS{extents, Vector{}};
            if constexpr(MappingIndex == 1)
                return llama::mapping::
                    SoA<ArrayExtents, Vector, llama::mapping::Blobs::Single, llama::mapping::SubArrayAlignment::Pack>{
                        extents};
            if constexpr(MappingIndex == 2)
                return llama::mapping::SoA<
                    llama::ArrayExtents<std::size_t, problemSize>,
                    Vector,
                    llama::mapping::Blobs::Single,
                    llama::mapping::SubArrayAlignment::Align>{};
            if constexpr(MappingIndex == 3)
                return llama::mapping::
                    SoA<ArrayExtents, Vector, llama::mapping::Blobs::Single, llama::mapping::SubArrayAlignment::Align>{
                        extents};
            if constexpr(MappingIndex == 4)
                return llama::mapping::SoA<ArrayExtents, Vector, llama::mapping::Blobs::OnePerField>{extents};
            if constexpr(MappingIndex == 5)
                return llama::mapping::AoSoA<ArrayExtents, Vector, aosoaLanes>{extents};
        }();

        auto a = allocViewUninitialized(mapping);
        auto b = allocViewUninitialized(mapping);
        auto c = allocViewUninitialized(mapping);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            a[i](tag::x) = i; // X
            a[i](tag::y) = i; // Y
            a[i](llama::RecordCoord<2>{}) = i; // Z
            b(i) = i; // writes to all (X, Y, Z)
        }
        watch.printAndReset("init");

        common::Stats stats;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            compute(a, b, c);
            stats(watch.printAndReset("vectoradd"));
        }
        plotFile << "\"LLAMA " << mappingname << "\"\t" << stats.mean() << '\t' << stats.sem() << '\n';

        return static_cast<int>(c.blobs()[0][0]);
    }
} // namespace usellama

namespace manualAoS
{
    struct Vector
    {
        X_t x;
        Y_t y;
        Z_t z;
    };

    [[gnu::noinline]] void compute(const Vector* a, const Vector* b, Vector* c, std::size_t n)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < n; i++)
        {
            c[i].x = a[i].x + b[i].x;
            c[i].y = a[i].y - b[i].y;
            c[i].z = a[i].z * b[i].z;
        }
    }

    auto main(std::ofstream& plotFile) -> int
    {
        std::cout << "\nAoS\n";
        Stopwatch watch;

        std::vector<Vector> a(problemSize);
        std::vector<Vector> b(problemSize);
        std::vector<Vector> c(problemSize);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            b[i].x = a[i].x = static_cast<X_t>(i);
            b[i].y = a[i].y = static_cast<Y_t>(i);
            b[i].z = a[i].z = static_cast<Z_t>(i);
        }
        watch.printAndReset("init");

        common::Stats stats;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            compute(a.data(), b.data(), c.data(), problemSize);
            stats(watch.printAndReset("vectoradd"));
        }
        plotFile << "AoS\t" << stats.mean() << '\t' << stats.sem() << '\n';

        return static_cast<int>(c[0].x);
    }
} // namespace manualAoS

namespace manualSoA
{
    [[gnu::noinline]] void compute(
        const X_t* ax,
        const Y_t* ay,
        const Z_t* az,
        const X_t* bx,
        const Y_t* by,
        const Z_t* bz,
        X_t* cx,
        Y_t* cy,
        Z_t* cz,
        std::size_t n)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < n; i++)
        {
            cx[i] = ax[i] + bx[i];
            cy[i] = ay[i] - by[i];
            cz[i] = az[i] * bz[i];
        }
    }

    auto main(std::ofstream& plotFile) -> int
    {
        std::cout << "\nSoA\n";
        Stopwatch watch;

        using VectorX = std::vector<X_t, llama::bloballoc::AlignedAllocator<X_t, 64>>;
        using VectorY = std::vector<Y_t, llama::bloballoc::AlignedAllocator<Y_t, 64>>;
        using VectorZ = std::vector<Z_t, llama::bloballoc::AlignedAllocator<Z_t, 64>>;
        VectorX ax(problemSize);
        VectorY ay(problemSize);
        VectorZ az(problemSize);
        VectorX bx(problemSize);
        VectorY by(problemSize);
        VectorZ bz(problemSize);
        VectorX cx(problemSize);
        VectorY cy(problemSize);
        VectorZ cz(problemSize);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            bx[i] = ax[i] = static_cast<X_t>(i);
            by[i] = ay[i] = static_cast<Y_t>(i);
            bz[i] = az[i] = static_cast<Z_t>(i);
        }
        watch.printAndReset("init");

        common::Stats stats;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            compute(
                ax.data(),
                ay.data(),
                az.data(),
                bx.data(),
                by.data(),
                bz.data(),
                cx.data(),
                cy.data(),
                cz.data(),
                problemSize);
            stats(watch.printAndReset("vectoradd"));
        }
        plotFile << "SoA\t" << stats.mean() << '\t' << stats.sem() << '\n';

        return static_cast<int>(cx[0]);
    }
} // namespace manualSoA


namespace manualAoSoA
{
    struct alignas(64) VectorBlock
    {
        X_t x[aosoaLanes];
        Y_t y[aosoaLanes];
        Z_t z[aosoaLanes];
    };

    [[gnu::noinline]] void compute(const VectorBlock* a, const VectorBlock* b, VectorBlock* c, std::size_t n)
    {
        for(std::size_t bi = 0; bi < n / aosoaLanes; bi++)
        {
// the unroll 1 is needed to prevent unrolling, which prevents vectorization in GCC
#pragma GCC unroll 1
            LLAMA_INDEPENDENT_DATA
            for(std::size_t i = 0; i < aosoaLanes; ++i)
            {
                const auto& blockA = a[bi];
                const auto& blockB = b[bi];
                auto& blockC = c[bi];
                blockC.x[i] = blockA.x[i] + blockB.x[i];
                blockC.y[i] = blockA.y[i] - blockB.y[i];
                blockC.z[i] = blockA.z[i] * blockB.z[i];
            }
        }
    }

    auto main(std::ofstream& plotFile) -> int
    {
        std::cout << "\nAoSoA\n";
        Stopwatch watch;

        constexpr auto blocks = problemSize / aosoaLanes;
        std::vector<VectorBlock> a(blocks);
        std::vector<VectorBlock> b(blocks);
        std::vector<VectorBlock> c(blocks);
        watch.printAndReset("alloc");

        for(std::size_t bi = 0; bi < problemSize / aosoaLanes; ++bi)
        {
            LLAMA_INDEPENDENT_DATA
            for(std::size_t i = 0; i < aosoaLanes; ++i)
            {
                b[bi].x[i] = a[bi].x[i] = static_cast<X_t>(bi * aosoaLanes + i);
                b[bi].y[i] = a[bi].y[i] = static_cast<Y_t>(bi * aosoaLanes + i);
                b[bi].z[i] = a[bi].z[i] = static_cast<Z_t>(bi * aosoaLanes + i);
            }
        }
        watch.printAndReset("init");

        common::Stats stats;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            compute(a.data(), b.data(), c.data(), problemSize);
            stats(watch.printAndReset("vectoradd"));
        }
        plotFile << "AoSoA\t" << stats.mean() << '\t' << stats.sem() << '\n';

        return static_cast<int>(c[0].x[0]);
    }
} // namespace manualAoSoA


auto main() -> int
try
{
    const auto env = common::captureEnv();
    std::cout << problemSize / 1000 / 1000 << "M values "
              << "(" << problemSize * sizeof(float) / 1024 << "kiB)\n"
              << env << '\n';

    std::ofstream plotFile{"vectoradd.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# {}
set title "vectoradd CPU {}Mi elements"
set style data histograms
set style histogram errorbars
set style fill solid border -1
set xtics rotate by 45 right nomirror
set key off
set yrange [0:*]
set ylabel "runtime [s]"
$data << EOD
""	"runtime"	"runtime_sem"
)",
        env,
        problemSize / 1024 / 1024);

    int r = 0;
    boost::mp11::mp_for_each<boost::mp11::mp_iota_c<6>>([&](auto ic)
                                                        { r += usellama::main<decltype(ic)::value>(plotFile); });
    r += manualAoS::main(plotFile);
    r += manualSoA::main(plotFile);
    r += manualAoSoA::main(plotFile);

    plotFile << R"(EOD
plot $data using 2:3:xtic(1) ti col
)";
    std::cout << "Plot with: ./vectoradd.sh\n";

    return r;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

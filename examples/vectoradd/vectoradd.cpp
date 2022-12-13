#include "../common/Stopwatch.hpp"
#include "../common/hostname.hpp"

#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>

constexpr auto mappingIndex = 2; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blobs), 3 tree AoS, 4 tree SoA
constexpr auto problemSize = 64 * 1024 * 1024; ///< problem size
constexpr auto steps = 10; ///< number of vector adds to perform

using FP = float;

namespace usellama
{
    // clang-format off
    namespace tag
    {
        struct X{};
        struct Y{};
        struct Z{};
    } // namespace tag

    using Vector = llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>
    >;
    // clang-format on

    template<typename View>
    void add(const View& a, const View& b, View& c)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; i++)
        {
            c(i)(tag::X{}) = a(i)(tag::X{}) + b(i)(tag::X{});
            c(i)(tag::Y{}) = a(i)(tag::Y{}) - b(i)(tag::Y{});
            c(i)(tag::Z{}) = a(i)(tag::Z{}) * b(i)(tag::Z{});
        }
    }

    auto main(std::ofstream& plotFile) -> int
    {
        std::cout << "\nLLAMA\n";
        Stopwatch watch;

        const auto mapping = [&]
        {
            using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 1>;
            const auto extents = ArrayExtents{problemSize};
            if constexpr(mappingIndex == 0)
                return llama::mapping::AoS{extents, Vector{}};
            if constexpr(mappingIndex == 1)
                return llama::mapping::SoA<ArrayExtents, Vector, llama::mapping::Blobs::Single>{extents};
            if constexpr(mappingIndex == 2)
                return llama::mapping::SoA<ArrayExtents, Vector, llama::mapping::Blobs::OnePerField>{extents};
            if constexpr(mappingIndex == 3)
                return llama::mapping::tree::Mapping{extents, llama::Tuple{}, Vector{}};
            if constexpr(mappingIndex == 4)
                return llama::mapping::tree::Mapping{
                    extents,
                    llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                    Vector{}};
        }();

        auto a = allocViewUninitialized(mapping);
        auto b = allocViewUninitialized(mapping);
        auto c = allocViewUninitialized(mapping);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            const auto value = static_cast<FP>(i);
            a[i](tag::X{}) = value; // X
            a[i](tag::Y{}) = value; // Y
            a[i](llama::RecordCoord<2>{}) = value; // Z
            b(i) = value; // writes to all (X, Y, Z)
        }
        watch.printAndReset("init");

        double acc = 0;
        for(std::size_t s = 0; s < steps; ++s)
        {
            add(a, b, c);
            acc += watch.printAndReset("add");
        }
        plotFile << "LLAMA\t" << acc / steps << '\n';

        return static_cast<int>(c.storageBlobs[0][0]);
    }
} // namespace usellama

namespace manualAoS
{
    struct Vector
    {
        FP x;
        FP y;
        FP z;
    };

    inline void add(const Vector* a, const Vector* b, Vector* c)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; i++)
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
            const auto value = static_cast<FP>(i);
            a[i].x = value;
            a[i].y = value;
            a[i].z = value;
            b[i].x = value;
            b[i].y = value;
            b[i].z = value;
        }
        watch.printAndReset("init");

        double acc = 0;
        for(std::size_t s = 0; s < steps; ++s)
        {
            add(a.data(), b.data(), c.data());
            acc += watch.printAndReset("add");
        }
        plotFile << "AoS\t" << acc / steps << '\n';

        return static_cast<int>(c[0].x);
    }
} // namespace manualAoS

namespace manualSoA
{
    inline void add(
        const FP* ax,
        const FP* ay,
        const FP* az,
        const FP* bx,
        const FP* by,
        const FP* bz,
        FP* cx,
        FP* cy,
        FP* cz)
    {
        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; i++)
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

        using Vector = std::vector<float, llama::bloballoc::AlignedAllocator<float, 64>>;
        Vector ax(problemSize);
        Vector ay(problemSize);
        Vector az(problemSize);
        Vector bx(problemSize);
        Vector by(problemSize);
        Vector bz(problemSize);
        Vector cx(problemSize);
        Vector cy(problemSize);
        Vector cz(problemSize);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            const auto value = static_cast<FP>(i);
            ax[i] = value;
            ay[i] = value;
            az[i] = value;
            bx[i] = value;
            by[i] = value;
            bz[i] = value;
        }
        watch.printAndReset("init");

        double acc = 0;
        for(std::size_t s = 0; s < steps; ++s)
        {
            add(ax.data(), ay.data(), az.data(), bx.data(), by.data(), bz.data(), cx.data(), cy.data(), cz.data());
            acc += watch.printAndReset("add");
        }
        plotFile << "SoA\t" << acc / steps << '\n';

        return static_cast<int>(cx[0]);
    }
} // namespace manualSoA


namespace manualAoSoA
{
    constexpr auto lanes = 16;

    struct alignas(64) VectorBlock
    {
        FP x[lanes];
        FP y[lanes];
        FP z[lanes];
    };

    constexpr auto blocks = problemSize / lanes;

    inline void add(const VectorBlock* a, const VectorBlock* b, VectorBlock* c)
    {
        for(std::size_t bi = 0; bi < problemSize / lanes; bi++)
        {
// the unroll 1 is needed to prevent unrolling, which prevents vectorization in GCC
#pragma GCC unroll 1
            LLAMA_INDEPENDENT_DATA
            for(std::size_t i = 0; i < lanes; ++i)
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

        std::vector<VectorBlock> a(blocks);
        std::vector<VectorBlock> b(blocks);
        std::vector<VectorBlock> c(blocks);
        watch.printAndReset("alloc");

        for(std::size_t bi = 0; bi < problemSize / lanes; ++bi)
        {
            LLAMA_INDEPENDENT_DATA
            for(std::size_t i = 0; i < lanes; ++i)
            {
                const auto value = static_cast<float>(bi * lanes + i);
                a[bi].x[i] = value;
                a[bi].y[i] = value;
                a[bi].z[i] = value;
                b[bi].x[i] = value;
                b[bi].y[i] = value;
                b[bi].z[i] = value;
            }
        }
        watch.printAndReset("init");

        double acc = 0;
        for(std::size_t s = 0; s < steps; ++s)
        {
            add(a.data(), b.data(), c.data());
            acc += watch.printAndReset("add");
        }
        plotFile << "AoSoA\t" << acc / steps << '\n';

        return static_cast<int>(c[0].x[0]);
    }
} // namespace manualAoSoA


auto main() -> int
try
{
    std::cout << problemSize / 1000 / 1000 << "M values "
              << "(" << problemSize * sizeof(float) / 1024 << "kiB)\n";

    std::ofstream plotFile{"vectoradd.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "vectoradd CPU {}Mi elements on {}"
set style data histograms
set style fill solid
#set key out top center maxrows 3
set yrange [0:*]
set ylabel "update runtime [s]"
$data << EOD
)",
        problemSize / 1024 / 1024,
        common::hostname());

    int r = 0;
    r += usellama::main(plotFile);
    r += manualAoS::main(plotFile);
    r += manualSoA::main(plotFile);
    r += manualAoSoA::main(plotFile);

    plotFile << R"(EOD
plot $data using 2:xtic(1) ti "runtime"
)";
    std::cout << "Plot with: ./vectoradd.sh\n";

    return r;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

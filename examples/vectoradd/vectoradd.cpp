#include "../common/Stopwatch.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <utility>

constexpr auto MAPPING = 2; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blobs), 3 tree AoS, 4 tree SoA
constexpr auto PROBLEM_SIZE = 64 * 1024 * 1024; ///< problem size
constexpr auto STEPS = 10; ///< number of vector adds to perform

using FP = float;

namespace usellama
{
    // clang-format off
    namespace tag
    {
        struct X{};
        struct Y{};
        struct Z{};
    }

    using Vector = llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>
    >;
    // clang-format on

    template <typename View>
    void add(const View& a, const View& b, View& c)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            c(i)(tag::X{}) = a(i)(tag::X{}) + b(i)(tag::X{});
            c(i)(tag::Y{}) = a(i)(tag::Y{}) - b(i)(tag::Y{});
            c(i)(tag::Z{}) = a(i)(tag::Z{}) * b(i)(tag::Z{});
        }
    }

    int main(std::ofstream& plotFile)
    {
        std::cout << "\nLLAMA\n";
        Stopwatch watch;

        const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};

        const auto mapping = [&] {
            if constexpr (MAPPING == 0)
                return llama::mapping::AoS{arrayDomain, Vector{}};
            if constexpr (MAPPING == 1)
                return llama::mapping::SoA{arrayDomain, Vector{}};
            if constexpr (MAPPING == 2)
                return llama::mapping::SoA{arrayDomain, Vector{}, std::true_type{}};
            if constexpr (MAPPING == 3)
                return llama::mapping::tree::Mapping{arrayDomain, llama::Tuple{}, Vector{}};
            if constexpr (MAPPING == 4)
                return llama::mapping::tree::Mapping{
                    arrayDomain,
                    llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                    Vector{}};
        }();

        auto a = allocView(mapping);
        auto b = allocView(mapping);
        auto c = allocView(mapping);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
        {
            a[i](tag::X{}) = i; // X
            a[i](tag::Y{}) = i; // Y
            a[i](llama::DatumCoord<2>{}) = i; // Z
            b(i) = i; // writes to all (X, Y, Z)
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            add(a, b, c);
            acc += watch.printAndReset("add");
        }
        plotFile << "LLAMA\t" << acc / STEPS << '\n';

        return (int) c.storageBlobs[0][0];
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
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            c[i].x = a[i].x + b[i].x;
            c[i].y = a[i].y - b[i].y;
            c[i].z = a[i].z * b[i].z;
        }
    }

    int main(std::ofstream& plotFile)
    {
        std::cout << "\nAoS\n";
        Stopwatch watch;

        std::vector<Vector> a(PROBLEM_SIZE);
        std::vector<Vector> b(PROBLEM_SIZE);
        std::vector<Vector> c(PROBLEM_SIZE);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
        {
            a[i].x = i;
            a[i].y = i;
            a[i].z = i;
            b[i].x = i;
            b[i].y = i;
            b[i].z = i;
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            add(a.data(), b.data(), c.data());
            acc += watch.printAndReset("add");
        }
        plotFile << "AoS\t" << acc / STEPS << '\n';

        return c[0].x;
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
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            cx[i] = ax[i] + bx[i];
            cy[i] = ay[i] - by[i];
            cz[i] = az[i] * bz[i];
        }
    }

    int main(std::ofstream& plotFile)
    {
        std::cout << "\nSoA\n";
        Stopwatch watch;

        using Vector = std::vector<float, llama::bloballoc::AlignedAllocator<float, 64>>;
        Vector ax(PROBLEM_SIZE);
        Vector ay(PROBLEM_SIZE);
        Vector az(PROBLEM_SIZE);
        Vector bx(PROBLEM_SIZE);
        Vector by(PROBLEM_SIZE);
        Vector bz(PROBLEM_SIZE);
        Vector cx(PROBLEM_SIZE);
        Vector cy(PROBLEM_SIZE);
        Vector cz(PROBLEM_SIZE);
        watch.printAndReset("alloc");

        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
        {
            ax[i] = i;
            ay[i] = i;
            az[i] = i;
            bx[i] = i;
            by[i] = i;
            bz[i] = i;
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            add(ax.data(), ay.data(), az.data(), bx.data(), by.data(), bz.data(), cx.data(), cy.data(), cz.data());
            acc += watch.printAndReset("add");
        }
        plotFile << "SoA\t" << acc / STEPS << '\n';

        return cx[0];
    }
} // namespace manualSoA


namespace manualAoSoA
{
    constexpr auto LANES = 16;

    struct alignas(64) VectorBlock
    {
        FP x[LANES];
        FP y[LANES];
        FP z[LANES];
    };

    constexpr auto BLOCKS = PROBLEM_SIZE / LANES;

    inline void add(const VectorBlock* a, const VectorBlock* b, VectorBlock* c)
    {
        for (std::size_t bi = 0; bi < PROBLEM_SIZE / LANES; bi++)
        {
// the unroll 1 is needed to prevent unrolling, which prevents vectorization in GCC
#pragma GCC unroll 1
            LLAMA_INDEPENDENT_DATA
            for (std::size_t i = 0; i < LANES; ++i)
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

    int main(std::ofstream& plotFile)
    {
        std::cout << "\nAoSoA\n";
        Stopwatch watch;

        std::vector<VectorBlock> a(BLOCKS);
        std::vector<VectorBlock> b(BLOCKS);
        std::vector<VectorBlock> c(BLOCKS);
        watch.printAndReset("alloc");

        for (std::size_t bi = 0; bi < PROBLEM_SIZE / LANES; ++bi)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t i = 0; i < LANES; ++i)
            {
                a[bi].x[i] = bi * LANES + i;
                a[bi].y[i] = bi * LANES + i;
                a[bi].z[i] = bi * LANES + i;
                b[bi].x[i] = bi * LANES + i;
                b[bi].y[i] = bi * LANES + i;
                b[bi].z[i] = bi * LANES + i;
            }
        }
        watch.printAndReset("init");

        double acc = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            add(a.data(), b.data(), c.data());
            acc += watch.printAndReset("add");
        }
        plotFile << "AoSoA\t" << acc / STEPS << '\n';

        return c[0].x[0];
    }
} // namespace manualAoSoA


int main()
{
    std::cout << PROBLEM_SIZE / 1000 / 1000 << "M values "
              << "(" << PROBLEM_SIZE * sizeof(float) / 1024 << "kiB)\n";

    std::ofstream plotFile{"vectoradd.tsv"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);

    int r = 0;
    r += usellama::main(plotFile);
    r += manualAoS::main(plotFile);
    r += manualSoA::main(plotFile);
    r += manualAoSoA::main(plotFile);

    std::cout << "Plot with: ./vectoradd.sh\n";
    std::ofstream{"vectoradd.sh"} << R"(#!/usr/bin/gnuplot -p
set style data histograms
set style fill solid
set key out top center maxrows 3
set yrange [0:*]
plot 'vectoradd.tsv' using 2:xtic(1)
)";

    return r;
}

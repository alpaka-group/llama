#include <chrono>
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

    int main(int argc, char** argv)
    {
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

        std::cout << PROBLEM_SIZE / 1000 / 1000 << " million vectors LLAMA\n";

        auto a = allocView(mapping);
        auto b = allocView(mapping);
        auto c = allocView(mapping);

        const auto start = std::chrono::high_resolution_clock::now();

        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
        {
            a[i](tag::X{}) = i; // X
            a[i](tag::Y{}) = i; // Y
            a[i](llama::DatumCoord<2>{}) = i; // Z
            b(i) = i; // writes to all (X, Y, Z)
        }

        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            const auto start = std::chrono::high_resolution_clock::now();
            add(a, b, c);
            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "add took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

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

    int main(int argc, char** argv)
    {
        std::cout << PROBLEM_SIZE / 1000 / 1000 << " million vectors AoS\n";

        std::vector<Vector> a(PROBLEM_SIZE);
        std::vector<Vector> b(PROBLEM_SIZE);
        std::vector<Vector> c(PROBLEM_SIZE);

        const auto start = std::chrono::high_resolution_clock::now();

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

        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            const auto start = std::chrono::high_resolution_clock::now();
            add(a.data(), b.data(), c.data());
            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "add took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

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

    int main(int argc, char** argv)
    {
        std::cout << PROBLEM_SIZE / 1000 / 1000 << " million vectors SoA\n";

        std::vector<FP> ax(PROBLEM_SIZE);
        std::vector<FP> ay(PROBLEM_SIZE);
        std::vector<FP> az(PROBLEM_SIZE);
        std::vector<FP> bx(PROBLEM_SIZE);
        std::vector<FP> by(PROBLEM_SIZE);
        std::vector<FP> bz(PROBLEM_SIZE);
        std::vector<FP> cx(PROBLEM_SIZE);
        std::vector<FP> cy(PROBLEM_SIZE);
        std::vector<FP> cz(PROBLEM_SIZE);

        const auto start = std::chrono::high_resolution_clock::now();

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

        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            const auto start = std::chrono::high_resolution_clock::now();
            add(ax.data(), ay.data(), az.data(), bx.data(), by.data(), bz.data(), cx.data(), cy.data(), cz.data());
            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "add took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        return cx[0];
    }
} // namespace manualSoA

int main(int argc, char** argv)
{
    int r = 0;
    r += usellama::main(argc, argv);
    r += manualAoS::main(argc, argv);
    r += manualSoA::main(argc, argv);
    return r;
}

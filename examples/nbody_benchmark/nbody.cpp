#include <chrono>
#include <fmt/core.h>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>
#include <vector>

constexpr auto PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto STEPS = 10; ///< number of steps to calculate

using FP = float;
constexpr FP EPS2 = 0.01f;

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>
    >>,
    llama::DE<tag::Vel, llama::DS<
        llama::DE<tag::X, FP>,
        llama::DE<tag::Y, FP>,
        llama::DE<tag::Z, FP>
    >>,
    llama::DE<tag::Mass, FP>
>;
// clang-format on

template <typename VirtualParticle>
LLAMA_FN_HOST_ACC_INLINE void pPInteraction(VirtualParticle p1, VirtualParticle p2, FP ts)
{
    auto dist = p1(tag::Pos{}) - p2(tag::Pos{});
    dist *= dist;
    const FP distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = 1.0f / std::sqrt(distSixth);
    const FP s = p2(tag::Mass{}) * invDistCube;
    dist *= s * ts;
    p1(tag::Vel{}) += dist;
}

template <typename View>
void update(View& particles, FP ts)
{
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
            pPInteraction(particles(j), particles(i), ts);
    }
}

template <typename View>
void move(View& particles, FP ts)
{
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * ts;
}

template <std::size_t Mapping, std::size_t Alignment>
void run()
{
    constexpr FP ts = 0.0001f;

    const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
    auto mapping = [&] {
        if constexpr (Mapping == 0)
            return llama::mapping::AoS{arrayDomain, Particle{}};
        if constexpr (Mapping == 1)
            return llama::mapping::SoA{arrayDomain, Particle{}};
        if constexpr (Mapping == 2)
            return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
    }();

    auto particles = llama::allocView(std::move(mapping), llama::allocator::Vector<Alignment>{});

    std::default_random_engine engine;
    std::normal_distribution<FP> dist(FP(0), FP(1));
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        auto p = particles(i);
        p(tag::Pos{}, tag::X{}) = dist(engine);
        p(tag::Pos{}, tag::Y{}) = dist(engine);
        p(tag::Pos{}, tag::Z{}) = dist(engine);
        p(tag::Vel{}, tag::X{}) = dist(engine) / FP(10);
        p(tag::Vel{}, tag::Y{}) = dist(engine) / FP(10);
        p(tag::Vel{}, tag::Z{}) = dist(engine) / FP(10);
        p(tag::Mass{}) = dist(engine) / FP(100);
    }

    std::chrono::nanoseconds acc{};
    for (std::size_t s = 0; s < STEPS; ++s)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        update(particles, ts);
        const auto stop = std::chrono::high_resolution_clock::now();
        acc += stop - start;
        move(particles, ts);
    }

    const auto mappingStr = Mapping == 0 ? "AoS" : Mapping == 1 ? "SoA" : "SoA.B";
    fmt::print("{:7}\t{:9}\t{:.5}\n", mappingStr, Alignment, std::chrono::duration<double>{acc / STEPS}.count());
}

int main()
{
    using namespace boost::mp11;

    fmt::print("{:7}\t{:9}\t{}\n", "Mapping", "Alignment", "Time [s]");

    // AoS
    mp_for_each<mp_list_c<std::size_t, 0>>([](auto m) {
        mp_for_each<mp_iota_c<15>>([](auto ae) {
            constexpr auto mapping = decltype(m)::value;
            constexpr auto alignment = std::size_t{1} << decltype(ae)::value;
            run<mapping, alignment>();
        });
    });

    // SoA single and multi blob
    mp_for_each<mp_list_c<std::size_t, 1, 2>>([](auto m) {
        mp_for_each<mp_iota_c<30>>([](auto ae) {
            constexpr auto mapping = decltype(m)::value;
            constexpr auto alignment = std::size_t{1} << decltype(ae)::value;
            run<mapping, alignment>();
        });
    });
}
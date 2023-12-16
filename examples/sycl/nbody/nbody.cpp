// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "../../common/Stopwatch.hpp"
#include "../../common/env.hpp"

#include <chrono>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <sycl/sycl.hpp>
#include <utility>
#include <vector>

using FP = float;

constexpr auto problemSize = 64 * 1024; ///< total number of particles
constexpr auto steps = 5; ///< number of steps to calculate
constexpr auto runUpate = true; // run update step. Useful to disable for benchmarking the move step.
constexpr auto allowRsqrt = true;
constexpr auto aosoaLanes = 8;

constexpr auto timestep = FP{0.0001};
constexpr auto eps2 = FP{0.01};

constexpr auto rngSeed = 42;

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

using Vec3 = llama::Record<
    llama::Field<tag::X, FP>,
    llama::Field<tag::Y, FP>,
    llama::Field<tag::Z, FP>>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Vel, Vec3>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

// using SharedMemoryParticle = Particle;
// using SharedMemoryParticle = ParticleJ;

template<typename ParticleRefI, typename ParticleRefJ>
void pPInteraction(ParticleRefI& pi, ParticleRefJ pj)
{
    auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
    dist *= dist;
    const FP distSqr = eps2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = allowRsqrt ? sycl::rsqrt(distSixth) : (1.0f / sycl::sqrt(distSixth));
    const FP sts = pj(tag::Mass{}) * invDistCube * +timestep;
    pi(tag::Vel{}) += dist * sts;
}

template<int Mapping>
int run(sycl::queue& queue)
{
    auto mappingName = [](int m) -> std::string
    {
        if(m == 0)
            return "AoS";
        if(m == 1)
            return "SoA MB";
        if(m == 2)
            return "AoSoA" + std::to_string(aosoaLanes);
        std::abort();
    };
    std::cout << "LLAMA " << mappingName(Mapping) << "\n";

    Stopwatch watch;
    auto mapping = [&]
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 1>;
        const auto extents = ArrayExtents{problemSize};
        if constexpr(Mapping == 0)
            return llama::mapping::AoS<ArrayExtents, Particle>{extents};
        if constexpr(Mapping == 1)
            return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::OnePerField>{extents};
        if constexpr(Mapping == 2)
            return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{extents};
    }();

    auto particles = llama::allocViewUninitialized(std::move(mapping), llama::bloballoc::SyclMallocShared{queue});
    watch.printAndReset("alloc");

    std::default_random_engine engine{rngSeed};
    std::normal_distribution<FP> dist(FP{0}, FP{1});
    for(std::size_t i = 0; i < problemSize; ++i)
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
    watch.printAndReset("init");
    for(int i = 0; i < particles.mapping().blobCount; i++)
        queue.prefetch(&particles.blobs()[i][0], particles.mapping().blobSize(i));
    watch.printAndReset("prefetch");

    const auto range = sycl::range<1>{problemSize};

    double sumUpdate = 0;
    double sumMove = 0;
    for(std::size_t s = 0; s < steps; ++s)
    {
        queue.submit(
            [&](sycl::handler& h)
            {
                h.parallel_for(
                    range,
                    [particles = llama::shallowCopy(particles)](sycl::item<1> it)
                    {
                        const auto i = it[0];
                        llama::One<Particle> pi = particles(i);
                        for(std::size_t j = 0; j < problemSize; j++)
                            pPInteraction(pi, particles(j));
                        particles(i)(tag::Vel{}) = pi(tag::Vel{});
                    });
            });
        queue.wait();
        sumUpdate += watch.printAndReset("update", '\t');
        queue.submit(
            [&](sycl::handler& h)
            {
                h.parallel_for(
                    range,
                    [particles = llama::shallowCopy(particles)](sycl::item<1> it)
                    { particles(it[0])(tag::Pos{}) += particles(it[0])(tag::Vel{}) * timestep; });
            });
        queue.wait();
        sumMove += watch.printAndReset("move");
    }

    return 0;
}

auto main(int argc, char** argv) -> int
try
{
    std::cout << problemSize / 1000 << "k particles "
              << "(" << problemSize * sizeof(float) * 7 / 1024 << "kiB)\n";

    auto queue = sycl::queue{sycl::cpu_selector_v};
    std::cout << "running on device: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << std::fixed;

    int r = 0;
    r += run<0>(queue);
    r += run<1>(queue);
    r += run<2>(queue);
    return r;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << "\n";
}

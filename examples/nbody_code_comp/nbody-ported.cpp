// Copyright 2024 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// clang-format off

#include "llama/llama.hpp"
#include <random>
#include <vector>

using FP = float;
constexpr FP timestep = 0.0001f, eps2 = 0.01f;
constexpr int steps = 5, problemSize = 64 * 1024;

struct Pos{}; struct Vel{}; struct X{}; struct Y{}; struct Z{}; struct Mass{};
using Vec3 = llama::Record<
    llama::Field<X, FP>,
    llama::Field<Y, FP>,
    llama::Field<Z, FP>
>;
using Particle = llama::Record<
    llama::Field<Pos, Vec3>,
    llama::Field<Vel, Vec3>,
    llama::Field<Mass, FP>
>;

LLAMA_FN_HOST_ACC_INLINE void pPInteraction(auto&& pi, auto&& pj) {
    auto dist = pi(Pos{}) - pj(Pos{});
    dist *= dist;
    const auto distSqr = eps2 + dist(X{}) + dist(Y{}) + dist(Z{});
    const auto distSixth = distSqr * distSqr * distSqr;
    const auto invDistCube = FP{1} / std::sqrt(distSixth);
    const auto sts = pj(Mass{}) * timestep * invDistCube;
    pi(Vel{}) += dist * sts;
}

void update(auto&& particles) {
    LLAMA_INDEPENDENT_DATA
    for(int i = 0; i < problemSize; i++) {
        llama::One<Particle> pi = particles[i];
        for(std::size_t j = 0; j < problemSize; ++j)
            pPInteraction(pi, particles[j]);
        particles[i](Vel{}) = pi(Vel{});
    }
}

void move(auto&& particles) {
    LLAMA_INDEPENDENT_DATA
    for(int i = 0; i < problemSize; i++)
        particles[i](Pos{}) += particles[i](Vel{}) * timestep;
}

auto main() -> int {
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
    const auto extents = ArrayExtents{problemSize};
    auto mapping = llama::mapping::AoS<ArrayExtents, Particle>{extents};
    auto particles = llama::allocViewUninitialized(mapping);

    std::default_random_engine engine;
    std::normal_distribution<FP> dist(FP{0}, FP{1});
    for(auto&& p : particles) {
        p(Pos{}, X{}) = dist(engine);
        p(Pos{}, Y{}) = dist(engine);
        p(Pos{}, Z{}) = dist(engine);
        p(Vel{}, X{}) = dist(engine) / FP{10};
        p(Vel{}, Y{}) = dist(engine) / FP{10};
        p(Vel{}, Z{}) = dist(engine) / FP{10};
        p(Mass{}) = dist(engine) / FP{100};
    }

    for(int s = 0; s < steps; ++s) {
        update(particles);
        ::move(particles);
    }

    return 0;
}

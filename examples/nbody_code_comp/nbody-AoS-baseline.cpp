// Copyright 2024 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// clang-format off

#include <random>
#include <span>
#include <vector>

using FP = float;
constexpr FP timestep = 0.0001f, eps2 = 0.01f;
constexpr int steps = 5, problemSize = 64 * 1024;

struct Vec3 {
    FP x, y, z;
};
struct Particle {
    Vec3 pos, vel;
    FP mass;
};

inline void pPInteraction(Particle& pi, const Particle& pj) {
    auto dist = Vec3{pi.pos.x - pj.pos.x, pi.pos.y - pj.pos.y, pi.pos.z - pj.pos.z};
    dist.x *= dist.x;
    dist.y *= dist.y;
    dist.z *= dist.z;
    const auto distSqr = eps2 + dist.x + dist.y + dist.z;
    const auto distSixth = distSqr * distSqr * distSqr;
    const auto invDistCube = FP{1} / std::sqrt(distSixth);
    const auto sts = pj.mass * timestep * invDistCube;
    pi.vel.x += dist.x * sts;
    pi.vel.y += dist.y * sts;
    pi.vel.z += dist.z * sts;
}

void update(std::span<Particle> particles) {
#pragma GCC ivdep
    for(int i = 0; i < problemSize; i++) {
        Particle pi = particles[i];
        for(int j = 0; j < problemSize; ++j)
            pPInteraction(pi, particles[j]);
        particles[i].vel = pi.vel;
    }
}

void move(std::span<Particle> particles) {
#pragma GCC ivdep
    for(int i = 0; i < problemSize; i++) {
        particles[i].pos.x += particles[i].vel.x * timestep;
        particles[i].pos.y += particles[i].vel.y * timestep;
        particles[i].pos.z += particles[i].vel.z * timestep;
    }
}

auto main() -> int {
    auto particles = std::vector<Particle>(problemSize);
	
    std::default_random_engine engine;
    std::normal_distribution<FP> dist(FP{0}, FP{1});
    for(auto& p : particles) {
        p.pos.x = dist(engine);
        p.pos.y = dist(engine);
        p.pos.z = dist(engine);
        p.vel.x = dist(engine) / FP{10};
        p.vel.y = dist(engine) / FP{10};
        p.vel.z = dist(engine) / FP{10};
        p.mass = dist(engine) / FP{100};
    }

    for(int s = 0; s < steps; ++s) {
        update(particles);
        ::move(particles);
    }

    return 0;
}

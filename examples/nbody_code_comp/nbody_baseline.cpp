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

    auto operator*=(Vec3 v) -> Vec3& {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    auto operator+=(Vec3 v) -> Vec3& {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    friend auto operator-(Vec3 a, Vec3 b) -> Vec3 {
        return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
    }

    friend auto operator*(Vec3 a, FP s) -> Vec3 {
        return Vec3{a.x * s, a.y * s, a.z * s};
    }
};

struct Particle {
    Vec3 pos, vel;
    FP mass;
};

inline void pPInteraction(Particle& pi, const Particle& pj) {
    auto dist = pi.pos - pj.pos;
    dist *= dist;
    const auto distSqr = eps2 + dist.x + dist.y + dist.z;
    const auto distSixth = distSqr * distSqr * distSqr;
    const auto invDistCube = FP{1} / std::sqrt(distSixth);
    const auto sts = pj.mass * timestep * invDistCube;
    pi.vel += dist * sts;
}

void update(std::span<Particle> particles) {
#pragma GCC ivdep
    for(int i = 0; i < problemSize; i++) {
        Particle pi = particles[i];
        for(std::size_t j = 0; j < problemSize; ++j)
            pPInteraction(pi, particles[j]);
        particles[i].vel = pi.vel;
    }
}

void move(std::span<Particle> particles) {
#pragma GCC ivdep
    for(int i = 0; i < problemSize; i++)
        particles[i].pos += particles[i].vel * timestep;
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

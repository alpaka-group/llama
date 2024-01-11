// Copyright 2024 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// clang-format off

#include <random>
#include <span>
#include <vector>

using FP = float;
constexpr FP timestep = 0.0001f, eps2 = 0.01f;
constexpr int steps = 5, problemSize = 64 * 1024;

constexpr auto lanes = 16;
constexpr auto blocks = problemSize / lanes;

struct alignas(lanes * sizeof(FP)) Vec3Block {
    FP x[lanes];
    FP y[lanes];
    FP z[lanes];
};
struct alignas(lanes * sizeof(FP)) ParticleBlock {
    Vec3Block pos, vel;
    FP mass[lanes];
};

inline void pPInteraction(FP piposx, FP piposy, FP piposz, FP& pivelx, FP& pively, FP& pivelz,
                          FP pjposx, FP pjposy, FP pjposz, FP pjmass) {
    auto xdist = piposx - pjposx;
    auto ydist = piposy - pjposy;
    auto zdist = piposz - pjposz;
    xdist *= xdist;
    ydist *= ydist;
    zdist *= zdist;
    const auto distSqr = eps2 + xdist + ydist + zdist;
    const auto distSixth = distSqr * distSqr * distSqr;
    const auto invDistCube = FP{1} / std::sqrt(distSixth);
    const auto sts = pjmass * timestep * invDistCube;
    pivelx += xdist * sts;
    pively += ydist * sts;
    pivelz += zdist * sts;
}

void update(std::span<ParticleBlock> particles) {
    for(int bi = 0; bi < blocks; bi++) {
        auto blockI = particles[bi];
        for(int bj = 0; bj < blocks; bj++)
            for(int j = 0; j < lanes; j++) {
#pragma GCC ivdep
                for(int i = 0; i < lanes; i++) {
                    pPInteraction(
                        blockI.pos.x[i],
                        blockI.pos.y[i],
                        blockI.pos.z[i],
                        blockI.vel.x[i],
                        blockI.vel.y[i],
                        blockI.vel.z[i],
                        particles[bj].pos.x[j],
                        particles[bj].pos.y[j],
                        particles[bj].pos.z[j],
                        particles[bj].mass[j]);
                }
            }

        particles[bi].vel = blockI.vel;
    }
}

void move(std::span<ParticleBlock> particles) {
    for(int bi = 0; bi < blocks; bi++) {
        #pragma GCC ivdep
        for(std::size_t i = 0; i < lanes; ++i) {
            particles[bi].pos.x[i] += particles[bi].vel.x[i] * timestep;
            particles[bi].pos.y[i] += particles[bi].vel.y[i] * timestep;
            particles[bi].pos.z[i] += particles[bi].vel.z[i] * timestep;
        }
    }
}

auto main() -> int {
    auto particles = std::vector<ParticleBlock>(blocks);
	
    std::default_random_engine engine;
    std::normal_distribution<FP> dist(FP{0}, FP{1});
    for(int bi = 0; bi < blocks; ++bi) {
        for(int i = 0; i < lanes; ++i) {
            particles[bi].pos.x[i] = dist(engine);
            particles[bi].pos.y[i] = dist(engine);
            particles[bi].pos.z[i] = dist(engine);
            particles[bi].vel.x[i] = dist(engine) / FP{10};
            particles[bi].vel.y[i] = dist(engine) / FP{10};
            particles[bi].vel.z[i] = dist(engine) / FP{10};
            particles[bi].mass[i] = dist(engine) / FP{100};
        }
    }

    for(int s = 0; s < steps; ++s) {
        update(particles);
        ::move(particles);
    }

    return 0;
}

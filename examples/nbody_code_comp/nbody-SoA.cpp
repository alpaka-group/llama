// Copyright 2024 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// clang-format off

#include <random>
#include <vector>

using FP = float;
constexpr FP timestep = 0.0001f, eps2 = 0.01f;
constexpr int steps = 5, problemSize = 64 * 1024;

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

void update(const FP* posx, const FP* posy, const FP* posz, FP* velx, FP* vely, FP* velz, const FP* mass) {
#pragma GCC ivdep
    for(int i = 0; i < problemSize; i++) {
        const auto piposx = posx[i];
        const auto piposy = posy[i];
        const auto piposz = posz[i];
        auto pivelx = velx[i];
        auto pively = vely[i];
        auto pivelz = velz[i];
        for(int j = 0; j < problemSize; ++j)
            pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, posx[j], posy[j], posz[j], mass[j]);
        velx[i] = pivelx;
        vely[i] = pively;
        velz[i] = pivelz;
    }
}

void move(FP* posx, FP* posy, FP* posz, const FP* velx, const FP* vely, const FP* velz) {
#pragma GCC ivdep
    for(int i = 0; i < problemSize; i++) {
        posx[i] += velx[i] * timestep;
        posy[i] += vely[i] * timestep;
        posz[i] += velz[i] * timestep;
    }
}

template<typename T>
struct AlignedAllocator {
    using value_type = T;

    auto allocate(std::size_t n) const -> T* {
        return new(std::align_val_t{64}) T[n];
    }

    void deallocate(T* p, std::size_t) const {
        ::operator delete[] (p, std::align_val_t{64});
    }
};

auto main() -> int {
    using Vector = std::vector<FP, AlignedAllocator<FP>>;
    auto posx = Vector(problemSize);
    auto posy = Vector(problemSize);
    auto posz = Vector(problemSize);
    auto velx = Vector(problemSize);
    auto vely = Vector(problemSize);
    auto velz = Vector(problemSize);
    auto mass = Vector(problemSize);
	
    std::default_random_engine engine;
    std::normal_distribution<FP> dist(FP{0}, FP{1});
    for(int i = 0; i < problemSize; ++i) {
        posx[i] = dist(engine);
        posy[i] = dist(engine);
        posz[i] = dist(engine);
        velx[i] = dist(engine) / FP{10};
        vely[i] = dist(engine) / FP{10};
        velz[i] = dist(engine) / FP{10};
        mass[i] = dist(engine) / FP{100};
    }

    for(int s = 0; s < steps; ++s) {
        update(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), mass.data());
        move(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data());
    }

    return 0;
}

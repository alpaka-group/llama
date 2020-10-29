#include <chrono>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>
#include <vector>

// needs -fno-math-errno, so std::sqrt() can be vectorized

constexpr auto MAPPING = 2; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blos), 3 tree AoS, 4 tree SoA
constexpr auto PROBLEM_SIZE = 16 * 1024; ///< total number of particles
constexpr auto STEPS = 5; ///< number of steps to calculate
constexpr auto TRACE = false;

using FP = float;
constexpr FP EPS2 = 0.01f;

namespace usellama
{
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
        p1(tag::Vel{}) += dist * s * ts;
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

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
        auto mapping = [&] {
            if constexpr (MAPPING == 0)
                return llama::mapping::AoS{arrayDomain, Particle{}};
            if constexpr (MAPPING == 1)
                return llama::mapping::SoA{arrayDomain, Particle{}};
            if constexpr (MAPPING == 2)
                return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
            if constexpr (MAPPING == 3)
                return llama::mapping::tree::Mapping{arrayDomain, llama::Tuple{}, Particle{}};
            if constexpr (MAPPING == 4)
                return llama::mapping::tree::Mapping{
                    arrayDomain,
                    llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                    Particle{}};
        }();

        auto tmapping = [&] {
            if constexpr (TRACE)
                return llama::mapping::Trace{std::move(mapping)};
            else
                return std::move(mapping);
        }();

        auto particles = llama::allocView(std::move(tmapping));

        std::cout << "LLAMA\n";

        const auto start = std::chrono::high_resolution_clock::now();
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

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

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(particles, ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles, ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace usellama

namespace manualAoS
{
    struct Vec
    {
        FP x;
        FP y;
        FP z;

        auto operator*=(FP s) -> Vec&
        {
            x *= s;
            y *= s;
            z *= s;
            return *this;
        }

        auto operator*=(Vec v) -> Vec&
        {
            x *= v.x;
            y *= v.y;
            z *= v.z;
            return *this;
        }

        auto operator+=(Vec v) -> Vec&
        {
            x += v.x;
            y += v.y;
            z += v.z;
            return *this;
        }

        auto operator-=(Vec v) -> Vec&
        {
            x -= v.x;
            y -= v.y;
            z -= v.z;
            return *this;
        }

        friend auto operator-(Vec a, Vec b) -> Vec
        {
            return a -= b;
        }

        friend auto operator*(Vec a, FP s) -> Vec
        {
            return a *= s;
        }

        friend auto operator*(Vec a, Vec b) -> Vec
        {
            return a *= b;
        }
    };

    using Pos = Vec;
    using Vel = Vec;

    struct Particle
    {
        Pos pos;
        Vel vel;
        FP mass;
    };

    inline void pPInteraction(Particle& p1, const Particle& p2, FP ts)
    {
        auto distance = p1.pos - p2.pos;
        distance *= distance;
        const FP distSqr = EPS2 + distance.x + distance.y + distance.z;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2.mass * invDistCube;
        distance *= s * ts;
        p1.vel += distance;
    }

    void update(Particle* particles, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(particles[j], particles[i], ts);
        }
    }

    void move(Particle* particles, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles[i].pos += particles[i].vel * ts;
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << "AoS\n";

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<Particle> particles(PROBLEM_SIZE);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            for (auto& p : particles)
            {
                p.pos.x = dist(engine);
                p.pos.y = dist(engine);
                p.pos.z = dist(engine);
                p.vel.x = dist(engine) / FP(10);
                p.vel.y = dist(engine) / FP(10);
                p.vel.z = dist(engine) / FP(10);
                p.mass = dist(engine) / FP(100);
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualAoS

namespace manualSoA
{
    inline void pPInteraction(
        FP p1posx,
        FP p1posy,
        FP p1posz,
        FP& p1velx,
        FP& p1vely,
        FP& p1velz,
        FP p2posx,
        FP p2posy,
        FP p2posz,
        FP p2mass,
        FP ts)
    {
        auto xdistance = p1posx - p2posx;
        auto ydistance = p1posy - p2posy;
        auto zdistance = p1posz - p2posz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2mass * invDistCube;
        p1velx += xdistance * s * ts;
        p1vely += ydistance * s * ts;
        p1velz += zdistance * s * ts;
    }

    void update(FP* posx, FP* posy, FP* posz, FP* velx, FP* vely, FP* velz, FP* mass, FP ts)
    {
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; j++)
                pPInteraction(
                    posx[j],
                    posy[j],
                    posz[j],
                    velx[j],
                    vely[j],
                    velz[j],
                    posx[i],
                    posy[i],
                    posz[i],
                    mass[i],
                    ts);
        }
    }

    void move(FP* posx, FP* posy, FP* posz, FP* velx, FP* vely, FP* velz, FP* mass, FP ts)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            posx[i] += velx[i] * ts;
            posy[i] += vely[i] * ts;
            posz[i] += velz[i] * ts;
        }
    }

    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << "SoA\n";

        const auto start = std::chrono::high_resolution_clock::now();
        using Vector = std::vector<FP, llama::allocator::AlignedAllocator<FP, 64>>;
        Vector posx(PROBLEM_SIZE);
        Vector posy(PROBLEM_SIZE);
        Vector posz(PROBLEM_SIZE);
        Vector velx(PROBLEM_SIZE);
        Vector vely(PROBLEM_SIZE);
        Vector velz(PROBLEM_SIZE);
        Vector mass(PROBLEM_SIZE);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
            {
                posx[i] = dist(engine);
                posy[i] = dist(engine);
                posz[i] = dist(engine);
                velx[i] = dist(engine) / FP(10);
                vely[i] = dist(engine) / FP(10);
                velz[i] = dist(engine) / FP(10);
                mass[i] = dist(engine) / FP(100);
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                update(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), mass.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), mass.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualSoA

namespace manualAoSoA
{
    constexpr auto LANES = 16;
    constexpr auto L1D_SIZE = 32 * 1024;

    struct alignas(64) ParticleBlock
    {
        struct
        {
            FP x[LANES];
            FP y[LANES];
            FP z[LANES];
        } pos;
        struct
        {
            FP x[LANES];
            FP y[LANES];
            FP z[LANES];
        } vel;
        FP mass[LANES];
    };

    constexpr auto BLOCKS_PER_TILE = 64; // L1D_SIZE / sizeof(ParticleBlock);
    constexpr auto BLOCKS = PROBLEM_SIZE / LANES;

    inline void pPInteraction(
        FP p1posx,
        FP p1posy,
        FP p1posz,
        FP& p1velx,
        FP& p1vely,
        FP& p1velz,
        FP p2posx,
        FP p2posy,
        FP p2posz,
        FP p2mass,
        FP ts)
    {
        auto xdistance = p1posx - p2posx;
        auto ydistance = p1posy - p2posy;
        auto zdistance = p1posz - p2posz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP s = p2mass * invDistCube;
        p1velx += xdistance * s * ts;
        p1vely += ydistance * s * ts;
        p1velz += zdistance * s * ts;
    }

    void update(ParticleBlock* particles, FP ts)
    {
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t i = 0; i < LANES; i++)
                {
                    LLAMA_INDEPENDENT_DATA
                    for (std::size_t j = 0; j < LANES; j++)
                    {
                        const auto& blockI = particles[bi];
                        auto& blockJ = particles[bj];
                        pPInteraction(
                            blockJ.pos.x[j],
                            blockJ.pos.y[j],
                            blockJ.pos.z[j],
                            blockJ.vel.x[j],
                            blockJ.vel.y[j],
                            blockJ.vel.z[j],
                            blockI.pos.x[i],
                            blockI.pos.y[i],
                            blockI.pos.z[i],
                            blockI.mass[i],
                            ts);
                    }
                }
    }

    void updateTiled(ParticleBlock* particles, FP ts)
    {
        for (std::size_t ti = 0; ti < BLOCKS / BLOCKS_PER_TILE; ti++)
            for (std::size_t tj = 0; tj < BLOCKS / BLOCKS_PER_TILE; tj++)
                for (std::size_t bi = 0; bi < BLOCKS_PER_TILE; bi++)
                    for (std::size_t bj = 0; bj < BLOCKS_PER_TILE; bj++)
                        for (std::size_t i = 0; i < LANES; i++)
                        {
                            LLAMA_INDEPENDENT_DATA
                            for (std::size_t j = 0; j < LANES; j++)
                            {
                                const auto& blockI = particles[ti * BLOCKS_PER_TILE + bi];
                                auto& blockJ = particles[tj * BLOCKS_PER_TILE + bj];
                                pPInteraction(
                                    blockJ.pos.x[j],
                                    blockJ.pos.y[j],
                                    blockJ.pos.z[j],
                                    blockJ.vel.x[j],
                                    blockJ.vel.y[j],
                                    blockJ.vel.z[j],
                                    blockI.pos.x[i],
                                    blockI.pos.y[i],
                                    blockI.pos.z[i],
                                    blockI.mass[i],
                                    ts);
                            }
                        }
    }

    void move(ParticleBlock* particles, FP ts)
    {
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& block = particles[bi];
            LLAMA_INDEPENDENT_DATA
            for (std::size_t i = 0; i < LANES; ++i)
            {
                block.pos.x[i] += block.vel.x[i] * ts;
                block.pos.y[i] += block.vel.y[i] * ts;
                block.pos.z[i] += block.vel.z[i] * ts;
            }
        }
    }

    template <bool Tiled>
    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << "AoSoA";
        if constexpr (Tiled)
            std::cout << " tiled";
        std::cout << "\n";

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<ParticleBlock> particles(BLOCKS);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            for (std::size_t bi = 0; bi < BLOCKS; ++bi)
            {
                auto& block = particles[bi];
                for (std::size_t i = 0; i < LANES; ++i)
                {
                    block.pos.x[i] = dist(engine);
                    block.pos.y[i] = dist(engine);
                    block.pos.z[i] = dist(engine);
                    block.vel.x[i] = dist(engine) / FP(10);
                    block.vel.y[i] = dist(engine) / FP(10);
                    block.vel.z[i] = dist(engine) / FP(10);
                    block.mass[i] = dist(engine) / FP(100);
                }
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                if constexpr (Tiled)
                    updateTiled(particles.data(), ts);
                else
                    update(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualAoSoA

#ifdef __AVX2__
#    include <immintrin.h>

namespace manualAoSoA_manualAVX
{
    constexpr auto LANES = sizeof(__m256) / sizeof(float);

    struct alignas(32) ParticleBlock
    {
        struct
        {
            float x[LANES];
            float y[LANES];
            float z[LANES];
        } pos;
        struct
        {
            float x[LANES];
            float y[LANES];
            float z[LANES];
        } vel;
        float mass[LANES];
    };

    constexpr auto BLOCKS = PROBLEM_SIZE / LANES;
    const __m256 vEPS2 = _mm256_broadcast_ss(&EPS2);

    inline void pPInteraction(
        __m256 p1posx,
        __m256 p1posy,
        __m256 p1posz,
        __m256& p1velx,
        __m256& p1vely,
        __m256& p1velz,
        __m256 p2posx,
        __m256 p2posy,
        __m256 p2posz,
        __m256 p2mass,
        __m256 ts)
    {
        const __m256 xdistance = _mm256_sub_ps(p1posx, p2posx);
        const __m256 ydistance = _mm256_sub_ps(p1posy, p2posy);
        const __m256 zdistance = _mm256_sub_ps(p1posz, p2posz);
        const __m256 xdistanceSqr = _mm256_mul_ps(xdistance, xdistance);
        const __m256 ydistanceSqr = _mm256_mul_ps(ydistance, ydistance);
        const __m256 zdistanceSqr = _mm256_mul_ps(zdistance, zdistance);
        const __m256 distSqr
            = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(vEPS2, xdistanceSqr), ydistanceSqr), zdistanceSqr);
        const __m256 distSixth = _mm256_mul_ps(_mm256_mul_ps(distSqr, distSqr), distSqr);
        const __m256 invDistCube = _mm256_rsqrt_ps(distSixth);
        const __m256 s = _mm256_mul_ps(p2mass, invDistCube);
        const __m256 sts = _mm256_mul_ps(s, ts);
        p1velx = _mm256_fmadd_ps(xdistanceSqr, sts, p1velx);
        p1vely = _mm256_fmadd_ps(ydistanceSqr, sts, p1vely);
        p1velz = _mm256_fmadd_ps(zdistanceSqr, sts, p1velz);
    }

    // update (read/write) 8 particles J based on the influence of 1 particle I
    void update8(ParticleBlock* particles, FP ts)
    {
        const auto vts = _mm256_broadcast_ss(&ts);
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t i = 0; i < LANES; i++)
                {
                    const auto& blockI = particles[bi];
                    const __m256 posxI = _mm256_broadcast_ss(&blockI.pos.x[i]);
                    const __m256 posyI = _mm256_broadcast_ss(&blockI.pos.y[i]);
                    const __m256 poszI = _mm256_broadcast_ss(&blockI.pos.z[i]);
                    const __m256 massI = _mm256_broadcast_ss(&blockI.mass[i]);

                    auto& blockJ = particles[bj];
                    __m256 p1velx = _mm256_load_ps(blockJ.vel.x);
                    __m256 p1vely = _mm256_load_ps(blockJ.vel.y);
                    __m256 p1velz = _mm256_load_ps(blockJ.vel.z);
                    pPInteraction(
                        _mm256_load_ps(blockJ.pos.x),
                        _mm256_load_ps(blockJ.pos.y),
                        _mm256_load_ps(blockJ.pos.z),
                        p1velx,
                        p1vely,
                        p1velz,
                        posxI,
                        posyI,
                        poszI,
                        massI,
                        vts);
                    _mm256_store_ps(blockJ.vel.x, p1velx);
                    _mm256_store_ps(blockJ.vel.y, p1vely);
                    _mm256_store_ps(blockJ.vel.z, p1velz);
                }
    }

    inline auto horizontalSum(__m256 v) -> float
    {
        // from:
        // http://jtdz-solenoids.com/stackoverflow_/questions/13879609/horizontal-sum-of-8-packed-32bit-floats/18616679#18616679
        const __m256 t1 = _mm256_hadd_ps(v, v);
        const __m256 t2 = _mm256_hadd_ps(t1, t1);
        const __m128 t3 = _mm256_extractf128_ps(t2, 1);
        const __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
        return _mm_cvtss_f32(t4);
    }

    // update (read/write) 1 particles J based on the influence of 8 particles I
    void update1(ParticleBlock* particles, FP ts)
    {
        const auto vts = _mm256_broadcast_ss(&ts);
        for (std::size_t bj = 0; bj < BLOCKS; bj++)
            for (std::size_t j = 0; j < LANES; j++)
            {
                auto& blockJ = particles[bj];
                const __m256 p1posx = _mm256_broadcast_ss(&blockJ.pos.x[j]);
                const __m256 p1posy = _mm256_broadcast_ss(&blockJ.pos.y[j]);
                const __m256 p1posz = _mm256_broadcast_ss(&blockJ.pos.z[j]);
                __m256 p1velx = _mm256_broadcast_ss(&blockJ.vel.x[j]);
                __m256 p1vely = _mm256_broadcast_ss(&blockJ.vel.y[j]);
                __m256 p1velz = _mm256_broadcast_ss(&blockJ.vel.z[j]);

                for (std::size_t bi = 0; bi < BLOCKS; bi++)
                {
                    const auto& blockI = particles[bi];
                    const __m256 posxI = _mm256_load_ps(blockI.pos.x);
                    const __m256 posyI = _mm256_load_ps(blockI.pos.y);
                    const __m256 poszI = _mm256_load_ps(blockI.pos.z);
                    const __m256 massI = _mm256_load_ps(blockI.mass);
                    pPInteraction(p1posx, p1posy, p1posz, p1velx, p1vely, p1velz, posxI, posyI, poszI, massI, vts);
                }

                blockJ.vel.x[j] = horizontalSum(p1velx);
                blockJ.vel.y[j] = horizontalSum(p1vely);
                blockJ.vel.z[j] = horizontalSum(p1velz);
            }
    }

    void move(ParticleBlock* particles, FP ts)
    {
        const auto vts = _mm256_broadcast_ss(&ts);
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& block = particles[bi];
            _mm256_store_ps(
                block.pos.x,
                _mm256_fmadd_ps(_mm256_load_ps(block.vel.x), vts, _mm256_load_ps(block.pos.x)));
            _mm256_store_ps(
                block.pos.y,
                _mm256_fmadd_ps(_mm256_load_ps(block.vel.y), vts, _mm256_load_ps(block.pos.y)));
            _mm256_store_ps(
                block.pos.z,
                _mm256_fmadd_ps(_mm256_load_ps(block.vel.z), vts, _mm256_load_ps(block.pos.z)));
        }
    }

    template <bool UseUpdate1>
    int main(int argc, char** argv)
    {
        constexpr FP ts = 0.0001f;

        std::cout << (UseUpdate1 ? "AoSoA AVX2 updating 1 particle from 8\n" : "AoSoA AVX2 updating 8 particles from 1\n ");

        const auto start = std::chrono::high_resolution_clock::now();
        std::vector<ParticleBlock> particles(BLOCKS);
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "alloc took " << std::chrono::duration<double>(stop - start).count() << "s\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            std::default_random_engine engine;
            std::normal_distribution<FP> dist(FP(0), FP(1));
            for (std::size_t bi = 0; bi < BLOCKS; ++bi)
            {
                auto& block = particles[bi];
                for (std::size_t i = 0; i < LANES; ++i)
                {
                    block.pos.x[i] = dist(engine);
                    block.pos.y[i] = dist(engine);
                    block.pos.z[i] = dist(engine);
                    block.vel.x[i] = dist(engine) / FP(10);
                    block.vel.y[i] = dist(engine) / FP(10);
                    block.vel.z[i] = dist(engine) / FP(10);
                    block.mass[i] = dist(engine) / FP(100);
                }
            }

            const auto stop = std::chrono::high_resolution_clock::now();
            std::cout << "init took " << std::chrono::duration<double>(stop - start).count() << "s\n";
        }

        for (std::size_t s = 0; s < STEPS; ++s)
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();
                if constexpr (UseUpdate1)
                    update1(particles.data(), ts);
                else
                    update8(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "update took " << std::chrono::duration<double>(stop - start).count() << "s\t";
            }
            {
                const auto start = std::chrono::high_resolution_clock::now();
                move(particles.data(), ts);
                const auto stop = std::chrono::high_resolution_clock::now();
                std::cout << "move took   " << std::chrono::duration<double>(stop - start).count() << "s\n";
            }
        }

        return 0;
    }
} // namespace manualAoSoA_manualAVX
#endif

int main(int argc, char** argv)
{
    std::cout << PROBLEM_SIZE / 1000 << "k particles "
              << "(" << PROBLEM_SIZE * sizeof(float) * 7 / 1024 << "kiB)\n";

    int r = 0;
    r += usellama::main(argc, argv);
    r += manualAoS::main(argc, argv);
    r += manualSoA::main(argc, argv);
    r += manualAoSoA::main<false>(argc, argv);
    r += manualAoSoA::main<true>(argc, argv);
#ifdef __AVX2__
    r += manualAoSoA_manualAVX::main<false>(argc, argv);
    r += manualAoSoA_manualAVX::main<true>(argc, argv);
#endif
    return r;
}

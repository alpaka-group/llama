#include "../common/Stopwatch.hpp"

#include <boost/asio/ip/host_name.hpp>
#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>
#include <random>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

// needs -fno-math-errno, so std::sqrt() can be vectorized
// for multithreading, specify thread affinity (GNU OpenMP):
// e.g. for a 32 core CPU with SMT/hyperthreading: GOMP_CPU_AFFINITY='0-30:2,1-31:2' llama-nbody
// e.g. for a 16 core CPU without SMT/hyperthreading: GOMP_CPU_AFFINITY='0-15' llama-nbody

using FP = float;

constexpr auto PROBLEM_SIZE = 16 * 1024;
constexpr auto STEPS = 5;
constexpr auto TRACE = false;
constexpr auto HEATMAP = false;
constexpr auto DUMP_MAPPING = false;
constexpr auto ALLOW_RSQRT = true; // rsqrt can be way faster, but less accurate
constexpr auto NEWTON_RAPHSON_AFTER_RSQRT
    = true; // generate a newton raphson refinement after explicit calls to rsqrt()
constexpr FP TIMESTEP = 0.0001f;
constexpr FP EPS2 = 0.01f;

constexpr auto L1D_SIZE = 32 * 1024;
constexpr auto L2D_SIZE = 512 * 1024;

using namespace std::string_literals;

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
    } // namespace tag

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

    template <typename VirtualParticleI, typename VirtualParticleJ>
    LLAMA_FN_HOST_ACC_INLINE void pPInteraction(VirtualParticleI&& pi, VirtualParticleJ pj)
    {
        auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
        dist *= dist;
        const FP distSqr = EPS2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP sts = pj(tag::Mass{}) * invDistCube * TIMESTEP;
        pi(tag::Vel{}) += dist * sts;
    }

    template <bool UseAccumulator, typename View>
    void update(View& particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            llama::One<Particle> pi;
            if constexpr (UseAccumulator)
                pi = particles(i);
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; ++j)
            {
                if constexpr (UseAccumulator)
                    pPInteraction(pi, particles(j));
                else
                    pPInteraction(particles(i), particles(j));
            }
            if constexpr (UseAccumulator)
                particles(i) = pi;
        }
    }

    template <typename View>
    void move(View& particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * TIMESTEP;
    }

    template <int Mapping, bool UseAccumulator, std::size_t AoSoALanes = 8 /*AVX2*/>
    auto main(std::ostream& plotFile) -> int
    {
        auto mappingName = [](int m) -> std::string {
            if (m == 0)
                return "AoS";
            if (m == 1)
                return "SoA";
            if (m == 2)
                return "SoA MB";
            if (m == 3)
                return "AoSoA" + std::to_string(AoSoALanes);
            if (m == 4)
                return "Split SoA";
            std::abort();
        };
        auto title = "LLAMA " + mappingName(Mapping);
        if (UseAccumulator)
            title += " Acc";
        std::cout << title << "\n";
        Stopwatch watch;
        auto mapping = [&] {
            const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
            if constexpr (Mapping == 0)
                return llama::mapping::AoS{arrayDomain, Particle{}};
            if constexpr (Mapping == 1)
                return llama::mapping::SoA{arrayDomain, Particle{}};
            if constexpr (Mapping == 2)
                return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
            if constexpr (Mapping == 3)
                return llama::mapping::AoSoA<decltype(arrayDomain), Particle, AoSoALanes>{arrayDomain};
            if constexpr (Mapping == 4)
                return llama::mapping::Split<
                    decltype(arrayDomain),
                    Particle,
                    llama::DatumCoord<1>,
                    llama::mapping::SoA,
                    llama::mapping::SoA,
                    true>{arrayDomain};
        }();
        if constexpr (DUMP_MAPPING)
            std::ofstream{title + ".svg"} << llama::toSvg(mapping);

        auto tmapping = [&] {
            if constexpr (TRACE)
                return llama::mapping::Trace{std::move(mapping)};
            else
                return std::move(mapping);
        }();

        auto hmapping = [&] {
            if constexpr (HEATMAP)
                return llama::mapping::Heatmap{std::move(tmapping)};
            else
                return std::move(tmapping);
        }();

        auto particles = llama::allocView(std::move(hmapping));
        watch.printAndReset("alloc");

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
        watch.printAndReset("init");

        double sumUpdate = 0;
        double sumMove = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            update<UseAccumulator>(particles);
            sumUpdate += watch.printAndReset("update", '\t');
            move(particles);
            sumMove += watch.printAndReset("move");
        }
        if (!UseAccumulator)
            plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\t';
        else
            plotFile << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

        if constexpr (HEATMAP)
            std::ofstream{"nbody_heatmap_" + mappingName(Mapping) + ".dat"} << particles.mapping.toGnuplotDatFile();

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

    inline void pPInteraction(Particle& pi, const Particle& pj)
    {
        auto distance = pi.pos - pj.pos;
        distance *= distance;
        const FP distSqr = EPS2 + distance.x + distance.y + distance.z;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP sts = pj.mass * invDistCube * TIMESTEP;
        pi.vel += distance * sts;
    }

    template <bool UseAccumulator>
    void update(Particle* particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            Particle pi{};
            if constexpr (UseAccumulator)
                pi = particles[i];
            LLAMA_INDEPENDENT_DATA
            for (std::size_t j = 0; j < PROBLEM_SIZE; ++j)
            {
                if constexpr (UseAccumulator)
                    pPInteraction(pi, particles[j]);
                else
                    pPInteraction(particles[i], particles[j]);
            }
            if constexpr (UseAccumulator)
                particles[i] = pi;
        }
    }

    void move(Particle* particles)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
            particles[i].pos += particles[i].vel * TIMESTEP;
    }

    template <bool UseAccumulator>
    auto main(std::ostream& plotFile) -> int
    {
        auto title = "AoS"s;
        if (UseAccumulator)
            title += " Acc";
        std::cout << title << "\n";
        Stopwatch watch;

        std::vector<Particle> particles(PROBLEM_SIZE);
        watch.printAndReset("alloc");

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
        watch.printAndReset("init");

        double sumUpdate = 0;
        double sumMove = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            update<UseAccumulator>(particles.data());
            sumUpdate += watch.printAndReset("update", '\t');
            move(particles.data());
            sumMove += watch.printAndReset("move");
        }
        if (!UseAccumulator)
            plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\t';
        else
            plotFile << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

        return 0;
    }
} // namespace manualAoS

namespace manualSoA
{
    inline void pPInteraction(
        FP piposx,
        FP piposy,
        FP piposz,
        FP& pivelx,
        FP& pively,
        FP& pivelz,
        FP pjposx,
        FP pjposy,
        FP pjposz,
        FP pjmass)
    {
        auto xdistance = piposx - pjposx;
        auto ydistance = piposy - pjposy;
        auto zdistance = piposz - pjposz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP sts = pjmass * invDistCube * TIMESTEP;
        pivelx += xdistance * sts;
        pively += ydistance * sts;
        pivelz += zdistance * sts;
    }

    template <bool UseAccumulator>
    void update(FP* posx, FP* posy, FP* posz, FP* velx, FP* vely, FP* velz, FP* mass)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            FP piposx = 0;
            FP piposy = 0;
            FP piposz = 0;
            FP pivelx = 0;
            FP pively = 0;
            FP pivelz = 0;
            if constexpr (UseAccumulator)
            {
                piposx = posx[i];
                piposy = posy[i];
                piposz = posz[i];
                pivelx = velx[i];
                pively = vely[i];
                pivelz = velz[i];
            }
            for (std::size_t j = 0; j < PROBLEM_SIZE; ++j)
            {
                if constexpr (UseAccumulator)
                    pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, posx[j], posy[j], posz[j], mass[j]);
                else
                    pPInteraction(
                        posx[i],
                        posy[i],
                        posz[i],
                        velx[i],
                        vely[i],
                        velz[i],
                        posx[j],
                        posy[j],
                        posz[j],
                        mass[j]);
            }
            if constexpr (UseAccumulator)
            {
                velx[i] = pivelx;
                vely[i] = pively;
                velz[i] = pivelz;
            }
        }
    }

    void move(FP* posx, FP* posy, FP* posz, const FP* velx, const FP* vely, const FP* velz)
    {
        LLAMA_INDEPENDENT_DATA
        for (std::size_t i = 0; i < PROBLEM_SIZE; i++)
        {
            posx[i] += velx[i] * TIMESTEP;
            posy[i] += vely[i] * TIMESTEP;
            posz[i] += velz[i] * TIMESTEP;
        }
    }

    template <bool UseAccumulator>
    auto main(std::ostream& plotFile) -> int
    {
        auto title = "SoA"s;
        if (UseAccumulator)
            title += " Acc";
        std::cout << title << "\n";
        Stopwatch watch;

        using Vector = std::vector<FP, llama::bloballoc::AlignedAllocator<FP, 64>>;
        Vector posx(PROBLEM_SIZE);
        Vector posy(PROBLEM_SIZE);
        Vector posz(PROBLEM_SIZE);
        Vector velx(PROBLEM_SIZE);
        Vector vely(PROBLEM_SIZE);
        Vector velz(PROBLEM_SIZE);
        Vector mass(PROBLEM_SIZE);
        watch.printAndReset("alloc");

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
        watch.printAndReset("init");

        double sumUpdate = 0;
        double sumMove = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            update<UseAccumulator>(
                posx.data(),
                posy.data(),
                posz.data(),
                velx.data(),
                vely.data(),
                velz.data(),
                mass.data());
            sumUpdate += watch.printAndReset("update", '\t');
            move(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data());
            sumMove += watch.printAndReset("move");
        }
        if (!UseAccumulator)
            plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\t';
        else
            plotFile << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

        return 0;
    }
} // namespace manualSoA

namespace manualAoSoA
{
    template <std::size_t Lanes>
    struct alignas(64) ParticleBlock
    {
        struct
        {
            FP x[Lanes];
            FP y[Lanes];
            FP z[Lanes];
        } pos;
        struct
        {
            FP x[Lanes];
            FP y[Lanes];
            FP z[Lanes];
        } vel;
        FP mass[Lanes];
    };

    inline void pPInteraction(
        FP piposx,
        FP piposy,
        FP piposz,
        FP& pivelx,
        FP& pively,
        FP& pivelz,
        FP pjposx,
        FP pjposy,
        FP pjposz,
        FP pjmass)
    {
        auto xdistance = piposx - pjposx;
        auto ydistance = piposy - pjposy;
        auto zdistance = piposz - pjposz;
        xdistance *= xdistance;
        ydistance *= ydistance;
        zdistance *= zdistance;
        const FP distSqr = EPS2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP sts = pjmass * invDistCube * TIMESTEP;
        pivelx += xdistance * sts;
        pively += ydistance * sts;
        pivelz += zdistance * sts;
    }

    template <bool UseAccumulator, std::size_t Lanes>
    void update(ParticleBlock<Lanes>* particles)
    {
        constexpr auto blocks = PROBLEM_SIZE / Lanes;
        for (std::size_t bi = 0; bi < blocks; bi++)
        {
            std::conditional_t<UseAccumulator, ParticleBlock<Lanes>, ParticleBlock<Lanes>&> blockI = particles[bi];
            for (std::size_t bj = 0; bj < blocks; bj++)
                for (std::size_t j = 0; j < Lanes; j++)
                {
                    const auto& blockJ = particles[bj];
                    LLAMA_INDEPENDENT_DATA
                    for (std::size_t i = 0; i < Lanes; i++)
                    {
                        pPInteraction(
                            blockI.pos.x[i],
                            blockI.pos.y[i],
                            blockI.pos.z[i],
                            blockI.vel.x[i],
                            blockI.vel.y[i],
                            blockI.vel.z[i],
                            blockJ.pos.x[j],
                            blockJ.pos.y[j],
                            blockJ.pos.z[j],
                            blockJ.mass[j]);
                    }
                }
            if constexpr (UseAccumulator)
                particles[bi] = blockI;
        }
    }

    template <bool UseAccumulator, std::size_t Lanes>
    void updateTiled(ParticleBlock<Lanes>* particles)
    {
        constexpr auto blocks = PROBLEM_SIZE / Lanes;
        constexpr auto blocksPerTile = 128; // L1D_SIZE / sizeof(ParticleBlock<Lanes>);
        static_assert(blocks % blocksPerTile == 0);
        for (std::size_t ti = 0; ti < blocks / blocksPerTile; ti++)
            for (std::size_t tj = 0; tj < blocks / blocksPerTile; tj++)
                for (std::size_t bi = 0; bi < blocksPerTile; bi++)
                    for (std::size_t bj = 0; bj < blocksPerTile; bj++)
                        for (std::size_t j = 0; j < Lanes; j++)
                        {
                            LLAMA_INDEPENDENT_DATA
                            for (std::size_t i = 0; i < Lanes; i++)
                            {
                                auto& blockI = particles[ti * blocksPerTile + bi];
                                const auto& blockJ = particles[tj * blocksPerTile + bj];
                                pPInteraction(
                                    blockI.pos.x[i],
                                    blockI.pos.y[i],
                                    blockI.pos.z[i],
                                    blockI.vel.x[i],
                                    blockI.vel.y[i],
                                    blockI.vel.z[i],
                                    blockJ.pos.x[j],
                                    blockJ.pos.y[j],
                                    blockJ.pos.z[j],
                                    blockJ.mass[j]);
                            }
                        }
    }

    template <std::size_t Lanes>
    void move(ParticleBlock<Lanes>* particles)
    {
        constexpr auto blocks = PROBLEM_SIZE / Lanes;
        for (std::size_t bi = 0; bi < blocks; bi++)
        {
            LLAMA_INDEPENDENT_DATA
            for (std::size_t i = 0; i < Lanes; ++i)
            {
                auto& block = particles[bi];
                block.pos.x[i] += block.vel.x[i] * TIMESTEP;
                block.pos.y[i] += block.vel.y[i] * TIMESTEP;
                block.pos.z[i] += block.vel.z[i] * TIMESTEP;
            }
        }
    }

    template <bool UseAccumulator, bool Tiled, std::size_t Lanes>
    auto main(std::ostream& plotFile) -> int
    {
        auto title = "AoSoA" + std::to_string(Lanes);
        if (Tiled)
            title += " tiled";
        if (UseAccumulator)
            title += " Acc";
        std::cout << title << "\n";
        Stopwatch watch;

        constexpr auto blocks = PROBLEM_SIZE / Lanes;

        std::vector<ParticleBlock<Lanes>> particles(blocks);
        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> dist(FP(0), FP(1));
        for (std::size_t bi = 0; bi < blocks; ++bi)
        {
            auto& block = particles[bi];
            for (std::size_t i = 0; i < Lanes; ++i)
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
        watch.printAndReset("init");

        double sumUpdate = 0;
        double sumMove = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            if constexpr (Tiled)
                updateTiled<UseAccumulator>(particles.data());
            else
                update<UseAccumulator>(particles.data());
            sumUpdate += watch.printAndReset("update", '\t');
            move(particles.data());
            sumMove += watch.printAndReset("move");
        }
        if (!UseAccumulator)
            plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\t';
        else
            plotFile << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

        return 0;
    }
} // namespace manualAoSoA

#ifdef __AVX2__
#    include <immintrin.h>

namespace manualAoSoA_manualAVX
{
    // hard coded to AVX2 register length, should be 8
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
    const __m256 vEPS2 = _mm256_set1_ps(EPS2); // NOLINT(cert-err58-cpp)
    const __m256 vTIMESTEP = _mm256_broadcast_ss(&TIMESTEP); // NOLINT(cert-err58-cpp)

    inline void pPInteraction(
        __m256 piposx,
        __m256 piposy,
        __m256 piposz,
        __m256& pivelx,
        __m256& pively,
        __m256& pivelz,
        __m256 pjposx,
        __m256 pjposy,
        __m256 pjposz,
        __m256 pjmass)
    {
        const __m256 xdistance = _mm256_sub_ps(piposx, pjposx);
        const __m256 ydistance = _mm256_sub_ps(piposy, pjposy);
        const __m256 zdistance = _mm256_sub_ps(piposz, pjposz);
        const __m256 xdistanceSqr = _mm256_mul_ps(xdistance, xdistance);
        const __m256 ydistanceSqr = _mm256_mul_ps(ydistance, ydistance);
        const __m256 zdistanceSqr = _mm256_mul_ps(zdistance, zdistance);
        const __m256 distSqr
            = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(vEPS2, xdistanceSqr), ydistanceSqr), zdistanceSqr);
        const __m256 distSixth = _mm256_mul_ps(_mm256_mul_ps(distSqr, distSqr), distSqr);
        const __m256 invDistCube = [&] {
            if constexpr (ALLOW_RSQRT)
            {
                const __m256 r = _mm256_rsqrt_ps(distSixth);
                if constexpr (NEWTON_RAPHSON_AFTER_RSQRT)
                {
                    // from: http://stackoverflow.com/q/14752399/556899
                    const __m256 three = _mm256_set1_ps(3.0f);
                    const __m256 half = _mm256_set1_ps(0.5f);
                    const __m256 muls = _mm256_mul_ps(_mm256_mul_ps(distSixth, r), r);
                    return _mm256_mul_ps(_mm256_mul_ps(half, r), _mm256_sub_ps(three, muls));
                }
                else
                    return r;
            }
            else
                return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(distSixth));
        }();
        const __m256 sts = _mm256_mul_ps(_mm256_mul_ps(pjmass, invDistCube), vTIMESTEP);
        pivelx = _mm256_fmadd_ps(xdistanceSqr, sts, pivelx);
        pively = _mm256_fmadd_ps(ydistanceSqr, sts, pively);
        pivelz = _mm256_fmadd_ps(zdistanceSqr, sts, pivelz);
    }

    // update (read/write) 8 particles I based on the influence of 1 particle J
    void update8(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t j = 0; j < LANES; j++)
                {
                    const auto& blockJ = particles[bj];
                    const __m256 pjposx = _mm256_broadcast_ss(&blockJ.pos.x[j]);
                    const __m256 pjposy = _mm256_broadcast_ss(&blockJ.pos.y[j]);
                    const __m256 pjposz = _mm256_broadcast_ss(&blockJ.pos.z[j]);
                    const __m256 pjmass = _mm256_broadcast_ss(&blockJ.mass[j]);

                    auto& blockI = particles[bi];
                    __m256 pivelx = _mm256_load_ps(&blockI.vel.x[0]);
                    __m256 pively = _mm256_load_ps(&blockI.vel.y[0]);
                    __m256 pivelz = _mm256_load_ps(&blockI.vel.z[0]);
                    pPInteraction(
                        _mm256_load_ps(&blockI.pos.x[0]),
                        _mm256_load_ps(&blockI.pos.y[0]),
                        _mm256_load_ps(&blockI.pos.z[0]),
                        pivelx,
                        pively,
                        pivelz,
                        pjposx,
                        pjposy,
                        pjposz,
                        pjmass);
                    _mm256_store_ps(&blockI.vel.x[0], pivelx);
                    _mm256_store_ps(&blockI.vel.y[0], pively);
                    _mm256_store_ps(&blockI.vel.z[0], pivelz);
                }
    }

    // update (read/write) 8 particles I based on the influence of 1 particle J with accumulator
    void update8Acc(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& blockI = particles[bi];
            const __m256 piposx = _mm256_load_ps(&blockI.pos.x[0]);
            const __m256 piposy = _mm256_load_ps(&blockI.pos.y[0]);
            const __m256 piposz = _mm256_load_ps(&blockI.pos.z[0]);
            __m256 pivelx = _mm256_load_ps(&blockI.vel.x[0]);
            __m256 pively = _mm256_load_ps(&blockI.vel.y[0]);
            __m256 pivelz = _mm256_load_ps(&blockI.vel.z[0]);

            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t j = 0; j < LANES; j++)
                {
                    const auto& blockJ = particles[bj];
                    const __m256 posxJ = _mm256_broadcast_ss(&blockJ.pos.x[j]);
                    const __m256 posyJ = _mm256_broadcast_ss(&blockJ.pos.y[j]);
                    const __m256 poszJ = _mm256_broadcast_ss(&blockJ.pos.z[j]);
                    const __m256 massJ = _mm256_broadcast_ss(&blockJ.mass[j]);
                    pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, posxJ, posyJ, poszJ, massJ);
                }

            _mm256_store_ps(&blockI.vel.x[0], pivelx);
            _mm256_store_ps(&blockI.vel.y[0], pively);
            _mm256_store_ps(&blockI.vel.z[0], pivelz);
        }
    }

    inline auto horizontalSum(__m256 v) -> float
    {
        // from:
        // http://jtdz-solenoids.com/stackoverflow_/questions/13879609/horizontal-sum-of-8-packed-32bit-floats/18616679#18616679
        const __m256 t1 = _mm256_hadd_ps(v, v);
        const __m256 t2 = _mm256_hadd_ps(t1, t1);
        const __m128 t3 = _mm256_extractf128_ps(t2, 1); // NOLINT(hicpp-use-auto, modernize-use-auto)
        const __m128 t4 = _mm_add_ss(_mm256_castps256_ps128(t2), t3);
        return _mm_cvtss_f32(t4);

        // alignas(32) float a[LANES];
        //_mm256_store_ps(a, v);
        // return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
    }

    // update (read/write) 1 particles I based on the influence of 8 particles J
    void update1(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t i = 0; i < LANES; i++)
                {
                    auto& blockI = particles[bi];
                    const __m256 piposx = _mm256_broadcast_ss(&blockI.pos.x[i]);
                    const __m256 piposy = _mm256_broadcast_ss(&blockI.pos.y[i]);
                    const __m256 piposz = _mm256_broadcast_ss(&blockI.pos.z[i]);
                    __m256 pivelx = _mm256_broadcast_ss(&blockI.vel.x[i]);
                    __m256 pively = _mm256_broadcast_ss(&blockI.vel.y[i]);
                    __m256 pivelz = _mm256_broadcast_ss(&blockI.vel.z[i]);

                    const auto& blockJ = particles[bj];
                    const __m256 pjposx = _mm256_load_ps(&blockJ.pos.x[0]);
                    const __m256 pjposy = _mm256_load_ps(&blockJ.pos.y[0]);
                    const __m256 pjposz = _mm256_load_ps(&blockJ.pos.z[0]);
                    const __m256 pjmass = _mm256_load_ps(&blockJ.mass[0]);
                    pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);

                    blockI.vel.x[i] = horizontalSum(pivelx);
                    blockI.vel.y[i] = horizontalSum(pively);
                    blockI.vel.z[i] = horizontalSum(pivelz);
                }
    }

    // update (read/write) 1 particles I based on the influence of 8 particles J with accumulator
    void update1Acc(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
            for (std::size_t i = 0; i < LANES; i++)
            {
                auto& blockI = particles[bi];
                const __m256 piposx = _mm256_broadcast_ss(&blockI.pos.x[i]);
                const __m256 piposy = _mm256_broadcast_ss(&blockI.pos.y[i]);
                const __m256 piposz = _mm256_broadcast_ss(&blockI.pos.z[i]);
                __m256 pivelx = _mm256_broadcast_ss(&blockI.vel.x[i]);
                __m256 pively = _mm256_broadcast_ss(&blockI.vel.y[i]);
                __m256 pivelz = _mm256_broadcast_ss(&blockI.vel.z[i]);

                for (std::size_t bj = 0; bj < BLOCKS; bj++)
                {
                    const auto& blockJ = particles[bj];
                    const __m256 pjposx = _mm256_load_ps(&blockJ.pos.x[0]);
                    const __m256 pjposy = _mm256_load_ps(&blockJ.pos.y[0]);
                    const __m256 pjposz = _mm256_load_ps(&blockJ.pos.z[0]);
                    const __m256 pjmass = _mm256_load_ps(&blockJ.mass[0]);
                    pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);
                }

                blockI.vel.x[i] = horizontalSum(pivelx);
                blockI.vel.y[i] = horizontalSum(pively);
                blockI.vel.z[i] = horizontalSum(pivelz);
            }
    }

    void move(ParticleBlock* particles)
    {
        for (std::size_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& block = particles[bi];
            _mm256_store_ps(
                &block.pos.x[0],
                _mm256_fmadd_ps(_mm256_load_ps(&block.vel.x[0]), vTIMESTEP, _mm256_load_ps(&block.pos.x[0])));
            _mm256_store_ps(
                &block.pos.y[0],
                _mm256_fmadd_ps(_mm256_load_ps(&block.vel.y[0]), vTIMESTEP, _mm256_load_ps(&block.pos.y[0])));
            _mm256_store_ps(
                &block.pos.z[0],
                _mm256_fmadd_ps(_mm256_load_ps(&block.vel.z[0]), vTIMESTEP, _mm256_load_ps(&block.pos.z[0])));
        }
    }

    template <bool UseAccumulator, bool UseUpdate1>
    auto main(std::ostream& plotFile) -> int
    {
        auto title = "AoSoA" + std::to_string(LANES) + " AVX2 " + (UseUpdate1 ? "w1r8" : "w8r1"); // NOLINT
        if (UseAccumulator)
            title += " Acc";
        std::cout << title << '\n';
        Stopwatch watch;

        std::vector<ParticleBlock> particles(BLOCKS);
        watch.printAndReset("alloc");

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
        watch.printAndReset("init");

        double sumUpdate = 0;
        double sumMove = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            if constexpr (UseUpdate1)
            {
                if constexpr (UseAccumulator)
                    update1Acc(particles.data());
                else
                    update1(particles.data());
            }
            else
            {
                if constexpr (UseAccumulator)
                    update8Acc(particles.data());
                else
                    update8(particles.data());
            }
            sumUpdate += watch.printAndReset("update", '\t');
            move(particles.data());
            sumMove += watch.printAndReset("move");
        }
        if (!UseAccumulator)
            plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\t';
        else
            plotFile << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

        return 0;
    }
} // namespace manualAoSoA_manualAVX
#endif

#if __has_include(<Vc/Vc>)
#    include <Vc/Vc>

namespace manualAoSoA_Vc
{
    using vec = Vc::Vector<FP>;

    // this automatically determined LANES based on Vc's chosen vector length
    constexpr auto LANES = sizeof(vec) / sizeof(FP);

    struct alignas(32) ParticleBlock
    {
        struct
        {
            vec x;
            vec y;
            vec z;
        } pos;
        struct
        {
            vec x;
            vec y;
            vec z;
        } vel;
        vec mass;
    };

    constexpr auto BLOCKS = PROBLEM_SIZE / LANES;

    inline void pPInteraction(
        vec piposx,
        vec piposy,
        vec piposz,
        vec& pivelx,
        vec& pively,
        vec& pivelz,
        vec pjposx,
        vec pjposy,
        vec pjposz,
        vec pjmass)
    {
        const vec xdistance = piposx - pjposx;
        const vec ydistance = piposy - pjposy;
        const vec zdistance = piposz - pjposz;
        const vec xdistanceSqr = xdistance * xdistance;
        const vec ydistanceSqr = ydistance * ydistance;
        const vec zdistanceSqr = zdistance * zdistance;
        const vec distSqr = EPS2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
        const vec distSixth = distSqr * distSqr * distSqr;
        const vec invDistCube = [&] {
            if constexpr (ALLOW_RSQRT)
            {
                const vec r = Vc::rsqrt(distSixth);
                if constexpr (NEWTON_RAPHSON_AFTER_RSQRT)
                {
                    // from: http://stackoverflow.com/q/14752399/556899
                    const vec three = 3.0f;
                    const vec half = 0.5f;
                    const vec muls = distSixth * r * r;
                    return (half * r) * (three - muls);
                }
                else
                    return r;
            }
            else
                return 1.0f / Vc::sqrt(distSixth);
        }();
        const vec sts = pjmass * invDistCube * TIMESTEP;
        pivelx = xdistanceSqr * sts + pivelx;
        pively = ydistanceSqr * sts + pively;
        pivelz = zdistanceSqr * sts + pivelz;
    }

    // update (read/write) 8 particles I based on the influence of 1 particle J
    void update8(ParticleBlock* particles, int threads)
    {
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& blockI = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t j = 0; j < LANES; j++)
                {
                    const auto& blockJ = particles[bj];
                    pPInteraction(
                        blockI.pos.x,
                        blockI.pos.y,
                        blockI.pos.z,
                        blockI.vel.x,
                        blockI.vel.y,
                        blockI.vel.z,
                        blockJ.pos.x[j],
                        blockJ.pos.y[j],
                        blockJ.pos.z[j],
                        blockJ.mass[j]);
                }
            // });
        }
    }

    void update8Acc(ParticleBlock* particles, int threads)
    {
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& blockI = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            const vec piposx = blockI.pos.x;
            const vec piposy = blockI.pos.y;
            const vec piposz = blockI.pos.z;
            vec pivelx = blockI.vel.x;
            vec pively = blockI.vel.y;
            vec pivelz = blockI.vel.z;

            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t j = 0; j < LANES; j++)
                {
                    const auto& blockJ = particles[bj];
                    const vec pjposx = blockJ.pos.x[j];
                    const vec pjposy = blockJ.pos.y[j];
                    const vec pjposz = blockJ.pos.z[j];
                    const vec pjmass = blockJ.mass[j];

                    pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);
                }

            blockI.vel.x = pivelx;
            blockI.vel.y = pively;
            blockI.vel.z = pivelz;
            // });
        }
    }

    // update (read/write) 8 particles I based on the influence of 1 particle J
    void update8Tiled(ParticleBlock* particles, int threads)
    {
        constexpr auto blocksPerTile = 128; // L1D_SIZE / sizeof(ParticleBlock);
        static_assert(BLOCKS % blocksPerTile == 0);
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t ti = 0; ti < BLOCKS / blocksPerTile; ti++)
            for (std::size_t tj = 0; tj < BLOCKS / blocksPerTile; tj++)
                for (std::size_t bi = 0; bi < blocksPerTile; bi++)
                    for (std::size_t bj = 0; bj < blocksPerTile; bj++)
                        for (std::size_t j = 0; j < LANES; j++)
                        {
                            auto& blockI = particles[ti * blocksPerTile + bi];
                            const auto& blockJ = particles[tj * blocksPerTile + bj];
                            pPInteraction(
                                blockI.pos.x,
                                blockI.pos.y,
                                blockI.pos.z,
                                blockI.vel.x,
                                blockI.vel.y,
                                blockI.vel.z,
                                blockJ.pos.x[j],
                                blockJ.pos.y[j],
                                blockJ.pos.z[j],
                                blockJ.mass[j]);
                        }
    }

    void update8TiledAcc(ParticleBlock* particles, int threads)
    {
        constexpr auto blocksPerTile = 128; // L1D_SIZE / sizeof(ParticleBlock);
        static_assert(BLOCKS % blocksPerTile == 0);
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t ti = 0; ti < BLOCKS / blocksPerTile; ti++)
            for (std::size_t bi = 0; bi < blocksPerTile; bi++)
            {
                auto& blockI = particles[bi];
                const vec piposx = blockI.pos.x;
                const vec piposy = blockI.pos.y;
                const vec piposz = blockI.pos.z;
                vec pivelx = blockI.vel.x;
                vec pively = blockI.vel.y;
                vec pivelz = blockI.vel.z;
                for (std::size_t tj = 0; tj < BLOCKS / blocksPerTile; tj++)
                    for (std::size_t bj = 0; bj < blocksPerTile; bj++)
                        for (std::size_t j = 0; j < LANES; j++)
                        {
                            const auto& blockJ = particles[bj];
                            const vec pjposx = blockJ.pos.x[j];
                            const vec pjposy = blockJ.pos.y[j];
                            const vec pjposz = blockJ.pos.z[j];
                            const vec pjmass = blockJ.mass[j];

                            pPInteraction(
                                piposx,
                                piposy,
                                piposz,
                                pivelx,
                                pively,
                                pivelz,
                                pjposx,
                                pjposy,
                                pjposz,
                                pjmass);
                        }

                blockI.vel.x = pivelx;
                blockI.vel.y = pively;
                blockI.vel.z = pivelz;
            }
    }

    // update (read/write) 8 particles I based on the influence of 1 particle J
    void update8Tiled2(ParticleBlock* particles, int threads)
    {
        constexpr auto blocksPerL1Tile = 128; // L1D_SIZE / sizeof(ParticleBlock);
        constexpr auto L1TilesPerL2Tile = 16; // L2D_SIZE / (blocksPerL1Tile * sizeof(ParticleBlock));
        static_assert(BLOCKS % blocksPerL1Tile == 0);
        static_assert(BLOCKS % (L1TilesPerL2Tile * blocksPerL1Tile) == 0);

#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t ti2 = 0; ti2 < BLOCKS / (L1TilesPerL2Tile * blocksPerL1Tile); ti2++)
            for (std::size_t tj2 = 0; tj2 < BLOCKS / (L1TilesPerL2Tile * blocksPerL1Tile); tj2++)
                for (std::size_t ti1 = 0; ti1 < L1TilesPerL2Tile; ti1++)
                    for (std::size_t tj1 = 0; tj1 < L1TilesPerL2Tile; tj1++)
                        for (std::size_t bi = 0; bi < blocksPerL1Tile; bi++)
                            for (std::size_t bj = 0; bj < blocksPerL1Tile; bj++)
                                for (std::size_t j = 0; j < LANES; j++)
                                {
                                    auto& blockI = particles
                                        [ti2 * L1TilesPerL2Tile * blocksPerL1Tile + ti1 * blocksPerL1Tile + bi];
                                    const auto& blockJ = particles
                                        [tj2 * L1TilesPerL2Tile * blocksPerL1Tile + tj1 * blocksPerL1Tile + bj];
                                    pPInteraction(
                                        blockI.pos.x,
                                        blockI.pos.y,
                                        blockI.pos.z,
                                        blockI.vel.x,
                                        blockI.vel.y,
                                        blockI.vel.z,
                                        blockJ.pos.x[j],
                                        blockJ.pos.y[j],
                                        blockJ.pos.z[j],
                                        blockJ.mass[j]);
                                }
    }

    // update (read/write) 1 particles I based on the influence of 8 particles J
    void update1(ParticleBlock* particles, int threads)
    {
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& blockI = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            for (std::size_t bj = 0; bj < BLOCKS; bj++)
                for (std::size_t i = 0; i < LANES; i++)
                {
                    const auto& blockJ = particles[bj];
                    vec pivelx = static_cast<FP>(blockI.vel.x[i]);
                    vec pively = static_cast<FP>(blockI.vel.y[i]);
                    vec pivelz = static_cast<FP>(blockI.vel.z[i]);
                    pPInteraction(
                        static_cast<FP>(blockI.pos.x[i]),
                        static_cast<FP>(blockI.pos.y[i]),
                        static_cast<FP>(blockI.pos.z[i]),
                        pivelx,
                        pively,
                        pivelz,
                        blockJ.pos.x,
                        blockJ.pos.y,
                        blockJ.pos.z,
                        blockJ.mass);
                    blockI.vel.x[i] = pivelx.sum();
                    blockI.vel.y[i] = pively.sum();
                    blockI.vel.z[i] = pivelz.sum();
                }
            // });
        }
    }

    void update1Acc(ParticleBlock* particles, int threads)
    {
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < BLOCKS; bi++)
        {
            auto& blockI = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            for (std::size_t i = 0; i < LANES; i++)
            {
                const vec piposx = static_cast<FP>(blockI.pos.x[i]);
                const vec piposy = static_cast<FP>(blockI.pos.y[i]);
                const vec piposz = static_cast<FP>(blockI.pos.z[i]);
                vec pivelx = static_cast<FP>(blockI.vel.x[i]);
                vec pively = static_cast<FP>(blockI.vel.y[i]);
                vec pivelz = static_cast<FP>(blockI.vel.z[i]);

                for (std::size_t bj = 0; bj < BLOCKS; bj++)
                {
                    const auto& blockJ = particles[bj];
                    pPInteraction(
                        piposx,
                        piposy,
                        piposz,
                        pivelx,
                        pively,
                        pivelz,
                        blockJ.pos.x,
                        blockJ.pos.y,
                        blockJ.pos.z,
                        blockJ.mass);
                }

                blockI.vel.x[i] = pivelx.sum();
                blockI.vel.y[i] = pively.sum();
                blockI.vel.z[i] = pivelz.sum();
            }
            // });
        }
    }

    void move(ParticleBlock* particles, int threads)
    {
#    pragma omp parallel for schedule(static) num_threads(threads)
        for (std::ptrdiff_t bi = 0; bi < BLOCKS; bi++)
        {
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& block) {
            auto& block = particles[bi];
            block.pos.x += block.vel.x * TIMESTEP;
            block.pos.y += block.vel.y * TIMESTEP;
            block.pos.z += block.vel.z * TIMESTEP;
            // });
        }
    }

    template <bool UseAccumulator>
    auto main(std::ostream& plotFile, int threads, bool useUpdate1, bool tiled = false) -> int
    {
        auto title = "AoSoA" + std::to_string(LANES) + " Vc" + (useUpdate1 ? " w1r8" : " w8r1"); // NOLINT
        if (tiled)
            title += " tiled";
        if (UseAccumulator)
            title += " Acc";
        title += " " + std::to_string(threads) + "Th";

        std::cout << title << '\n';
        Stopwatch watch;

        std::vector<ParticleBlock> particles(BLOCKS);
        watch.printAndReset("alloc");

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
        watch.printAndReset("init");

        double sumUpdate = 0;
        double sumMove = 0;
        for (std::size_t s = 0; s < STEPS; ++s)
        {
            if (useUpdate1)
            {
                if (UseAccumulator)
                    update1Acc(particles.data(), threads);
                else
                    update1(particles.data(), threads);
            }
            else
            {
                if (tiled && UseAccumulator)
                    update8TiledAcc(particles.data(), threads);
                else if (tiled && !UseAccumulator)
                    // update8Tiled2(particles.data(), threads);
                    update8Tiled(particles.data(), threads);
                else if (!tiled && UseAccumulator)
                    update8Acc(particles.data(), threads);
                else if (!tiled && !UseAccumulator)
                    update8(particles.data(), threads);
            }
            sumUpdate += watch.printAndReset("update", '\t');
            move(particles.data(), threads);
            sumMove += watch.printAndReset("move");
        }
        if (!UseAccumulator)
            plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\t';
        else
            plotFile << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

        return 0;
    }
} // namespace manualAoSoA_Vc
#endif

auto main() -> int
try
{
    std::cout << PROBLEM_SIZE / 1000 << "k particles "
              << "(" << PROBLEM_SIZE * sizeof(FP) * 7 / 1024 << "kiB)\n"
              << "Threads: " << std::thread::hardware_concurrency() << "\n";

    std::ofstream plotFile{"nbody.tsv"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << "\"\"\t\"update\"\t\"move\"\t\"update with acc\"\t\"move with acc\"\n";

    int r = 0;
    using namespace boost::mp11;
    mp_for_each<mp_iota_c<5>>([&](auto i) {
        // only AoSoA (3) needs lanes
        using Lanes
            = std::conditional_t<decltype(i)::value == 3, mp_list_c<std::size_t, 8, 16>, mp_list_c<std::size_t, 0>>;
        mp_for_each<Lanes>([&, i](auto lanes) {
            mp_for_each<mp_list_c<bool, false, true>>([&, i](auto useAccumulator) {
                r += usellama::main<decltype(i)::value, decltype(useAccumulator)::value, decltype(lanes)::value>(
                    plotFile);
            });
        });
    });
    r += manualAoS::main<false>(plotFile);
    r += manualAoS::main<true>(plotFile);
    r += manualSoA::main<false>(plotFile);
    r += manualSoA::main<true>(plotFile);
    mp_for_each<mp_list_c<std::size_t, 8, 16>>([&](auto lanes) {
        r += manualAoSoA::main<false, false, decltype(lanes)::value>(plotFile);
        r += manualAoSoA::main<true, false, decltype(lanes)::value>(plotFile);
    });
    // r += manualAoSoA::main<false, true>(plotFile);
    // r += manualAoSoA::main<true, true>(plotFile);
#ifdef __AVX2__
    r += manualAoSoA_manualAVX::main<false, false>(plotFile);
    r += manualAoSoA_manualAVX::main<true, false>(plotFile);
    r += manualAoSoA_manualAVX::main<false, true>(plotFile);
    r += manualAoSoA_manualAVX::main<true, true>(plotFile);
#endif
#if __has_include(<Vc/Vc>)
    for (int threads = 1; threads <= std::thread::hardware_concurrency(); threads *= 2)
        for (auto useUpdate1 : {false, true})
            for (auto tiled : {false, true})
            {
                if (useUpdate1 && tiled)
                    continue;
                mp_for_each<mp_list_c<bool, false, true>>([&](auto useAccumulator) {
                    r += manualAoSoA_Vc::main<decltype(useAccumulator)::value>(plotFile, threads, useUpdate1, tiled);
                });
            }
#endif

    std::cout << "Plot with: ./nbody.sh\n";
    std::ofstream{"nbody.sh"} << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "nbody CPU {0}k particles on {1}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
plot 'nbody.tsv' using 2:xtic(1) ti col, "" using 4 ti col
)",
        PROBLEM_SIZE / 1000,
        boost::asio::ip::host_name());

    return r;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

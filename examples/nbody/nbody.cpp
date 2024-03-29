// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "../common/Stats.hpp"
#include "../common/Stopwatch.hpp"
#include "../common/env.hpp"

#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#if __has_include(<xsimd/xsimd.hpp>)
#    include <xsimd/xsimd.hpp>
#    define HAVE_XSIMD
#endif

#ifdef _OPENMP
#    define PARALLEL_FOR _Pragma("omp parallel for simd")
#else
#    define PARALLEL_FOR LLAMA_INDEPENDENT_DATA
#endif

using FP = float;

constexpr auto problemSize = 16 * 1024;
constexpr auto steps = 20; ///< number of steps to calculate, excluding 1 warmup run
constexpr auto countFieldAccesses = false;
constexpr auto heatmap = false;
constexpr auto dumpMapping = false;
constexpr auto allowRsqrt = false; // rsqrt can be way faster, but less accurate
constexpr auto newtonRaphsonAfterRsqrt = true; // generate a newton raphson refinement after explicit calls to rsqrt()
constexpr auto runUpdate = true; // run update step. Useful to disable for benchmarking the move step.

constexpr auto timestep = FP{0.0001};
constexpr auto eps2 = FP{0.01};

constexpr auto rngSeed = 42;
constexpr auto referenceParticleIndex = 1338;
constexpr auto maxPosDiff = FP{0.001};

using namespace std::string_literals;

#ifdef HAVE_XSIMD
template<typename Batch>
struct llama::SimdTraits<Batch, std::enable_if_t<xsimd::is_batch<Batch>::value>>
{
    using value_type = typename Batch::value_type;

    inline static constexpr std::size_t lanes = Batch::size;

    static LLAMA_FORCE_INLINE auto loadUnaligned(const value_type* mem) -> Batch
    {
        return Batch::load_unaligned(mem);
    }

    static LLAMA_FORCE_INLINE void storeUnaligned(Batch batch, value_type* mem)
    {
        batch.store_unaligned(mem);
    }

    template<std::size_t... Is>
    static LLAMA_FORCE_INLINE auto indicesToReg(std::array<int, lanes> indices, std::index_sequence<Is...>)
    {
        return xsimd::batch<int, typename Batch::arch_type>(indices[Is]...);
    }

    static LLAMA_FORCE_INLINE auto gather(const value_type* mem, std::array<int, lanes> indices) -> Batch
    {
        return Batch::gather(mem, indicesToReg(indices, std::make_index_sequence<lanes>{}));
    }

    static LLAMA_FORCE_INLINE void scatter(Batch batch, value_type* mem, std::array<int, lanes> indices)
    {
        batch.scatter(mem, indicesToReg(indices, std::make_index_sequence<lanes>{}));
    }
};

template<typename T>
using MakeBatch = xsimd::batch<T>;
#endif

struct Vec3
{
    FP x;
    FP y;
    FP z;

    auto operator*=(FP s) -> Vec3&
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    auto operator*=(Vec3 v) -> Vec3&
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    auto operator+=(Vec3 v) -> Vec3&
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    auto operator-=(Vec3 v) -> Vec3&
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    friend auto operator-(Vec3 a, Vec3 b) -> Vec3
    {
        return a -= b;
    }

    friend auto operator*(Vec3 a, FP s) -> Vec3
    {
        return a *= s;
    }

    friend auto operator*(Vec3 a, Vec3 b) -> Vec3
    {
        return a *= b;
    }

    friend auto operator<<(std::ostream& os, Vec3 p) -> std::ostream&
    {
        return os << fmt::format("{{{} {} {}}}", p.x, p.y, p.z);
    }
};

auto printReferenceParticle(Vec3 position)
{
    std::cout << "reference pos: " << position << "\n";
    return position;
}

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

    using V3 = llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>;
    using Particle = llama::Record<
        llama::Field<tag::Pos, V3>,
        llama::Field<tag::Vel, V3>,
        llama::Field<tag::Mass, FP>>;
    // clang-format on

    namespace stdext
    {
        // workaround until rsqrt lands in C++
        LLAMA_FN_HOST_ACC_INLINE auto rsqrt(FP f) -> FP
        {
            // WARNING: g++-12 cannot auto-vectorize across the following intrinsic. LLAMA SoA will NOT auto-vectorize.
            // return _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(f)));

            // WARNING: GCC and clang will generate a Newton-Raphson step for the following code, which is inconsistent
            // with other rsqrt implementations.
            return FP{1} / std::sqrt(f);
        }
    } // namespace stdext

    template<typename ParticleRefI, typename ParticleRefJ>
    LLAMA_FN_HOST_ACC_INLINE void pPInteraction(ParticleRefI& pi, ParticleRefJ pj)
    {
        auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
        dist *= dist;
        const auto distSqr = eps2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
        const auto distSixth = distSqr * distSqr * distSqr;
        const auto invDistCube = [&]
        {
            if constexpr(allowRsqrt)
            {
#ifdef HAVE_XSIMD
                using xsimd::rsqrt;
#endif
                using stdext::rsqrt;
                const auto r = rsqrt(distSixth);
                if constexpr(newtonRaphsonAfterRsqrt)
                {
                    // from: http://stackoverflow.com/q/14752399/556899
                    const auto three = FP{3};
                    const auto half = FP{0.5};
                    const auto muls = distSixth * r * r;
                    return (half * r) * (three - muls);
                }
                else
                    return r;
            }
            else
            {
#ifdef HAVE_XSIMD
                using xsimd::sqrt;
#endif
                using std::sqrt;
                return FP{1} / sqrt(distSixth);
            }
        }();
        const auto sts = pj(tag::Mass{}) * timestep * invDistCube;
        pi(tag::Vel{}) += dist * sts;
    }

#ifdef HAVE_XSIMD
    template<int Width, typename View>
    void updateSimd(View& particles)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i += Width)
        {
            using RecordDim = typename View::RecordDim;
            llama::SimdN<RecordDim, Width, xsimd::make_sized_batch_t> pis;
            llama::loadSimd(particles(i), pis);
            for(std::size_t j = 0; j < problemSize; ++j)
                pPInteraction(pis, particles(j));
            llama::storeSimd(pis(tag::Vel{}), particles(i)(tag::Vel{}));
        }
    }

    template<int Width, typename View>
    void moveSimd(View& particles)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i += Width)
        {
            using RecordDim = typename View::RecordDim;
            llama::SimdN<llama::GetType<RecordDim, tag::Pos>, Width, xsimd::make_sized_batch_t> pos;
            llama::SimdN<llama::GetType<RecordDim, tag::Vel>, Width, xsimd::make_sized_batch_t> vel;
            llama::loadSimd(particles(i)(tag::Pos{}), pos);
            llama::loadSimd(particles(i)(tag::Vel{}), vel);
            llama::storeSimd(pos + vel * timestep, particles(i)(tag::Pos{}));
        }
    }
#endif

    template<typename View>
    void update(View& particles)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i++)
        {
            llama::One<Particle> pi = particles(i);
            for(std::size_t j = 0; j < problemSize; ++j)
                pPInteraction(pi, particles(j));
            particles(i)(tag::Vel{}) = pi(tag::Vel{});
        }
    }

    template<typename View>
    void move(View& particles)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i++)
            particles(i)(tag::Pos{}) += particles(i)(tag::Vel{}) * timestep;
    }

    /// @tparam SIMDLanes 0: no SIMD, otherwise use the value as SIMD width
    template<int Mapping, std::size_t SIMDLanes, std::size_t AoSoALanes = 8 /*AVX2*/>
    auto main(std::ostream& plotFile) -> Vec3
    {
        auto mappingName = [](int m) -> std::string
        {
            if(m == 0)
                return "AoS";
            if(m == 1)
                return "SoA SB";
            if(m == 2)
                return "SoA MB";
            if(m == 3)
                return "AoSoA" + std::to_string(AoSoALanes);
            if(m == 4)
                return "Split SoA";
            if(m == 5)
                return "ByteSplit AoS";
            if(m == 6)
                return "ByteSplit SoA MB";
            if(m == 7)
                return "BitPack SoA 11e4";
            std::abort();
        };
        auto title = "LLAMA " + mappingName(Mapping) + (SIMDLanes == 0 ? "" : " SIMD W=" + std::to_string(SIMDLanes));
        std::cout << title << "\n";
        Stopwatch watch;
        auto mapping = [&]
        {
            using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 1>;
            const auto extents = ArrayExtents{problemSize};
            if constexpr(Mapping == 0)
                return llama::mapping::AoS<ArrayExtents, Particle>{extents};
            if constexpr(Mapping == 1)
                return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::Single>{extents};
            if constexpr(Mapping == 2)
                return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::OnePerField>{extents};
            if constexpr(Mapping == 3)
                return llama::mapping::AoSoA<ArrayExtents, Particle, AoSoALanes>{extents};
            if constexpr(Mapping == 4)
                return llama::mapping::Split<
                    ArrayExtents,
                    Particle,
                    llama::RecordCoord<1>,
                    llama::mapping::BindSoA<>::fn,
                    llama::mapping::BindSoA<>::fn,
                    true>{extents};
            if constexpr(Mapping == 5)
                return llama::mapping::Bytesplit<ArrayExtents, Particle, llama::mapping::BindAoS<>::fn>{extents};
            if constexpr(Mapping == 6)
                return llama::mapping::Bytesplit<ArrayExtents, Particle, llama::mapping::BindSoA<>::fn>{extents};
            if constexpr(Mapping == 7)
                return llama::mapping::
                    BitPackedFloatSoA<ArrayExtents, Particle, llama::Constant<4>, llama::Constant<11>>{extents};
        }();
        if constexpr(dumpMapping)
            std::ofstream(title + ".svg") << llama::toSvg(mapping);

        auto tmapping = [&]
        {
            if constexpr(countFieldAccesses)
                return llama::mapping::FieldAccessCount{std::move(mapping)};
            else
                return std::move(mapping);
        }();

        auto hmapping = [&]
        {
            if constexpr(heatmap)
                return llama::mapping::Heatmap{std::move(tmapping)};
            else
                return std::move(tmapping);
        }();

        auto particles = llama::allocViewUninitialized(std::move(hmapping));
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            auto p = particles(i);
            p(tag::Pos{}, tag::X{}) = dist(engine);
            p(tag::Pos{}, tag::Y{}) = dist(engine);
            p(tag::Pos{}, tag::Z{}) = dist(engine);
            p(tag::Vel{}, tag::X{}) = dist(engine) / FP{10};
            p(tag::Vel{}, tag::Y{}) = dist(engine) / FP{10};
            p(tag::Vel{}, tag::Z{}) = dist(engine) / FP{10};
            p(tag::Mass{}) = dist(engine) / FP{100};
        }
        if constexpr(countFieldAccesses)
            particles.mapping().fieldHits(particles.blobs()) = {};
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
#ifdef HAVE_XSIMD
                if constexpr(SIMDLanes != 0)
                    updateSimd<SIMDLanes>(particles);
                else
#endif
                    update(particles);
                statsUpdate(watch.printAndReset("update", '\t'));
            }
#ifdef HAVE_XSIMD
            if constexpr(SIMDLanes != 0)
                moveSimd<SIMDLanes>(particles);
            else
#endif
                move(particles);
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;

        if constexpr(heatmap)
            std::ofstream("nbody_heatmap_" + mappingName(Mapping) + ".sh") << particles.mapping().toGnuplotScript();
        if constexpr(countFieldAccesses)
            particles.mapping().printFieldHits(particles.blobs());

        return printReferenceParticle(particles(referenceParticleIndex)(tag::Pos{}).load());
    }
} // namespace usellama

namespace manualAoS
{
    using Pos = Vec3;
    using Vel = Vec3;

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
        const FP distSqr = eps2 + distance.x + distance.y + distance.z;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP sts = pj.mass * timestep * invDistCube;
        pi.vel += distance * sts;
    }

    void update(Particle* particles)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i++)
        {
            Particle pi = particles[i];
            for(std::size_t j = 0; j < problemSize; ++j)
                pPInteraction(pi, particles[j]);
            particles[i].vel = pi.vel;
        }
    }

    void move(Particle* particles)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i++)
            particles[i].pos += particles[i].vel * timestep;
    }

    auto main(std::ostream& plotFile) -> Vec3
    {
        auto title = "AoS"s;
        std::cout << title << "\n";
        Stopwatch watch;

        std::vector<Particle> particles(problemSize);
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(auto& p : particles)
        {
            p.pos.x = dist(engine);
            p.pos.y = dist(engine);
            p.pos.z = dist(engine);
            p.vel.x = dist(engine) / FP{10};
            p.vel.y = dist(engine) / FP{10};
            p.vel.z = dist(engine) / FP{10};
            p.mass = dist(engine) / FP{100};
        }
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
                update(particles.data());
                statsUpdate(watch.printAndReset("update", '\t'));
            }
            manualAoS::move(particles.data());
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;

        return printReferenceParticle(particles[referenceParticleIndex].pos);
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
        const FP distSqr = eps2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP sts = pjmass * timestep * invDistCube;
        pivelx += xdistance * sts;
        pively += ydistance * sts;
        pivelz += zdistance * sts;
    }

    void update(FP* posx, FP* posy, FP* posz, FP* velx, FP* vely, FP* velz, FP* mass)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i++)
        {
            const FP piposx = posx[i];
            const FP piposy = posy[i];
            const FP piposz = posz[i];
            FP pivelx = velx[i];
            FP pively = vely[i];
            FP pivelz = velz[i];
            for(std::size_t j = 0; j < problemSize; ++j)
                pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, posx[j], posy[j], posz[j], mass[j]);
            velx[i] = pivelx;
            vely[i] = pively;
            velz[i] = pivelz;
        }
    }

    void move(FP* posx, FP* posy, FP* posz, const FP* velx, const FP* vely, const FP* velz)
    {
        PARALLEL_FOR
        for(std::size_t i = 0; i < problemSize; i++)
        {
            posx[i] += velx[i] * timestep;
            posy[i] += vely[i] * timestep;
            posz[i] += velz[i] * timestep;
        }
    }

    auto main(std::ostream& plotFile) -> Vec3
    {
        auto title = "SoA MB"s;
        std::cout << title << "\n";
        Stopwatch watch;

        using Vector = std::vector<FP, llama::bloballoc::AlignedAllocator<FP, 64>>;
        Vector posx(problemSize);
        Vector posy(problemSize);
        Vector posz(problemSize);
        Vector velx(problemSize);
        Vector vely(problemSize);
        Vector velz(problemSize);
        Vector mass(problemSize);
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            posx[i] = dist(engine);
            posy[i] = dist(engine);
            posz[i] = dist(engine);
            velx[i] = dist(engine) / FP{10};
            vely[i] = dist(engine) / FP{10};
            velz[i] = dist(engine) / FP{10};
            mass[i] = dist(engine) / FP{100};
        }
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
                update(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data(), mass.data());
                statsUpdate(watch.printAndReset("update", '\t'));
            }
            move(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data());
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;

        return printReferenceParticle(
            {posx[referenceParticleIndex], posy[referenceParticleIndex], posz[referenceParticleIndex]});
    }
} // namespace manualSoA

namespace manualAoSoA
{
    template<std::size_t Lanes>
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
        const FP distSqr = eps2 + xdistance + ydistance + zdistance;
        const FP distSixth = distSqr * distSqr * distSqr;
        const FP invDistCube = 1.0f / std::sqrt(distSixth);
        const FP sts = pjmass * timestep * invDistCube;
        pivelx += xdistance * sts;
        pively += ydistance * sts;
        pivelz += zdistance * sts;
    }

    template<std::size_t Lanes>
    void update(ParticleBlock<Lanes>* particles)
    {
        constexpr auto blocks = problemSize / Lanes;
        PARALLEL_FOR
        for(std::size_t bi = 0; bi < blocks; bi++)
        {
            auto blockI = particles[bi];
            for(std::size_t bj = 0; bj < blocks; bj++)
                for(std::size_t j = 0; j < Lanes; j++)
                {
                    LLAMA_INDEPENDENT_DATA
                    for(std::size_t i = 0; i < Lanes; i++)
                    {
                        const auto& blockJ = particles[bj];
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

            particles[bi].vel = blockI.vel;
        }
    }

    template<std::size_t Lanes>
    void updateTiled(ParticleBlock<Lanes>* particles)
    {
        constexpr auto blocks = problemSize / Lanes;
        constexpr auto blocksPerTile = 128; // L1D_SIZE / sizeof(ParticleBlock<Lanes>);
        static_assert(blocks % blocksPerTile == 0);
        PARALLEL_FOR
        for(std::size_t ti = 0; ti < blocks / blocksPerTile; ti++)
            for(std::size_t tj = 0; tj < blocks / blocksPerTile; tj++)
                for(std::size_t bi = 0; bi < blocksPerTile; bi++)
                {
                    auto blockI = particles[ti * blocksPerTile + bi];
                    for(std::size_t bj = 0; bj < blocksPerTile; bj++)
                        for(std::size_t j = 0; j < Lanes; j++)
                        {
                            LLAMA_INDEPENDENT_DATA
                            for(std::size_t i = 0; i < Lanes; i++)
                            {
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
                    particles[bi].vel = blockI.vel;
                }
    }

    template<std::size_t Lanes>
    void move(ParticleBlock<Lanes>* particles)
    {
        constexpr auto blocks = problemSize / Lanes;
        PARALLEL_FOR
        for(std::size_t bi = 0; bi < blocks; bi++)
        {
            LLAMA_INDEPENDENT_DATA
            for(std::size_t i = 0; i < Lanes; ++i)
            {
                auto& block = particles[bi];
                block.pos.x[i] += block.vel.x[i] * timestep;
                block.pos.y[i] += block.vel.y[i] * timestep;
                block.pos.z[i] += block.vel.z[i] * timestep;
            }
        }
    }

    template<std::size_t Lanes>
    auto main(std::ostream& plotFile, bool tiled) -> Vec3
    {
        auto title = "AoSoA" + std::to_string(Lanes);
        if(tiled)
            title += " tiled";
        std::cout << title << "\n";
        Stopwatch watch;

        constexpr auto blocks = problemSize / Lanes;

        std::vector<ParticleBlock<Lanes>> particles(blocks);
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(std::size_t bi = 0; bi < blocks; ++bi)
        {
            auto& block = particles[bi];
            for(std::size_t i = 0; i < Lanes; ++i)
            {
                block.pos.x[i] = dist(engine);
                block.pos.y[i] = dist(engine);
                block.pos.z[i] = dist(engine);
                block.vel.x[i] = dist(engine) / FP{10};
                block.vel.y[i] = dist(engine) / FP{10};
                block.vel.z[i] = dist(engine) / FP{10};
                block.mass[i] = dist(engine) / FP{100};
            }
        }
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
                if(tiled)
                    updateTiled(particles.data());
                else
                    update(particles.data());
                statsUpdate(watch.printAndReset("update", '\t'));
            }
            move(particles.data());
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;

        const auto& refBlock = particles[referenceParticleIndex / Lanes];
        const auto refLane = referenceParticleIndex % Lanes;
        return printReferenceParticle({refBlock.pos.x[refLane], refBlock.pos.y[refLane], refBlock.pos.z[refLane]});
    }
} // namespace manualAoSoA

#if defined(__AVX2__) && defined(__FMA__)
#    include <immintrin.h>

namespace manualAoSoAManualAVX
{
    // hard coded to AVX2 register length, should be 8
    constexpr auto lanes = sizeof(__m256) / sizeof(float);

    struct alignas(32) ParticleBlock
    {
        struct
        {
            float x[lanes];
            float y[lanes];
            float z[lanes];
        } pos;
        struct
        {
            float x[lanes];
            float y[lanes];
            float z[lanes];
        } vel;
        float mass[lanes];
    };

    constexpr auto blocks = problemSize / lanes;
    const __m256 vEPS2 = _mm256_set1_ps(eps2); // NOLINT(cert-err58-cpp)
    const __m256 vTIMESTEP = _mm256_set1_ps(timestep); // NOLINT(cert-err58-cpp)

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
        const __m256 invDistCube = [&]
        {
            if constexpr(allowRsqrt)
            {
                const __m256 r = _mm256_rsqrt_ps(distSixth);
                if constexpr(newtonRaphsonAfterRsqrt)
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
        // FIXME(bgruber): we could keep pjmass scalar for the update8 version:
        const __m256 sts = _mm256_mul_ps(_mm256_mul_ps(pjmass, vTIMESTEP), invDistCube);
        pivelx = _mm256_fmadd_ps(xdistanceSqr, sts, pivelx);
        pively = _mm256_fmadd_ps(ydistanceSqr, sts, pively);
        pivelz = _mm256_fmadd_ps(zdistanceSqr, sts, pivelz);
    }

    // update (read/write) 8 particles I based on the influence of 1 particle J
    void updateOuterLoopSimd(ParticleBlock* particles)
    {
        PARALLEL_FOR
        for(std::size_t bi = 0; bi < blocks; bi++)
        {
            auto& blockI = particles[bi];
            const __m256 piposx = _mm256_load_ps(&blockI.pos.x[0]);
            const __m256 piposy = _mm256_load_ps(&blockI.pos.y[0]);
            const __m256 piposz = _mm256_load_ps(&blockI.pos.z[0]);
            __m256 pivelx = _mm256_load_ps(&blockI.vel.x[0]);
            __m256 pively = _mm256_load_ps(&blockI.vel.y[0]);
            __m256 pivelz = _mm256_load_ps(&blockI.vel.z[0]);

            for(std::size_t bj = 0; bj < blocks; bj++)
                for(std::size_t j = 0; j < lanes; j++)
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
    void updateInnerLoopSimd(ParticleBlock* particles)
    {
        PARALLEL_FOR
        for(std::size_t bi = 0; bi < blocks; bi++)
            for(std::size_t i = 0; i < lanes; i++)
            {
                auto& blockI = particles[bi];
                const __m256 piposx = _mm256_broadcast_ss(&blockI.pos.x[i]);
                const __m256 piposy = _mm256_broadcast_ss(&blockI.pos.y[i]);
                const __m256 piposz = _mm256_broadcast_ss(&blockI.pos.z[i]);
                __m256 pivelx = _mm256_broadcast_ss(&blockI.vel.x[i]);
                __m256 pively = _mm256_broadcast_ss(&blockI.vel.y[i]);
                __m256 pivelz = _mm256_broadcast_ss(&blockI.vel.z[i]);

                for(std::size_t bj = 0; bj < blocks; bj++)
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
        PARALLEL_FOR
        for(std::size_t bi = 0; bi < blocks; bi++)
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

    auto main(std::ostream& plotFile, bool innerLoopSimd) -> Vec3
    {
        auto title = "AoSoA" + std::to_string(lanes) + " AVX2 " + (innerLoopSimd ? "inner" : "outer"); // NOLINT
        std::cout << title << '\n';
        Stopwatch watch;

        std::vector<ParticleBlock> particles(blocks);
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(std::size_t bi = 0; bi < blocks; ++bi)
        {
            auto& block = particles[bi];
            for(std::size_t i = 0; i < lanes; ++i)
            {
                block.pos.x[i] = dist(engine);
                block.pos.y[i] = dist(engine);
                block.pos.z[i] = dist(engine);
                block.vel.x[i] = dist(engine) / FP{10};
                block.vel.y[i] = dist(engine) / FP{10};
                block.vel.z[i] = dist(engine) / FP{10};
                block.mass[i] = dist(engine) / FP{100};
            }
        }
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
                if(innerLoopSimd)
                    updateInnerLoopSimd(particles.data());
                else
                    updateOuterLoopSimd(particles.data());
                statsUpdate(watch.printAndReset("update", '\t'));
            }
            move(particles.data());
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;

        const auto& refBlock = particles[referenceParticleIndex / lanes];
        const auto refLane = referenceParticleIndex % lanes;
        return printReferenceParticle({refBlock.pos.x[refLane], refBlock.pos.y[refLane], refBlock.pos.z[refLane]});
    }
} // namespace manualAoSoAManualAVX
#endif

#ifdef HAVE_XSIMD
namespace manualAoSoASIMD
{
    template<typename Simd>
    struct alignas(sizeof(Simd)) ParticleBlock
    {
        struct
        {
            Simd x;
            Simd y;
            Simd z;
        } pos, vel;
        Simd mass;
    };

    template<typename Simd, typename SimdOrScalar>
    inline void pPInteraction(
        Simd piposx,
        Simd piposy,
        Simd piposz,
        Simd& pivelx,
        Simd& pively,
        Simd& pivelz,
        Simd pjposx,
        Simd pjposy,
        Simd pjposz,
        SimdOrScalar pjmass)
    {
        const Simd xdistance = piposx - pjposx;
        const Simd ydistance = piposy - pjposy;
        const Simd zdistance = piposz - pjposz;
        const Simd xdistanceSqr = xdistance * xdistance;
        const Simd ydistanceSqr = ydistance * ydistance;
        const Simd zdistanceSqr = zdistance * zdistance;
        const Simd distSqr = eps2 + xdistanceSqr + ydistanceSqr + zdistanceSqr;
        const Simd distSixth = distSqr * distSqr * distSqr;
        const Simd invDistCube = [&]
        {
            if constexpr(allowRsqrt)
            {
                const Simd r = xsimd::rsqrt(distSixth);
                if constexpr(newtonRaphsonAfterRsqrt)
                {
                    // from: http://stackoverflow.com/q/14752399/556899
                    const Simd three = FP{3};
                    const Simd half = FP{0.5};
                    const Simd muls = distSixth * r * r;
                    return (half * r) * (three - muls);
                }
                else
                    return r;
            }
            else
            {
                return FP{1} / xsimd::sqrt(distSixth);
            }
        }();
        const Simd sts = pjmass * timestep * invDistCube;
        pivelx = xdistanceSqr * sts + pivelx;
        pively = ydistanceSqr * sts + pively;
        pivelz = zdistanceSqr * sts + pivelz;
    }

    template<typename Simd>
    void updateOuterLoopSimd(ParticleBlock<Simd>* particles)
    {
        constexpr auto lanes = Simd::size;
        constexpr auto blocks = problemSize / lanes;

        PARALLEL_FOR
        for(std::ptrdiff_t bi = 0; bi < blocks; bi++)
        {
            auto& blockI = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            const Simd piposx = blockI.pos.x;
            const Simd piposy = blockI.pos.y;
            const Simd piposz = blockI.pos.z;
            Simd pivelx = blockI.vel.x;
            Simd pively = blockI.vel.y;
            Simd pivelz = blockI.vel.z;

            for(std::size_t bj = 0; bj < blocks; bj++)
                for(std::size_t j = 0; j < lanes; j++)
                {
                    const auto& blockJ = particles[bj];
                    const Simd pjposx = blockJ.pos.x.get(j);
                    const Simd pjposy = blockJ.pos.y.get(j);
                    const Simd pjposz = blockJ.pos.z.get(j);
                    const FP pjmass = blockJ.mass.get(j);

                    pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);
                }

            blockI.vel.x = pivelx;
            blockI.vel.y = pively;
            blockI.vel.z = pivelz;
            // });
        }
    }

    template<typename Simd>
    void updateOuterLoopSimdTiled(ParticleBlock<Simd>* particles)
    {
        constexpr auto lanes = Simd::size;
        constexpr auto blocks = problemSize / lanes;

        constexpr auto blocksPerTile = 128; // L1D_SIZE / sizeof(ParticleBlock);
        static_assert(blocks % blocksPerTile == 0);
        PARALLEL_FOR
        for(std::ptrdiff_t ti = 0; ti < blocks / blocksPerTile; ti++)
            for(std::size_t bi = 0; bi < blocksPerTile; bi++)
            {
                auto& blockI = particles[bi];
                const Simd piposx = blockI.pos.x;
                const Simd piposy = blockI.pos.y;
                const Simd piposz = blockI.pos.z;
                Simd pivelx = blockI.vel.x;
                Simd pively = blockI.vel.y;
                Simd pivelz = blockI.vel.z;
                for(std::size_t tj = 0; tj < blocks / blocksPerTile; tj++)
                    for(std::size_t bj = 0; bj < blocksPerTile; bj++)
                        for(std::size_t j = 0; j < lanes; j++)
                        {
                            const auto& blockJ = particles[bj];
                            const Simd pjposx = blockJ.pos.x.get(j);
                            const Simd pjposy = blockJ.pos.y.get(j);
                            const Simd pjposz = blockJ.pos.z.get(j);
                            const Simd pjmass = blockJ.mass.get(j);

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

    // xsimd renamed hadd() to reduce_all() on master (after release 8.1), so we need to SFINAE for the name
    template<typename Simd>
    auto sum(Simd v) -> decltype(reduce_add(v))
    {
        return reduce_add(v);
    }

    template<typename Simd>
    auto sum(Simd v) -> decltype(hadd(v))
    {
        return hadd(v);
    }

    template<typename Simd>
    void updateInnerLoopSimd(ParticleBlock<Simd>* particles)
    {
        constexpr auto lanes = Simd::size;
        constexpr auto blocks = problemSize / lanes;

        PARALLEL_FOR
        for(std::ptrdiff_t bi = 0; bi < blocks; bi++)
        {
            auto& blockI = particles[bi];
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& blockI) {
            for(std::size_t i = 0; i < lanes; i++)
            {
                const Simd piposx = blockI.pos.x.get(i);
                const Simd piposy = blockI.pos.y.get(i);
                const Simd piposz = blockI.pos.z.get(i);
                Simd pivelx = blockI.vel.x.get(i);
                Simd pively = blockI.vel.y.get(i);
                Simd pivelz = blockI.vel.z.get(i);

                for(std::size_t bj = 0; bj < blocks; bj++)
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

                reinterpret_cast<FP*>(&blockI.vel.x)[i] = sum(pivelx);
                reinterpret_cast<FP*>(&blockI.vel.y)[i] = sum(pively);
                reinterpret_cast<FP*>(&blockI.vel.z)[i] = sum(pivelz);
            }
            // });
        }
    }

    template<typename Simd>
    void move(ParticleBlock<Simd>* particles)
    {
        constexpr auto blocks = problemSize / Simd::size;

        PARALLEL_FOR
        for(std::ptrdiff_t bi = 0; bi < blocks; bi++)
        {
            // std::for_each(ex, particles, particles + BLOCKS, [&](ParticleBlock& block) {
            auto& block = particles[bi];
            block.pos.x += block.vel.x * timestep;
            block.pos.y += block.vel.y * timestep;
            block.pos.z += block.vel.z * timestep;
            // });
        }
    }

    template<typename Simd>
    auto main(std::ostream& plotFile, bool vectorizeInnerLoop, bool tiled = false) -> Vec3
    {
        auto title
            = "AoSoA" + std::to_string(Simd::size) + " SIMD" + (vectorizeInnerLoop ? " inner" : " outer"); // NOLINT
        if(tiled)
            title += " tiled";

        std::cout << title << '\n';
        Stopwatch watch;

        static_assert(problemSize % Simd::size == 0);
        constexpr auto blocks = problemSize / Simd::size;
        std::vector<ParticleBlock<Simd>> particles(blocks);
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(std::size_t bi = 0; bi < blocks; ++bi)
        {
            auto& block = particles[bi];
            for(std::size_t i = 0; i < Simd::size; ++i)
            {
                reinterpret_cast<FP*>(&block.pos.x)[i] = dist(engine);
                reinterpret_cast<FP*>(&block.pos.y)[i] = dist(engine);
                reinterpret_cast<FP*>(&block.pos.z)[i] = dist(engine);
                reinterpret_cast<FP*>(&block.vel.x)[i] = dist(engine) / FP{10};
                reinterpret_cast<FP*>(&block.vel.y)[i] = dist(engine) / FP{10};
                reinterpret_cast<FP*>(&block.vel.z)[i] = dist(engine) / FP{10};
                reinterpret_cast<FP*>(&block.mass)[i] = dist(engine) / FP{100};
            }
        }
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
                if(vectorizeInnerLoop)
                    updateInnerLoopSimd(particles.data());
                else
                {
                    if(tiled)
                        updateOuterLoopSimdTiled(particles.data());
                    else
                        updateOuterLoopSimd(particles.data());
                }
                statsUpdate(watch.printAndReset("update", '\t'));
            }
            move(particles.data());
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;


        const auto& refBlock = particles[referenceParticleIndex / Simd::size];
        const auto refLane = referenceParticleIndex % Simd::size;
        return printReferenceParticle(
            {refBlock.pos.x.get(refLane), refBlock.pos.y.get(refLane), refBlock.pos.z.get(refLane)});
    }
} // namespace manualAoSoASIMD

namespace manualAoSSIMD
{
    using manualAoS::Particle;
    using manualAoSoASIMD::pPInteraction;

    struct GenStrides
    {
        static constexpr auto get(int i, int) -> int
        {
            return i * static_cast<int>(sizeof(Particle) / sizeof(FP));
        }
    };

    template<typename Simd>
    const auto particleStrides = static_cast<xsimd::batch<int, typename Simd::arch_type>>(
        xsimd::make_batch_constant<xsimd::batch<int, typename Simd::arch_type>, GenStrides>());

    template<typename Simd>
    auto gather(const typename Simd::value_type* p) -> Simd
    {
        const auto strides = particleStrides<Simd>;
        return Simd::gather(p, strides);
        // Simd simd;
        // for(auto i = 0; i < Simd::size; i++)
        //     reinterpret_cast<typename Simd::value_type*>(&simd)[i] = p[i * (sizeof(Particle) / sizeof(FP))];
        // return simd;
    }

    template<typename Simd>
    void scatter(Simd simd, typename Simd::value_type* p)
    {
        const auto strides = particleStrides<Simd>;
        simd.scatter(p, strides);
        // for(auto i = 0; i < Simd::size; i++)
        //     p[i * (sizeof(Particle) / sizeof(FP))] = simd.get(i);
    }

    template<typename Simd>
    void update(std::span<Particle> particles)
    {
        PARALLEL_FOR
        for(std::ptrdiff_t i = 0; i < problemSize; i += Simd::size)
        {
            auto& pi = particles[i];
            const Simd piposx = gather<Simd>(&pi.pos.x);
            const Simd piposy = gather<Simd>(&pi.pos.y);
            const Simd piposz = gather<Simd>(&pi.pos.z);
            Simd pivelx = gather<Simd>(&pi.vel.x);
            Simd pively = gather<Simd>(&pi.vel.y);
            Simd pivelz = gather<Simd>(&pi.vel.z);

            for(std::size_t j = 0; j < problemSize; j++)
            {
                const auto& pj = particles[j];
                const Simd pjposx(pj.pos.x);
                const Simd pjposy(pj.pos.y);
                const Simd pjposz(pj.pos.z);
                const FP pjmass(pj.mass);

                pPInteraction(piposx, piposy, piposz, pivelx, pively, pivelz, pjposx, pjposy, pjposz, pjmass);
            }

            scatter(pivelx, &pi.vel.x);
            scatter(pively, &pi.vel.y);
            scatter(pivelz, &pi.vel.z);
        }
    }

    template<typename Simd>
    void move(std::span<Particle> particles)
    {
        PARALLEL_FOR
        for(std::ptrdiff_t i = 0; i < problemSize; i += Simd::size)
        {
            auto& pi = particles[i];
            scatter(gather<Simd>(&pi.pos.x) + gather<Simd>(&pi.vel.x) * timestep, &pi.pos.x);
            scatter(gather<Simd>(&pi.pos.y) + gather<Simd>(&pi.vel.y) * timestep, &pi.pos.y);
            scatter(gather<Simd>(&pi.pos.z) + gather<Simd>(&pi.vel.z) * timestep, &pi.pos.z);
        }
    }

    template<typename Simd>
    auto main(std::ostream& plotFile) -> Vec3
    {
        const auto title = "AoS SIMD"s;
        std::cout << title << '\n';
        Stopwatch watch;

        std::vector<Particle> particles(problemSize);
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(auto& p : particles)
        {
            p.pos.x = dist(engine);
            p.pos.y = dist(engine);
            p.pos.z = dist(engine);
            p.vel.x = dist(engine) / FP{10};
            p.vel.y = dist(engine) / FP{10};
            p.vel.z = dist(engine) / FP{10};
            p.mass = dist(engine) / FP{100};
        }
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
                update<Simd>(particles);
                statsUpdate(watch.printAndReset("update", '\t'));
            }
            manualAoSSIMD::move<Simd>(particles);
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;

        return printReferenceParticle(particles[referenceParticleIndex].pos);
    }
} // namespace manualAoSSIMD

namespace manualSoASIMD
{
    using manualAoSoASIMD::pPInteraction;

    template<typename Simd>
    void update(const FP* posx, const FP* posy, const FP* posz, FP* velx, FP* vely, FP* velz, const FP* mass)
    {
        PARALLEL_FOR
        for(std::ptrdiff_t i = 0; i < problemSize; i += Simd::size)
        {
            const Simd piposx = Simd::load_aligned(posx + i);
            const Simd piposy = Simd::load_aligned(posy + i);
            const Simd piposz = Simd::load_aligned(posz + i);
            Simd pivelx = Simd::load_aligned(velx + i);
            Simd pively = Simd::load_aligned(vely + i);
            Simd pivelz = Simd::load_aligned(velz + i);
            for(std::size_t j = 0; j < problemSize; ++j)
                pPInteraction(
                    piposx,
                    piposy,
                    piposz,
                    pivelx,
                    pively,
                    pivelz,
                    Simd(posx[j]),
                    Simd(posy[j]),
                    Simd(posz[j]),
                    mass[j]);
            pivelx.store_aligned(velx + i);
            pively.store_aligned(vely + i);
            pivelz.store_aligned(velz + i);
        }
    }

    template<typename Simd>
    void move(FP* posx, FP* posy, FP* posz, const FP* velx, const FP* vely, const FP* velz)
    {
        PARALLEL_FOR
        for(std::ptrdiff_t i = 0; i < problemSize; i += Simd::size)
        {
            (Simd::load_aligned(posx + i) + Simd::load_aligned(velx + i) * timestep).store_aligned(posx + i);
            (Simd::load_aligned(posy + i) + Simd::load_aligned(vely + i) * timestep).store_aligned(posy + i);
            (Simd::load_aligned(posz + i) + Simd::load_aligned(velz + i) * timestep).store_aligned(posz + i);
        }
    }

    template<typename Simd>
    auto main(std::ostream& plotFile) -> Vec3
    {
        const auto title = "SoA MB SIMD";
        std::cout << title << '\n';
        Stopwatch watch;

        using Vector = std::vector<FP, llama::bloballoc::AlignedAllocator<FP, 64>>;
        Vector posx(problemSize);
        Vector posy(problemSize);
        Vector posz(problemSize);
        Vector velx(problemSize);
        Vector vely(problemSize);
        Vector velz(problemSize);
        Vector mass(problemSize);
        watch.printAndReset("alloc");

        std::default_random_engine engine{rngSeed};
        std::normal_distribution<FP> dist(FP{0}, FP{1});
        for(std::size_t i = 0; i < problemSize; ++i)
        {
            posx[i] = dist(engine);
            posy[i] = dist(engine);
            posz[i] = dist(engine);
            velx[i] = dist(engine) / FP{10};
            vely[i] = dist(engine) / FP{10};
            velz[i] = dist(engine) / FP{10};
            mass[i] = dist(engine) / FP{100};
        }
        watch.printAndReset("init");

        common::Stats statsUpdate;
        common::Stats statsMove;
        for(std::size_t s = 0; s < steps + 1; ++s)
        {
            if constexpr(runUpdate)
            {
                update<Simd>(
                    posx.data(),
                    posy.data(),
                    posz.data(),
                    velx.data(),
                    vely.data(),
                    velz.data(),
                    mass.data());
                statsUpdate(watch.printAndReset("update", '\t'));
            }
            move<Simd>(posx.data(), posy.data(), posz.data(), velx.data(), vely.data(), velz.data());
            statsMove(watch.printAndReset("move"));
        }
        plotFile << std::quoted(title) << "\t" << statsUpdate.mean() << "\t" << statsUpdate.sem() << '\t'
                 << statsMove.mean() << "\t" << statsMove.sem() << std::endl;

        return printReferenceParticle(
            {posx[referenceParticleIndex], posy[referenceParticleIndex], posz[referenceParticleIndex]});
    }
} // namespace manualSoASIMD
#endif

auto arePositionsClose(const std::vector<Vec3>& finalPositions) -> bool
{
    Vec3 min = finalPositions.front();
    Vec3 max = min;
    for(const auto& p : finalPositions)
    {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);
        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }
    const auto diff = max - min;
    std::cout << "Reference pos range: " << diff << '\n';
    const auto wrongResults = diff.x > maxPosDiff || diff.y > maxPosDiff || diff.z > maxPosDiff;
    if(wrongResults)
        std::cerr << "WARNING: At least one run had substantially different results!\n";
    else
        std::cout << "All runs had similar results\n";
    return !wrongResults;
}

auto main() -> int
try
{
    const auto env = common::captureEnv();
    fmt::print("{}ki particles ({}kiB)\n{}\n", problemSize / 1024, problemSize * sizeof(FP) * 7 / 1024, env);

    std::ofstream plotFile{"nbody.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
# {}
set title "nbody CPU {}ki particles"
set style data histograms
set style histogram errorbars
set style fill solid border -1
set xtics rotate by 45 right nomirror
set key out top center maxrows 3
set yrange [0:*]
set y2range [0:*]
set ylabel "update runtime [s]"
set y2label "move runtime [s]"
set y2tics auto
$data << EOD
""	"update"	"update_sem"	"move"	"move_sem"
)",
        env,
        problemSize / 1024);

    // Note:
    // Tiled versions did not give any performance benefit, so they are disabled by default.
    // SIMD versions updating 8 particles by 1 are also a bit faster than updating 1 particle by 8, so the latter are
    // also disabled.

#ifdef HAVE_XSIMD
    static constexpr auto nativeSimdWidth = llama::simdLanesWithFullVectorsFor<usellama::Particle, MakeBatch>;
#endif

    std::vector<Vec3> finalPositions;
    using namespace boost::mp11;
    mp_for_each<mp_iota_c<5>>(
        [&](auto ic)
        {
            static constexpr auto i = decltype(ic)::value;
            // only AoSoA (3) needs lanes
            using Lanes = std::conditional_t<i == 3, mp_list_c<std::size_t, 8, 16>, mp_list_c<std::size_t, 0>>;
            mp_for_each<Lanes>(
                [&](auto aosoaLanesIc)
                {
                    static constexpr int aosoaLanes = decltype(aosoaLanesIc)::value;
                    finalPositions.push_back(usellama::main<i, 0, aosoaLanes>(plotFile));
#ifdef HAVE_XSIMD
                    // TODO(bgruber): simd does not work with proxy references yet
                    if constexpr(i < 5)
                    {
                        finalPositions.push_back(usellama::main<i, 1, aosoaLanes>(plotFile));
                        finalPositions.push_back(usellama::main<i, nativeSimdWidth, aosoaLanes>(plotFile));
                    }
#endif
                });
        });
    finalPositions.push_back(manualAoS::main(plotFile));
    finalPositions.push_back(manualSoA::main(plotFile));
    mp_for_each<mp_list_c<std::size_t, 8, 16>>(
        [&](auto lanes)
        {
            // for (auto tiled : {false, true})
            //    r += manualAoSoA::main<decltype(lanes)::value>(plotFile, tiled);
            finalPositions.push_back(manualAoSoA::main<decltype(lanes)::value>(plotFile, false));
        });
#if defined(__AVX2__) && defined(__FMA__)
    // for (auto innerLoopSimd : {false, true})
    //    r += manualAoSoA_manualAVX::main(plotFile, innerLoopSimd);
    finalPositions.push_back(manualAoSoAManualAVX::main(plotFile, false));
#endif
#ifdef HAVE_XSIMD
    using Simd = xsimd::batch<FP>;
    using Simd8 = xsimd::make_sized_batch_t<FP, 8>;
    using Simd16 = xsimd::make_sized_batch_t<FP, 16>;
    // for (auto innerLoopSimd : {false, true})
    //    for (auto tiled : {false, true})
    //    {
    //        if (innerLoopSimd && tiled)
    //            continue;
    //        r += manualAoSoA_SIMD::main<Simd>(plotFile, innerLoopSimd, tiled);
    //    }
    if constexpr(Simd::size != 8 && Simd::size != 16)
        finalPositions.push_back(manualAoSoASIMD::main<Simd>(plotFile, false, false));
    if constexpr(!std::is_void_v<Simd8>)
        finalPositions.push_back(manualAoSoASIMD::main<Simd8>(plotFile, false, false));
    if constexpr(!std::is_void_v<Simd16>)
        finalPositions.push_back(manualAoSoASIMD::main<Simd16>(plotFile, false, false));
    //        mp_for_each<mp_list_c<std::size_t, 1, 2, 4, 8, 16>>(
    //            [&](auto lanes)
    //            {
    //                using Simd = xsimd::make_sized_batch_t<FP, decltype(lanes)::value>;
    //                if constexpr(!std::is_void_v<Simd>)
    //                    r += manualAoS_SIMD::main<Simd>(plotFile);
    //            });
    finalPositions.push_back(manualAoSSIMD::main<Simd>(plotFile));
    finalPositions.push_back(manualSoASIMD::main<Simd>(plotFile));
#endif

    const auto ok = arePositionsClose(finalPositions);

    plotFile << R"(EOD
plot $data using 2:3:xtic(1) ti col axis x1y1, "" using 4:5 ti col axis x1y2
)";
    std::cout << "Plot with: ./nbody.sh\n";

    return ok ? 0 : 1;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

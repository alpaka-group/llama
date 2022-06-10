/*
 * main.cpp
 *
 *  Created on: Aug 20, 2020
 *      Author: Jiri Vyskocil
 */
#define _USE_MATH_DEFINES // NOLINT
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
#    define ALPAKA_ACC_GPU_CUDA_ONLY_MODE
#endif
#include "../../common/hostname.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <iostream>
#include <llama/llama.hpp>
#include <omp.h>
#include <optional>
#include <random>
#include <vector>

using Real = double;

// simulation parameters
constexpr size_t CELLS_X = 1000; // grid size (normalized units, speed of light = 1)
constexpr size_t CELLS_Y = 1000; // grid size
constexpr Real dx = 0.01; // size of the cell
constexpr Real dt = 0.99 * dx / M_SQRT2; // length of a timestep
constexpr int ppc = 10; // particles per cell
constexpr int R_ = 200; // beam size
constexpr int L = 400; // beam size
constexpr Real density = 0.2; // material density
constexpr Real B0x = 0; // initial magnetic field
constexpr Real B0y = 0;
constexpr Real B0z = 0.1; // perpendicular to 2D particle movement, makes particles whirl
constexpr Real v_drift = 0.2; // initial direction of particle beam
constexpr Real v_therm = 0.05; // temperature/speed (ratio of speed of light), influences particle distribution
constexpr Real alpha = 0.0; // beam angle
constexpr int NSTEPS = 100; // simulation steps
const std::optional<int> outputInterval = {}; // e.g. 10
constexpr Real Rcx = R_ * dx + 0; // beam center
constexpr Real Rcy = R_ * dx + 0;
constexpr Real Rcz = 0;
constexpr auto reportInterval = 10;

// tuning parameters
constexpr auto AOSOA_LANES = 32;
constexpr auto THREADS_PER_BLOCK = 128;

constexpr size_t X_ = CELLS_X + 2; // grid size including ghost cells
constexpr size_t Y_ = CELLS_Y + 2; // grid size including ghost cells

// clang-format off
struct X{};
struct Y{};
struct Z{};

using V3Real = llama::Record<
    llama::Field<X, Real>,
    llama::Field<Y, Real>,
    llama::Field<Z, Real>
>;

struct R{};
struct U{};
struct Q{};
struct M{};

using Particle = llama::Record<
    llama::Field<R, V3Real>,
    llama::Field<U, V3Real>,
    llama::Field<Q, Real>,
    llama::Field<M, Real>
>;
// clang-format on

constexpr Real PI = M_PI;

template<typename Acc>
inline constexpr bool isGPU = false;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template<typename Dim, typename Idx>
inline constexpr bool isGPU<alpaka::AccGpuCudaRt<Dim, Idx>> = true;
#endif

using Size = std::size_t;
using Dim = alpaka::DimInt<2>;
using Vec = alpaka::Vec<Dim, Size>;

template<typename Acc>
auto roundUp1DWorkDiv(std::size_t n)
{
    const auto tpb = std::size_t{isGPU<Acc> ? THREADS_PER_BLOCK : 1};
    auto wd = alpaka::WorkDivMembers<Dim, Size>(
        Vec{std::size_t{1}, (n + tpb - 1) / tpb},
        Vec{std::size_t{1}, tpb},
        Vec::ones());
    // std::cout << wd << '\n';
    return wd;
}

template<typename Acc>
auto roundUp2DWorkDiv(std::size_t y, std::size_t x)
{
    const auto tpb = isGPU<Acc> ? THREADS_PER_BLOCK : 1;
    const auto ytpb = std::size_t{1} << (static_cast<std::size_t>(std::log2(tpb)) / 2);
    const auto xtpb = tpb / ytpb;
    auto wd = alpaka::WorkDivMembers<Dim, Size>(
        Vec{(y + ytpb - 1) / ytpb, (x + xtpb - 1) / xtpb},
        Vec{ytpb, xtpb},
        Vec::ones());
    // std::cout << wd << '\n';
    return wd;
}

LLAMA_FN_HOST_ACC_INLINE auto makeV3Real(Real x, Real y, Real z) noexcept
{
    llama::One<V3Real> r;
    r(X{}) = x;
    r(Y{}) = y;
    r(Z{}) = z;
    return r;
}

const auto Rc = makeV3Real(Rcx, Rcy, Rcz);
constexpr Real inv_dx = 1 / dx;

template<typename FieldView>
LLAMA_FN_HOST_ACC_INLINE auto rot(const FieldView& field, size_t i, size_t j, int direction)
{
    const size_t ii = i + direction;
    const size_t jj = j + direction;

    return makeV3Real(
        (field(i, j)(Z{}) - field(i, jj)(Z{})) * inv_dx,
        (field(ii, j)(Z{}) - field(i, j)(Z{})) * inv_dx,
        (field(i, j)(Y{}) - field(ii, j)(Y{})) * inv_dx - (field(i, j)(X{}) - field(i, jj)(X{})) * inv_dx);
}

auto swapBytes(float t) -> float
{
    uint32_t a = 0;
    std::memcpy(&a, &t, sizeof(a));
    a = ((a & 0x000000FFu) << 24u) | ((a & 0x0000FF00u) << 8u) | ((a & 0x00FF0000u) >> 8u)
        | ((a & 0xFF000000u) >> 24u);
    std::memcpy(&t, &a, sizeof(a));
    return t;
}

template<typename VectorField2D>
void output(int n, const std::string& name, const VectorField2D& field)
{
    std::ofstream f("out-" + name + "-" + std::to_string(n) + ".vtk", std::ios::binary);

    f << "# vtk DataFile Version 3.0\nvtkfile\n"
      << "BINARY\nDATASET STRUCTURED_POINTS\n"
      << "DIMENSIONS 1 " << Y_ << " " << X_ << "\n"
      << "ORIGIN 0 0 0\n"
      << "SPACING " << dx << " " << dx << " " << dx << "\n"
      << "POINT_DATA " << X_ * Y_ << "\n"
      << "VECTORS " << name << " float\n";

    std::vector<float> buffer;
    buffer.reserve(X_ * Y_ * 3);
    for(size_t i = 0; i < X_; i++)
    {
        for(size_t j = 0; j < Y_; j++)
        {
            const auto& v = field(i, j);
            buffer.push_back(swapBytes(v(X{})));
            buffer.push_back(swapBytes(v(Y{})));
            buffer.push_back(swapBytes(v(Z{})));
        }
    }
    f.write(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(float)));
}

template<typename ParticleView>
void output(int n, const ParticleView& particles)
{
    std::ofstream outP("out-particles-" + std::to_string(n) + ".vtk", std::ios::binary);
    outP << "# vtk DataFile Version 3.0\nvtkfile\nBINARY\nDATASET POLYDATA\n";

    std::vector<float> buffer;
    auto flushBuffer = [&] {
        outP.write(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size() * sizeof(float)));
    };
    auto addFloat = [&](float f) { buffer.push_back(swapBytes(f)); };

    const auto pointCount = particles.mapping().extents()[0];
    outP << "POINTS " << pointCount << " float\n";
    buffer.reserve(pointCount * 3);
    for(auto i : llama::ArrayIndexRange{particles.mapping().extents()})
    {
        auto p = particles(i);
        addFloat(0);
        addFloat(p(U{}, Y{}));
        addFloat(p(U{}, X{}));
    }
    flushBuffer();

    outP << "POINT_DATA " << pointCount << "\nVECTORS velocity float\n";
    buffer.clear();
    for(auto i : llama::ArrayIndexRange{particles.mapping().extents()})
    {
        auto p = particles(i);
        addFloat(p(U{}, Z{}));
        addFloat(p(U{}, Y{}));
        addFloat(p(U{}, X{}));
    }
    flushBuffer();

    outP << "SCALARS q float 1\nLOOKUP_TABLE default\n";
    buffer.clear();
    for(auto i : llama::ArrayIndexRange{particles.mapping().extents()})
        addFloat(particles(i)(Q{}));
    flushBuffer();

    outP << "SCALARS m float 1\nLOOKUP_TABLE default\n";
    buffer.clear();
    for(auto i : llama::ArrayIndexRange{particles.mapping().extents()})
        addFloat(particles(i)(M{}));
    flushBuffer();
}

template<typename Acc, typename Queue, typename FieldView>
void initFields(Queue& queue, FieldView E, FieldView B, FieldView J)
{
    const auto workDiv = roundUp2DWorkDiv<Acc>(X_, Y_);
    alpaka::exec<Acc>(
        queue,
        workDiv,
        [] ALPAKA_FN_ACC(const auto& acc, auto E, auto B, auto J)
        {
            const auto [x, y] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            if(x < X_ && y < Y_)
            {
                E(x, y) = 0;
                B(x, y) = 1e6 * makeV3Real(B0x, B0y, B0z);
                J(x, y) = 0;
            }
        },
        llama::shallowCopy(E),
        llama::shallowCopy(B),
        llama::shallowCopy(J));
}

constexpr size_t numpart = 2 * ppc * R_ * L;

template<int FieldMapping, int ParticleMapping, typename Acc, typename Queue, typename Dev, typename DevHost>
auto setup(Queue& queue, const Dev& dev, const DevHost& devHost)
{
    // std::cout << "dt = " << dt << '\n';

    const auto fieldMapping = []
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<size_t, 2>;
        const auto fieldExtents = ArrayExtents{{X_, Y_}};
        if constexpr(FieldMapping == 0)
            return llama::mapping::AoS<ArrayExtents, V3Real>(fieldExtents);
        if constexpr(FieldMapping == 1)
            return llama::mapping::AoS<ArrayExtents, V3Real, true, llama::mapping::LinearizeArrayDimsFortran>(
                fieldExtents);
        if constexpr(FieldMapping == 2)
            return llama::mapping::AoS<ArrayExtents, V3Real, true, llama::mapping::LinearizeArrayDimsMorton>(
                fieldExtents);
        if constexpr(FieldMapping == 3)
            return llama::mapping::SoA<ArrayExtents, V3Real, false>(fieldExtents);
        if constexpr(FieldMapping == 4)
            return llama::mapping::SoA<ArrayExtents, V3Real, false, llama::mapping::LinearizeArrayDimsFortran>(
                fieldExtents);
        if constexpr(FieldMapping == 5)
            return llama::mapping::SoA<ArrayExtents, V3Real, false, llama::mapping::LinearizeArrayDimsMorton>(
                fieldExtents);
        if constexpr(FieldMapping == 6)
            return llama::mapping::SoA<ArrayExtents, V3Real, true>(fieldExtents);
        if constexpr(FieldMapping == 7)
            return llama::mapping::SoA<ArrayExtents, V3Real, true, llama::mapping::LinearizeArrayDimsFortran>(
                fieldExtents);
        if constexpr(FieldMapping == 8)
            return llama::mapping::SoA<ArrayExtents, V3Real, true, llama::mapping::LinearizeArrayDimsMorton>(
                fieldExtents);
        if constexpr(FieldMapping == 9)
            return llama::mapping::AoSoA<ArrayExtents, V3Real, AOSOA_LANES>{fieldExtents};
    }();

    int i = 0;
    auto alpakaBlobAlloc = [&](auto alignment, std::size_t count)
    {
        fmt::print("Buffer #{}: {}MiB\n", i++, count / 1024 / 1024);
        return alpaka::allocBuf<std::byte, Size>(dev, count);
    };
    //    auto alpakaBlobAlloc = llama::bloballoc::AlpakaBuf<Size, decltype(dev)>{dev};
    auto E = llama::allocViewUninitialized(fieldMapping, alpakaBlobAlloc);
    auto B = llama::allocViewUninitialized(fieldMapping, alpakaBlobAlloc);
    auto J = llama::allocViewUninitialized(fieldMapping, alpakaBlobAlloc);

    initFields<Acc>(queue, E, B, J);

    const Real r = R_ * dx;
    const Real l = L * dx;

    auto particleMapping = [&]
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<size_t, 1>;
        const auto particleExtents = ArrayExtents{numpart};
        if constexpr(ParticleMapping == 0)
            return llama::mapping::AoS<ArrayExtents, Particle>{particleExtents};
        if constexpr(ParticleMapping == 1)
            return llama::mapping::SoA<ArrayExtents, Particle, false>{particleExtents};
        if constexpr(ParticleMapping == 2)
            return llama::mapping::SoA<ArrayExtents, Particle, true>{particleExtents};
        if constexpr(ParticleMapping == 3)
            return llama::mapping::AoSoA<ArrayExtents, Particle, AOSOA_LANES>{particleExtents};
    }();
    const auto particleBufferSize = particleMapping.blobSize(0);

    auto particles = llama::allocViewUninitialized(particleMapping, alpakaBlobAlloc);
    auto particlesHost = llama::allocViewUninitialized(particleMapping);

    std::default_random_engine engine;
    auto uniform = [&] { return std::uniform_real_distribution<Real>{}(engine); };
    auto gauss = [&] { return sqrt(-2.0 * log(1.0 - uniform())) * sin(2 * PI * uniform()); };

    /* Beam initialization. Plasma must be neutral in each cell, so let's just
     * create an electron and a proton at the same spot. (I know...) */
    for(size_t i = 0; i < numpart; i += 2)
    {
        const Real rp = 2 * r * (uniform() - 0.5);
        const Real lp = l * uniform();
        const auto r = Rc + makeV3Real(lp * sin(alpha) + rp * cos(alpha), lp * cos(alpha) - rp * sin(alpha), 0)
            + dx * makeV3Real(1, 1, 0); // ghost cell
        const auto u = makeV3Real(v_drift * sin(alpha), v_drift * cos(alpha), 0);

        assert(r(X{}) >= 0 && r(X{}) <= (CELLS_X + 1) * dx);
        assert(r(Y{}) >= 0 && r(Y{}) <= (CELLS_Y + 1) * dx);

        /* Electron */
        auto p = particlesHost(i);
        p(Q{}) = -1;
        p(M{}) = 1;
        p(R{}) = r;
        p(U{}) = u + v_therm * makeV3Real(gauss(), gauss(), gauss());

        /* Proton */
        auto p2 = particlesHost(i + 1);
        p2(Q{}) = 1;
        p2(M{}) = 1386;
        p2(R{}) = r;
        p2(U{}) = u + v_therm * makeV3Real(gauss(), gauss(), gauss());
    }
    // std::shuffle(particlesHost.begin(), particlesHost.end(), engine);
    for(auto i = 0; i < decltype(particleMapping)::blobCount; i++)
        alpaka::memcpy(queue, particles.storageBlobs[i], particlesHost.storageBlobs[i]);

    return std::tuple{E, B, J, particles};
}

template<typename Vec3F>
LLAMA_FN_HOST_ACC_INLINE auto squaredNorm(const Vec3F& vec) -> Real
{
    return vec(X{}) * vec(X{}) + vec(Y{}) * vec(Y{}) + vec(Z{}) * vec(Z{});
}

template<typename Vec3F>
LLAMA_FN_HOST_ACC_INLINE auto cross(const Vec3F& a, const Vec3F& b)
{
    return makeV3Real(
        a(Y{}) * b(Z{}) - a(Z{}) * b(Y{}),
        a(Z{}) * b(Z{}) - a(Z{}) * b(Z{}),
        a(Z{}) * b(Y{}) - a(Y{}) * b(Z{}));
}

template<typename VirtualParticle, typename FieldView>
LLAMA_FN_HOST_ACC_INLINE void advance_particle(VirtualParticle part, const FieldView& E, const FieldView& B)
{
    /* Interpolate fields (nearest-neighbor) */
    const auto i = static_cast<size_t>(part(R{}, X{}) * inv_dx);
    const auto i_shift = static_cast<size_t>(std::lround(part(R{}, X{}) * inv_dx));
    const auto j = static_cast<size_t>(part(R{}, Y{}) * inv_dx);
    const auto j_shift = static_cast<size_t>(std::lround(part(R{}, Y{}) * inv_dx));

    const auto E_ngp = makeV3Real(E(i_shift, j)(X{}), E(i, j_shift)(Y{}), E(i, j)(Z{}));
    const auto B_ngp = makeV3Real(B(i, j_shift)(X{}), B(i_shift, j)(Y{}), B(i_shift, j_shift)(Z{}));

    /* q/m [SI] -> 2*PI*q/m in dimensionless */
    const Real qmpidt = part(Q{}) / part(M{}) * PI * dt;
    const Real igamma = 1 / sqrt(1 + squaredNorm(part(U{})));

    const auto F_E = qmpidt * E_ngp;
    const auto F_B = igamma * qmpidt * B_ngp;

    /* Buneman-Boris scheme */
    const auto u_m = part(U{}) + F_E;
    const auto u_0 = u_m + cross(u_m, F_B);
    const auto u_p = u_m + 2 / (1 + squaredNorm(F_B)) * cross(u_0, F_B);
    const auto u = u_p + F_E;

    /* Update stored velocity and advance particle */
    const Real igamma2 = 1 / sqrt(1 + squaredNorm(part(U{})));
    part(U{}) = u;
    part(R{}) += dt * igamma2 * u;
    part(R{}, Z{}) = 0;

#ifndef ALPAKA_ACC_GPU_CUDA_ENABLED
    assert(part(R{}, X{}) >= 0 && part(R{}, Y{}) <= (CELLS_X + 1) * dx);
    assert(part(R{}, Y{}) >= 0 && part(R{}, Y{}) <= (CELLS_Y + 1) * dx);
#endif
}

template<typename VirtualParticle>
LLAMA_FN_HOST_ACC_INLINE void particle_boundary_conditions(VirtualParticle p)
{
    //* Periodic boundary condition
    if(p(R{}, X{}) < dx)
        p(R{}, X{}) += (X_ - 3) * dx;
    else if(p(R{}, X{}) > (X_ - 2) * dx)
        p(R{}, X{}) -= (X_ - 3) * dx;
    if(p(R{}, Y{}) < dx)
        p(R{}, Y{}) += (Y_ - 3) * dx;
    else if(p(R{}, Y{}) > (Y_ - 2) * dx)
        p(R{}, Y{}) -= (Y_ - 3) * dx;
}

template<typename VirtualParticle, typename FieldView>
LLAMA_FN_HOST_ACC_INLINE void deposit_current(const VirtualParticle& part, FieldView& J)
{
    /* Calculation of previous position */
    const Real igamma = 1 / sqrt(1 + squaredNorm(part(U{})));
    const auto r_old = part(R{}) - igamma * dt * part(U{}); // Position should be a 2D vector

    /* Particle position to cell coordinates */
    const auto i = static_cast<size_t>(part(R{}, X{}) * inv_dx);
    const auto i_shift = static_cast<size_t>(std::lround(part(R{}, X{}) * inv_dx));
    const auto j = static_cast<size_t>(part(R{}, Y{}) * inv_dx);
    const auto j_shift = static_cast<size_t>(std::lround(part(R{}, Y{}) * inv_dx));

    const auto i_old = static_cast<size_t>(r_old(X{}) * inv_dx);
    const auto i_old_shift = static_cast<size_t>(std::lround(r_old(X{}) * inv_dx));
    const auto j_old = static_cast<size_t>(r_old(Y{}) * inv_dx);
    const auto j_old_shift = static_cast<size_t>(std::lround(r_old(Y{}) * inv_dx));

    /* Current density deposition */
    const auto F = density * part(Q{}) * igamma * part(U{});

    // mid-x edge at (0.5, 0)
    J(i_old_shift, j_old)(X{}) -= F(X{});
    J(i_shift, j)(X{}) += F(X{});

    // mid-y edge at (0, 0.5)
    J(i_old, j_old_shift)(Y{}) -= F(Y{});
    J(i, j_shift)(Y{}) += F(Y{});

    // Fake mid-z edge at node (0, 0)
    J(i_old, j_old)(Z{}) -= F(Z{});
    J(i, j)(Z{}) += F(Z{});
}

/* Half-step in B */
template<typename FieldView>
LLAMA_FN_HOST_ACC_INLINE void advance_B_half(const FieldView& E, FieldView& B, size_t i, size_t j)
{
    B(i, j) += 0.5 * dt * rot(E, i, j, 1);
}

/* Full step in E */
template<typename FieldView>
LLAMA_FN_HOST_ACC_INLINE void advance_E(FieldView& E, const FieldView& B, const FieldView& J, size_t i, size_t j)
{
    E(i, j) += +dt * rot(B, i, j, -1) - 2 * PI * +dt * J(i, j); // full step i n E
}

using clock_ = std::chrono::high_resolution_clock;

struct Times
{
    using duration = clock_::duration;
    duration clearCurrent;
    duration integrateMotion;
    duration depositCharge;
    duration boundariesCurrent;
    duration advanceB1;
    duration boundariesB1;
    duration advanceE;
    duration boundariesE;
    duration advanceB2;
    duration boundariesB2;

    auto sum() const -> duration
    {
        return clearCurrent + integrateMotion + depositCharge + boundariesCurrent + advanceB1 + boundariesB1 + advanceE
            + boundariesE + advanceB2 + boundariesB2;
    }

    auto operator+=(const Times& t) -> Times&
    {
        clearCurrent += t.clearCurrent;
        integrateMotion += t.integrateMotion;
        depositCharge += t.depositCharge;
        boundariesCurrent += t.boundariesCurrent;
        advanceB1 += t.advanceB1;
        boundariesB1 += t.boundariesB1;
        advanceE += t.advanceE;
        boundariesE += t.boundariesE;
        advanceB2 += t.advanceB2;
        boundariesB2 += t.boundariesB2;
        return *this;
    }
};

template<typename Acc, typename Queue, typename FieldView>
void field_boundary_condition(Queue& queue, FieldView& field)
{
    static const auto workDiv = roundUp1DWorkDiv<Acc>(X_ + Y_);
    alpaka::exec<Acc>(
        queue,
        workDiv,
        [] ALPAKA_FN_ACC(const auto& acc, auto field)
        {
            const auto [_, i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            if(i < X_)
            {
                field(i, size_t{0}) = field(i, Y_ - 2);
                field(i, Y_ - 1) = field(i, size_t{1});
            }
            else
            {
                const auto j = i - X_;
                if(j > 0 && j < Y_ - 1)
                {
                    field(size_t{0}, j) = field(X_ - 2, j);
                    field(X_ - 1, j) = field(size_t{1}, j);
                }
            }
        },
        llama::shallowCopy(field));
}

template<typename Acc, typename Queue, typename FieldView, typename ParticleView>
void step(Queue& queue, FieldView& E, FieldView& B, FieldView& J, ParticleView& particles, Times& times)
{
    static constexpr auto gridStride = false; // isGPU<Acc>;
    constexpr auto elements = gridStride ? 16 : 1;
    static const auto fieldX = (X_ + elements - 1) / elements;
    static const auto fieldWorkDiv = roundUp2DWorkDiv<Acc>(fieldX, Y_);
    static const auto particleWorkDiv = roundUp1DWorkDiv<Acc>(numpart);
    static const auto borderWorkDiv = roundUp1DWorkDiv<Acc>(X_ + Y_);

    // clear charge/current density fields
    auto start = clock_::now();
    alpaka::exec<Acc>(
        queue,
        fieldWorkDiv,
        [] ALPAKA_FN_ACC(const auto& acc, auto J)
        {
            auto [x, y] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            if constexpr(gridStride)
            {
                const auto [xStride, _] = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
                if(y < Y_)
                    while(x < X_)
                    {
                        J(x, y) = 0;
                        x += xStride;
                    }
            }
            else
            {
                if(x < X_ && y < Y_)
                    J(x, y) = 0;
            }
        },
        llama::shallowCopy(J));
    times.clearCurrent += clock_::now() - start;

    // integrate equations of motion
    start = clock_::now();
    alpaka::exec<Acc>(
        queue,
        particleWorkDiv,
        [] ALPAKA_FN_ACC(const auto& acc, auto particles, const auto E, const auto B)
        {
            const auto [_, i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            if(i < numpart)
            {
                advance_particle(particles(i), E, B);
                particle_boundary_conditions(particles(i));
            }
        },
        llama::shallowCopy(particles),
        llama::shallowCopy(E),
        llama::shallowCopy(B));
    times.integrateMotion += clock_::now() - start;

    // deposit charge/current density
    start = clock_::now();
    alpaka::exec<Acc>(
        queue,
        particleWorkDiv,
        [] ALPAKA_FN_ACC(const auto& acc, const auto particles, auto J)
        {
            const auto [_, i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            if(i < numpart)
                deposit_current(particles(i), J);
        },
        llama::shallowCopy(particles),
        llama::shallowCopy(J));
    times.depositCharge += clock_::now() - start;

    // current boundary conditions
    start = clock_::now();
    alpaka::exec<Acc>(
        queue,
        borderWorkDiv,
        [] ALPAKA_FN_ACC(const auto& acc, auto J)
        {
            const auto [_, i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            if(i < X_)
            {
                J(i, Y_ - 2) += J(i, size_t{0});
                J(i, size_t{1}) += J(i, Y_ - 1);
            }
            else
            {
                const auto j = i - X_;
                if(j > 0 && j < Y_ - 1)
                {
                    J(X_ - 2, j) += J(size_t{0}, j);
                    J(size_t{1}, j) += J(X_ - 1, j);
                }
            }
        },
        llama::shallowCopy(J));
    times.boundariesCurrent += clock_::now() - start;

    auto run_advance_B_half_kernel = [&]
    {
        alpaka::exec<Acc>(
            queue,
            fieldWorkDiv,
            [] ALPAKA_FN_ACC(const auto& acc, const auto E, auto B)
            {
                auto [x, y] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
                if constexpr(gridStride)
                {
                    const auto [xStride, _] = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
                    if(y > 0 && y < Y_ - 1)
                        while(x > 0 && x < X_ - 1)
                        {
                            advance_B_half(E, B, x, y);
                            x += xStride;
                        }
                }
                else
                {
                    if(x > 0 && x < X_ - 1 && y > 0 && y < Y_ - 1)
                        advance_B_half(E, B, x, y);
                }
            },
            llama::shallowCopy(E),
            llama::shallowCopy(B));
    };

    // Integrate Maxwell's equations
    start = clock_::now();
    run_advance_B_half_kernel();
    times.advanceB1 += clock_::now() - start;

    start = clock_::now();
    field_boundary_condition<Acc>(queue, B);
    times.boundariesB1 += clock_::now() - start;

    start = clock_::now();
    alpaka::exec<Acc>(
        queue,
        fieldWorkDiv,
        [] ALPAKA_FN_ACC(const auto& acc, auto E, const auto B, const auto J)
        {
            auto [x, y] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
            if constexpr(gridStride)
            {
                const auto [xStride, _] = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);
                if(y > 0 && y < Y_ - 1)
                    while(x > 0 && x < X_ - 1)
                    {
                        advance_E(E, B, J, x, y);
                        x += xStride;
                    }
            }
            else
            {
                if(x > 0 && x < X_ - 1 && y > 0 && y < Y_ - 1)
                    advance_E(E, B, J, x, y);
            }
        },
        llama::shallowCopy(E),
        llama::shallowCopy(B),
        llama::shallowCopy(J));
    times.advanceE += clock_::now() - start;

    start = clock_::now();
    field_boundary_condition<Acc>(queue, E);
    times.boundariesE += clock_::now() - start;

    start = clock_::now();
    run_advance_B_half_kernel();
    times.advanceB2 += clock_::now() - start;

    start = clock_::now();
    field_boundary_condition<Acc>(queue, B);
    times.boundariesB2 += clock_::now() - start;
}

auto particleMappingName(int m) -> std::string
{
    if(m == 0)
        return "AoS";
    if(m == 1)
        return "SoA";
    if(m == 2)
        return "SoA MB";
    if(m == 3)
        return "AoSoA" + std::to_string(AOSOA_LANES);
    std::abort();
}

auto fieldMappingName(int m) -> std::string
{
    if(m == 0)
        return "AoS";
    if(m == 1)
        return "AoS Col";
    if(m == 2)
        return "AoS Mor";
    if(m == 3)
        return "SoA";
    if(m == 4)
        return "SoA Col";
    if(m == 5)
        return "SoA Mor";
    if(m == 6)
        return "SoA MB";
    if(m == 7)
        return "SoA MB Col";
    if(m == 8)
        return "SoA MB Mor";
    if(m == 9)
        return "AoSoA" + std::to_string(AOSOA_LANES);
    std::abort();
}

template<int FieldMapping, int ParticleMapping>
void run(std::ostream& plotFile)
{
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;
    const DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    fmt::print("Particle mapping: {}\n", particleMappingName(ParticleMapping));
    fmt::print("Field mapping:    {}\n", fieldMappingName(FieldMapping));
    auto [E, B, J, particles] = setup<FieldMapping, ParticleMapping, Acc>(queue, devAcc, devHost);

    std::cout << std::string(reportInterval - 1, ' ')
              << "|  steps |  clr J | integr |  dep J |  bnd J | adv B1 | bnd B1 |  adv E |  bnd E | adv B2 | bnd B2 "
                 "|  total |\n";
    Times times = {};
    Times totalSimulationTimes = {};
    for(int n = 0; n <= NSTEPS; n++)
    {
        if(n % reportInterval != 0)
            std::cout << "." << std::flush;
        else if(n > 0)
        {
            auto avgTime = [](clock_::duration d)
            { return std::chrono::duration_cast<std::chrono::microseconds>(d).count() / reportInterval; };
            fmt::print(
                "| {:6} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} | {:6} |\n",
                n,
                avgTime(times.clearCurrent),
                avgTime(times.integrateMotion),
                avgTime(times.depositCharge),
                avgTime(times.boundariesCurrent),
                avgTime(times.advanceB1),
                avgTime(times.boundariesB1),
                avgTime(times.advanceE),
                avgTime(times.boundariesE),
                avgTime(times.advanceB2),
                avgTime(times.boundariesB2),
                avgTime(times.sum()));

            std::cout.flush();
            totalSimulationTimes += times;
            times = {};
        }
        step<Acc>(queue, E, B, J, particles, times);

        /* Output data */
        if(outputInterval && n % *outputInterval == 0)
        {
            {
                const auto fieldMapping = E.mapping();
                auto hostFieldView = llama::allocViewUninitialized(fieldMapping);
                auto copyBlobs = [&](auto& fieldView)
                {
                    for(auto i = 0; i < fieldMapping.blobCount; i++)
                        alpaka::memcpy(queue, hostFieldView.storageBlobs[i], fieldView.storageBlobs[i]);
                };
                copyBlobs(E);
                output(n, "E", hostFieldView);
                copyBlobs(B);
                output(n, "B", hostFieldView);
                copyBlobs(J);
                output(n, "J", hostFieldView);
            }

            const auto particlesMapping = particles.mapping();
            auto hostParticleView = llama::allocViewUninitialized(particlesMapping);
            for(auto i = 0; i < particlesMapping.blobCount; i++)
                alpaka::memcpy(queue, hostParticleView.storageBlobs[i], particles.storageBlobs[i]);
            output(n, hostParticleView);
        }
    }

    auto toSecs = [&](clock_::duration d) { return std::chrono::duration<double>{d}.count(); };

    const auto totalMu = std::chrono::duration_cast<std::chrono::microseconds>(totalSimulationTimes.sum()).count();
    fmt::print("Simulation finished after: {}, avg step: {}\n", totalMu, totalMu / NSTEPS);
    fmt::print(
        plotFile,
        "\"P: {} F: {}\"\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        particleMappingName(ParticleMapping),
        fieldMappingName(FieldMapping),
        toSecs(totalSimulationTimes.clearCurrent),
        toSecs(totalSimulationTimes.integrateMotion),
        toSecs(totalSimulationTimes.depositCharge),
        toSecs(totalSimulationTimes.boundariesCurrent),
        toSecs(totalSimulationTimes.advanceB1),
        toSecs(totalSimulationTimes.boundariesB1),
        toSecs(totalSimulationTimes.advanceE),
        toSecs(totalSimulationTimes.boundariesE),
        toSecs(totalSimulationTimes.advanceB2),
        toSecs(totalSimulationTimes.boundariesB2),
        toSecs(totalSimulationTimes.sum()));
}

auto main() -> int
try
{
    const auto numThreads = static_cast<std::size_t>(omp_get_max_threads());
    const char* affinity = std::getenv("GOMP_CPU_AFFINITY"); // NOLINT(concurrency-mt-unsafe)
    affinity = affinity == nullptr ? "NONE - PLEASE PIN YOUR THREADS!" : affinity;

    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    auto accName = alpaka::getName(alpaka::getDevByIdx<alpaka::Pltf<alpaka::Dev<Acc>>>(0u));
    while(static_cast<bool>(std::isspace(accName.back())))
        accName.pop_back();
    fmt::print("Running {} steps with grid {}x{} and {}k particles on {}\n", NSTEPS, X_, Y_, numpart / 1000, accName);

    std::ofstream plotFile{"pic.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    fmt::print(
        plotFile,
        R"aa(#!/usr/bin/gnuplot -p
# threads: {} affinity: {}
set title "PIC grid {}x{} {}k particles on {} ({})"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
set ylabel "runtime [s]"
$data << EOD
""	"clr J"	"integr"	" dep J"	" bnd J"	"adv B1"	"bnd B1"	" adv E"	" bnd E"	"adv B2"	"bnd B2"	"total"
)aa",
        numThreads,
        affinity,
        X_,
        Y_,
        numpart / 1000,
        accName,
        common::hostname());


    // FieldMapping: AoS RM, AoS CM, AoS Mo,
    //               SoA RM, SoA CM, SoA Mo,
    //               SoA MB RM, SoA MB CM, SoA MB Mo,
    //               AoSoA
    // ParticleMapping: AoS, SoA, SoA MB, AoSoA
    run<0, 0>(plotFile);
    run<0, 2>(plotFile);
    run<6, 0>(plotFile);
    run<6, 2>(plotFile);
    plotFile << R"(EOD
plot $data using 2:xtic(1) ti col, "" using 3 ti col, "" using 4 ti col, "" using 5 ti col, "" using 6 ti col, "" using 7 ti col, "" using 8 ti col, "" using 9 ti col, "" using 10 ti col, "" using 11 ti col, "" using 12 ti col
)";
    std::cout << "Plot with: ./pic.sh\n";
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

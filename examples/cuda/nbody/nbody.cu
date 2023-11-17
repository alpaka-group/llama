// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "../../common/Stopwatch.hpp"
#include "../../common/hostname.hpp"

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <string>

using FP = float;

constexpr auto problemSize = 64 * 1024; ///< total number of particles
constexpr auto steps = 5; ///< number of steps to calculate
constexpr auto allowRsqrt = true; // rsqrt can be way faster, but less accurate
constexpr auto runUpate = true; // run update step. Useful to disable for benchmarking the move step.
constexpr auto countFieldAccesses = false;
constexpr auto heatmap = false;

constexpr auto sharedElementsPerBlock = 512;
constexpr auto threadsPerBlock = 256;
constexpr auto aosoaLanes = 32; // coalesced memory access

// makes our life easier for now
static_assert(problemSize % sharedElementsPerBlock == 0);
static_assert(sharedElementsPerBlock % threadsPerBlock == 0);

static_assert(!countFieldAccesses || !heatmap, "Cannot turn on FieldAccessCount and Heatmap at the same time");

constexpr auto timestep = FP{0.0001};
constexpr auto eps2 = FP{0.01};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static_assert(
    !countFieldAccesses && !heatmap,
    "Since tracing/heatmap is enabled, this example needs compute capability >= 60 for 64bit atomics");
#endif
using CountType = unsigned long long int;

using namespace std::literals;

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

using Vec3 = llama::Record<
    llama::Field<tag::X, FP>,
    llama::Field<tag::Y, FP>,
    llama::Field<tag::Z, FP>>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Vel, Vec3>,
    llama::Field<tag::Mass, FP>
    // adding a padding element would nicely align a single Particle to 8 floats
    //, llama::Field<llama::NoName, FP>
>;

using ParticleJ = llama::Record<
    llama::Field<tag::Pos, Vec3>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

// using SharedMemoryParticle = Particle;
using SharedMemoryParticle = ParticleJ;

template<typename ParticleRefI, typename ParticleRefJ>
__device__ void pPInteraction(ParticleRefI& pi, ParticleRefJ pj)
{
    auto dist = pi(tag::Pos{}) - pj(tag::Pos{});
    dist *= dist;
    const FP distSqr = eps2 + dist(tag::X{}) + dist(tag::Y{}) + dist(tag::Z{});
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = allowRsqrt ? rsqrt(distSixth) : (1.0f / sqrt(distSixth));
    const FP sts = pj(tag::Mass{}) * invDistCube * +timestep;
    pi(tag::Vel{}) += dist * sts;
}

template<int MappingSM, typename View>
__global__ void updateSM(View particles)
{
    auto sharedView = []
    {
        constexpr auto sharedMapping = []
        {
            using ArrayExtents = llama::ArrayExtents<int, sharedElementsPerBlock>;
            if constexpr(MappingSM == 0)
                return llama::mapping::AoS<ArrayExtents, SharedMemoryParticle>{};
            if constexpr(MappingSM == 1)
                return llama::mapping::SoA<ArrayExtents, SharedMemoryParticle, llama::mapping::Blobs::Single>{};
            if constexpr(MappingSM == 2)
                return llama::mapping::SoA<ArrayExtents, SharedMemoryParticle, llama::mapping::Blobs::OnePerField>{};
            if constexpr(MappingSM == 3)
                return llama::mapping::AoSoA<ArrayExtents, SharedMemoryParticle, aosoaLanes>{};
        }();

        llama::Array<std::byte*, decltype(sharedMapping)::blobCount> sharedMems{};
        boost::mp11::mp_for_each<boost::mp11::mp_iota_c<decltype(sharedMapping)::blobCount>>(
            [&](auto i)
            {
                __shared__ std::byte sharedMem[sharedMapping.blobSize(i)];
                sharedMems[i] = &sharedMem[0];
            });
        return llama::View{sharedMapping, sharedMems};
    }();

    const int ti = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
    const int tbi = static_cast<int>(blockIdx.x);

    llama::One<Particle> pi = particles(ti);
    for(int blockOffset = 0; blockOffset < problemSize; blockOffset += sharedElementsPerBlock)
    {
#pragma unroll
        for(int j = 0; j < sharedElementsPerBlock; j += threadsPerBlock)
            sharedView(j) = particles(blockOffset + tbi + j);
        __syncthreads();

        // Sometimes nvcc takes a bad unrolling decision and you need to unroll yourself
        // #pragma unroll 8
        for(int j = 0; j < sharedElementsPerBlock; ++j)
            pPInteraction(pi, sharedView(j));
        __syncthreads();
    }
    particles(ti)(tag::Vel{}) = pi(tag::Vel{});
}

template<typename View>
__global__ void update(View particles)
{
    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;

    llama::One<Particle> pi = particles(ti);
    for(int j = 0; j < problemSize; ++j)
        pPInteraction(pi, particles(j));
    particles(ti)(tag::Vel{}) = pi(tag::Vel{});
}

template<typename View>
__global__ void move(View particles)
{
    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;
    particles(ti)(tag::Pos()) += particles(ti)(tag::Vel()) * +timestep;
}

void checkError(cudaError_t code)
{
    if(code != cudaSuccess)
        throw std::runtime_error("CUDA Error: "s + cudaGetErrorString(code));
}

template<int Mapping, int MappingSM>
void run(std::ostream& plotFile, bool useSharedMemory)
try
{
    auto mappingName = [](int m) -> std::string
    {
        if(m == 0)
            return "AoS";
        if(m == 1)
            return "SoA";
        if(m == 2)
            return "SoA MB";
        if(m == 3)
            return "AoSoA" + std::to_string(aosoaLanes);
        if(m == 4)
            return "Split SoA";
        if(m == 5)
            return "Split AoS";
        std::abort();
    };
    auto title = "GM " + mappingName(Mapping);
    if(useSharedMemory)
        title += " SM " + mappingName(MappingSM);
    std::cout << '\n' << title << '\n';

    auto mapping = []
    {
        using ArrayExtents = llama::ArrayExtentsDynamic<int, 1>;
        const auto extents = ArrayExtents{problemSize};
        if constexpr(Mapping == 0)
            return llama::mapping::AoS<ArrayExtents, Particle>{extents};
        if constexpr(Mapping == 1)
            return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::Single>{extents};
        if constexpr(Mapping == 2)
            return llama::mapping::SoA<ArrayExtents, Particle, llama::mapping::Blobs::OnePerField>{extents};
        if constexpr(Mapping == 3)
            return llama::mapping::AoSoA<ArrayExtents, Particle, aosoaLanes>{extents};
        if constexpr(Mapping == 4)
            return llama::mapping::Split<
                ArrayExtents,
                Particle,
                llama::RecordCoord<1>,
                llama::mapping::BindSoA<>::fn,
                llama::mapping::BindSoA<>::fn,
                true>{extents};
        if constexpr(Mapping == 5)
            return llama::mapping::Split<
                ArrayExtents,
                Particle,
                llama::RecordCoord<1>,
                llama::mapping::BindSoA<>::fn,
                llama::mapping::BindSoA<>::fn,

                true>{extents};
    }();
    auto tmapping = [&]
    {
        if constexpr(countFieldAccesses)
            return llama::mapping::FieldAccessCount<std::decay_t<decltype(mapping)>, CountType>{mapping};
        else if constexpr(heatmap)
            return llama::mapping::Heatmap<std::decay_t<decltype(mapping)>, 1, CountType>{mapping};
        else
            return mapping;
    }();

    Stopwatch watch;

    auto hostView = llama::allocViewUninitialized(tmapping);
    auto accView = llama::allocViewUninitialized(tmapping, llama::bloballoc::CudaMalloc{});

    watch.printAndReset("alloc");

    std::default_random_engine engine;
    std::normal_distribution distribution(FP{0}, FP{1});
    for(int i = 0; i < problemSize; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(engine);
        p(tag::Pos(), tag::Y()) = distribution(engine);
        p(tag::Pos(), tag::Z()) = distribution(engine);
        p(tag::Vel(), tag::X()) = distribution(engine) / FP{10};
        p(tag::Vel(), tag::Y()) = distribution(engine) / FP{10};
        p(tag::Vel(), tag::Z()) = distribution(engine) / FP{10};
        p(tag::Mass()) = distribution(engine) / FP{100};
        hostView(i) = p;
    }
    if constexpr(countFieldAccesses)
        hostView.mapping().fieldHits(hostView.blobs()) = {};

    watch.printAndReset("init");

    cudaEvent_t startEvent = {};
    cudaEvent_t stopEvent = {};
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    auto start = [&] { checkError(cudaEventRecord(startEvent)); };
    auto stop = [&]
    {
        checkError(cudaEventRecord(stopEvent));
        checkError(cudaEventSynchronize(stopEvent));
        float milliseconds = 0;
        checkError(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
        return milliseconds / 1000;
    };

    start();
    const auto blobs = hostView.blobs().size() / (heatmap ? 2 : 1); // exclude heatmap blobs
    for(std::size_t i = 0; i < blobs; i++)
        checkError(cudaMemcpy(
            &accView.blobs()[i][0],
            &hostView.blobs()[i][0],
            hostView.mapping().blobSize(i),
            cudaMemcpyHostToDevice));
    if constexpr(heatmap)
    {
        auto& hmap = accView.mapping();
        for(std::size_t i = 0; i < blobs; i++)
            cudaMemsetAsync(hmap.blockHitsPtr(i, accView.blobs()), 0, hmap.blockHitsSize(i) * sizeof(CountType));
    }
    std::cout << "copy H->D " << stop() << " s\n";

    const auto blocks = problemSize / threadsPerBlock;

    double sumUpdate = 0;
    double sumMove = 0;
    for(int s = 0; s < steps; ++s)
    {
        if constexpr(runUpate)
        {
            start();
            if(useSharedMemory)
                updateSM<MappingSM><<<blocks, threadsPerBlock>>>(llama::shallowCopy(accView));
            else
                update<<<blocks, threadsPerBlock>>>(llama::shallowCopy(accView));
            const auto secondsUpdate = stop();
            std::cout << "update " << secondsUpdate << " s\t";
            sumUpdate += secondsUpdate;
        }

        start();
        ::move<<<blocks, threadsPerBlock>>>(llama::shallowCopy(accView));
        const auto secondsMove = stop();
        std::cout << "move " << secondsMove << " s\n";
        sumMove += secondsMove;
    }
    plotFile << std::quoted(title) << "\t" << sumUpdate / steps << '\t' << sumMove / steps << '\n';

    start();
    for(std::size_t i = 0; i < hostView.blobs().size(); i++)
        checkError(cudaMemcpy(
            &hostView.blobs()[i][0],
            &accView.blobs()[i][0],
            hostView.mapping().blobSize(i),
            cudaMemcpyDeviceToHost));
    std::cout << "copy D->H " << stop() << " s\n";

    if constexpr(countFieldAccesses)
    {
        hostView.mapping().printFieldHits(hostView.blobs());
    }
    else if constexpr(heatmap)
    {
        auto titleCopy = title;
        for(char& c : titleCopy)
            if(c == ' ')
                c = '_';
        std::ofstream{"plot_heatmap.sh"} << hostView.mapping().gnuplotScript;
        hostView.mapping().writeGnuplotDataFile(
            hostView.blobs(),
            std::ofstream{"cuda_nbody_heatmap_" + titleCopy + ".dat"});
    }

    checkError(cudaEventDestroy(startEvent));
    checkError(cudaEventDestroy(stopEvent));
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << std::endl;
}

// based on:
// https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
// The original GPU gems implementation is with THREADS_PER_BLOCK == SHARED_ELEMENTS_PER_BLOCK
namespace manual
{
    using FP3 = std::conditional_t<std::is_same_v<FP, float>, float3, double3>;
    using FP4 = std::conditional_t<std::is_same_v<FP, float>, float4, double4>;

    __device__ auto bodyBodyInteraction(FP4 bi, FP4 bj, FP3 ai) -> FP3
    {
        FP3 r;
        r.x = bj.x - bi.x;
        r.y = bj.y - bi.y;
        r.z = bj.z - bi.z;
        FP distSqr = r.x * r.x + r.y * r.y + r.z * r.z + eps2;
        FP distSixth = distSqr * distSqr * distSqr;
        FP invDistCube = allowRsqrt ? rsqrt(distSixth) : (1.0f / sqrtf(distSixth));
        FP s = bj.w * invDistCube;
        ai.x += r.x * s * +timestep;
        ai.y += r.y * s * +timestep;
        ai.z += r.z * s * +timestep;
        return ai;
    }

    __shared__ FP4 shPosition[sharedElementsPerBlock];

    __device__ auto tileCalculation(FP4 myPosition, FP3 accel) -> FP3
    {
        for(auto& i : shPosition)
            accel = bodyBodyInteraction(myPosition, i, accel);
        return accel;
    }

    __global__ void calculateForces(const FP4* globalX, FP4* globalA)
    {
        FP3 acc = {0.0f, 0.0f, 0.0f};
        const unsigned gtid = blockIdx.x * threadsPerBlock + threadIdx.x;
        const FP4 myPosition = globalX[gtid];
        for(unsigned i = 0, tile = 0; i < problemSize; i += sharedElementsPerBlock, tile++)
        {
            for(unsigned j = threadIdx.x; j < sharedElementsPerBlock; j += threadsPerBlock)
                shPosition[j] = globalX[tile * sharedElementsPerBlock + j];
            __syncthreads();
            acc = tileCalculation(myPosition, acc);
            __syncthreads();
        }
        globalA[gtid] = {acc.x, acc.y, acc.z, 0.0f};
    }

    __global__ void move(FP4* globalX, const FP4* globalA)
    {
        const unsigned gtid = blockIdx.x * threadsPerBlock + threadIdx.x;
        FP4 pos = globalX[gtid];
        const FP4 vel = globalA[gtid];
        pos.x += vel.x * +timestep;
        pos.y += vel.y * +timestep;
        pos.z += vel.z * +timestep;
        globalX[gtid] = pos;
    }

    void run(std::ostream& plotFile)
    try
    {
        const auto* title = "GPU gems";
        std::cout << '\n' << title << '\n';

        Stopwatch watch;

        auto hostPositions = std::vector<FP4>(problemSize);
        auto hostVelocities = std::vector<FP4>(problemSize);

        FP4* accPositions = nullptr;
        checkError(cudaMalloc(&accPositions, problemSize * sizeof(FP4)));
        FP4* accVelocities = nullptr;
        checkError(cudaMalloc(&accVelocities, problemSize * sizeof(FP4)));

        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> distribution(FP{0}, FP{1});
        for(int i = 0; i < problemSize; ++i)
        {
            hostPositions[i].x = distribution(engine);
            hostPositions[i].y = distribution(engine);
            hostPositions[i].z = distribution(engine);
            hostVelocities[i].x = distribution(engine);
            hostVelocities[i].y = distribution(engine);
            hostVelocities[i].z = distribution(engine);
            hostPositions[i].w = distribution(engine);
        }

        watch.printAndReset("init");

        cudaEvent_t startEvent = {};
        cudaEvent_t stopEvent = {};
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        auto start = [&] { checkError(cudaEventRecord(startEvent)); };
        auto stop = [&]
        {
            checkError(cudaEventRecord(stopEvent));
            checkError(cudaEventSynchronize(stopEvent));
            float milliseconds = 0;
            checkError(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
            return milliseconds / 1000;
        };

        start();
        checkError(cudaMemcpy(accPositions, hostPositions.data(), problemSize * sizeof(FP4), cudaMemcpyHostToDevice));
        checkError(
            cudaMemcpy(accVelocities, hostVelocities.data(), problemSize * sizeof(FP4), cudaMemcpyHostToDevice));
        std::cout << "copy H->D " << stop() << " s\n";

        const auto blocks = problemSize / threadsPerBlock;

        double sumUpdate = 0;
        double sumMove = 0;
        for(int s = 0; s < steps; ++s)
        {
            if constexpr(runUpate)
            {
                start();
                calculateForces<<<blocks, threadsPerBlock>>>(accPositions, accVelocities);
                const auto secondsUpdate = stop();
                std::cout << "update " << secondsUpdate << " s\t";
                sumUpdate += secondsUpdate;
            }

            start();
            move<<<blocks, threadsPerBlock>>>(accPositions, accVelocities);
            const auto secondsMove = stop();
            std::cout << "move " << secondsMove << " s\n";
            sumMove += secondsMove;
        }
        plotFile << std::quoted(title) << "\t" << sumUpdate / steps << '\t' << sumMove / steps << '\n';

        start();
        checkError(cudaMemcpy(hostPositions.data(), accPositions, problemSize * sizeof(FP4), cudaMemcpyDeviceToHost));
        checkError(
            cudaMemcpy(hostVelocities.data(), accVelocities, problemSize * sizeof(FP4), cudaMemcpyDeviceToHost));
        std::cout << "copy D->H " << stop() << " s\n";

        checkError(cudaFree(accPositions));
        checkError(cudaFree(accVelocities));
        checkError(cudaEventDestroy(startEvent));
        checkError(cudaEventDestroy(stopEvent));
    }
    catch(const std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
} // namespace manual

auto main() -> int
try
{
    std::cout << problemSize / 1024 << "ki particles (" << problemSize * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << sharedElementsPerBlock << " particles ("
              << sharedElementsPerBlock * llama::sizeOf<SharedMemoryParticle> / 1024 << " kiB) in shared memory\n"
              << "Using " << threadsPerBlock << " per block\n";
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    fmt::print(
        "Running on {}, {}MiB GM, {}kiB SM\n",
        prop.name,
        prop.totalGlobalMem / 1024 / 1024,
        prop.sharedMemPerBlock / 1024);
    std::cout << std::fixed;

    std::ofstream plotFile{"nbody_cuda.sh"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "nbody CUDA {}ki particles on {}"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
set y2range [0:*]
set ylabel "update runtime [s]"
set y2label "move runtime [s]"
set y2tics auto
$data << EOD
)",
        problemSize / 1024,
        common::hostname());
    plotFile << "\"\"\t\"update\"\t\"move\"\n";

    using namespace boost::mp11;
    mp_for_each<mp_iota_c<6>>([&](auto i) { run<decltype(i)::value, 0>(plotFile, false); });
    mp_for_each<mp_iota_c<6>>(
        [&](auto i)
        { mp_for_each<mp_iota_c<4>>([&](auto j) { run<decltype(i)::value, decltype(j)::value>(plotFile, true); }); });
    manual::run(plotFile);

    plotFile <<
        R"(EOD
plot $data using 2:xtic(1) ti col axis x1y1, "" using 3 ti col axis x1y2
)";
    std::cout << "Plot with: ./nbody_cuda.sh\n";

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

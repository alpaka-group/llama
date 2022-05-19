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

constexpr auto PROBLEM_SIZE = 64 * 1024; ///< total number of particles
constexpr auto SHARED_ELEMENTS_PER_BLOCK = 512;
constexpr auto STEPS = 5; ///< number of steps to calculate
constexpr auto ALLOW_RSQRT = true; // rsqrt can be way faster, but less accurate
constexpr auto RUN_UPATE = true; // run update step. Useful to disable for benchmarking the move step.
constexpr auto TRACE = true;
constexpr FP TIMESTEP = 0.0001f;

constexpr auto THREADS_PER_BLOCK = 256;
constexpr auto AOSOA_LANES = 32; // coalesced memory access

// makes our life easier for now
static_assert(PROBLEM_SIZE % SHARED_ELEMENTS_PER_BLOCK == 0);
static_assert(SHARED_ELEMENTS_PER_BLOCK % THREADS_PER_BLOCK == 0);

constexpr FP EPS2 = 0.01;

#if __CUDA_ARCH__ >= 600
using CountType = unsigned long long int;
#else
using CountType = unsigned;
#endif

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

using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>>,
    llama::Field<tag::Vel, llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>>,
    llama::Field<tag::Mass, FP>
    // adding a padding element would nicely align a single Particle to 8 floats
    //, llama::Field<llama::NoName, FP>
>;

using ParticleJ = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, FP>,
        llama::Field<tag::Y, FP>,
        llama::Field<tag::Z, FP>>>,
    llama::Field<tag::Mass, FP>>;
// clang-format on

// using SharedMemoryParticle = Particle;
using SharedMemoryParticle = ParticleJ;

template<typename VirtualParticleI, typename VirtualParticleJ>
__device__ void pPInteraction(VirtualParticleI&& pi, VirtualParticleJ pj)
{
    auto dist = pi(tag::Pos()) - pj(tag::Pos());
    dist *= dist;
    const FP distSqr = EPS2 + dist(tag::X()) + dist(tag::Y()) + dist(tag::Z());
    const FP distSixth = distSqr * distSqr * distSqr;
    const FP invDistCube = ALLOW_RSQRT ? rsqrt(distSixth) : (1.0f / sqrt(distSixth));
    const FP sts = pj(tag::Mass()) * invDistCube * +TIMESTEP;
    pi(tag::Vel()) += dist * sts;
}

template<int MappingSM, typename View>
__global__ void updateSM(View particles)
{
    // FIXME: removing this lambda makes nvcc 11 segfault
    auto sharedView = []
    {
        constexpr auto sharedMapping = []
        {
            using ArrayExtents = llama::ArrayExtents<int, SHARED_ELEMENTS_PER_BLOCK>;
            if constexpr(MappingSM == 0)
                return llama::mapping::AoS<ArrayExtents, SharedMemoryParticle>{};
            if constexpr(MappingSM == 1)
                return llama::mapping::SoA<ArrayExtents, SharedMemoryParticle, false>{};
            if constexpr(MappingSM == 2)
                return llama::mapping::SoA<ArrayExtents, SharedMemoryParticle, true>{};
            if constexpr(MappingSM == 3)
                return llama::mapping::AoSoA<ArrayExtents, SharedMemoryParticle, AOSOA_LANES>{};
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

    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;
    const auto tbi = blockIdx.x;

    llama::One<Particle> pi = particles(ti);
    for(int blockOffset = 0; blockOffset < PROBLEM_SIZE; blockOffset += SHARED_ELEMENTS_PER_BLOCK)
    {
        LLAMA_INDEPENDENT_DATA
        for(int j = tbi; j < SHARED_ELEMENTS_PER_BLOCK; j += THREADS_PER_BLOCK)
            sharedView(j) = particles(blockOffset + j);
        __syncthreads();

        LLAMA_INDEPENDENT_DATA
        for(int j = 0; j < SHARED_ELEMENTS_PER_BLOCK; ++j)
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
    LLAMA_INDEPENDENT_DATA
    for(int j = 0; j < PROBLEM_SIZE; ++j)
        pPInteraction(pi, particles(j));
    particles(ti)(tag::Vel{}) = pi(tag::Vel{});
}

template<typename View>
__global__ void move(View particles)
{
    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;
    particles(ti)(tag::Pos()) += particles(ti)(tag::Vel()) * +TIMESTEP;
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
            return "AoSoA" + std::to_string(AOSOA_LANES);
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
        const auto extents = ArrayExtents{PROBLEM_SIZE};
        if constexpr(Mapping == 0)
            return llama::mapping::AoS<ArrayExtents, Particle>{extents};
        if constexpr(Mapping == 1)
            return llama::mapping::SoA<ArrayExtents, Particle, false>{extents};
        if constexpr(Mapping == 2)
            return llama::mapping::SoA<ArrayExtents, Particle, true>{extents};
        if constexpr(Mapping == 3)
            return llama::mapping::AoSoA<ArrayExtents, Particle, AOSOA_LANES>{extents};
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
        if constexpr(TRACE)
            return llama::mapping::Trace<std::decay_t<decltype(mapping)>, CountType>{mapping};
        else
            return mapping;
    }();

    Stopwatch watch;

    auto hostView = llama::allocViewUninitialized(tmapping);
    auto accView = llama::allocViewUninitialized(
        tmapping,
        [](auto alignment, std::size_t size)
        {
            std::byte* p = nullptr;
            checkError(cudaMalloc(&p, size));
            if(reinterpret_cast<std::uintptr_t>(p) & (alignment - 1 != 0u))
                throw std::runtime_error{"cudaMalloc does not align sufficiently"};
            return p;
        });

    watch.printAndReset("alloc");

    std::default_random_engine engine;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for(int i = 0; i < PROBLEM_SIZE; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(engine);
        p(tag::Pos(), tag::Y()) = distribution(engine);
        p(tag::Pos(), tag::Z()) = distribution(engine);
        p(tag::Vel(), tag::X()) = distribution(engine) / FP(10);
        p(tag::Vel(), tag::Y()) = distribution(engine) / FP(10);
        p(tag::Vel(), tag::Z()) = distribution(engine) / FP(10);
        p(tag::Mass()) = distribution(engine) / FP(100);
        hostView(i) = p;
    }
    if constexpr(TRACE)
        hostView.mapping().fieldHits(hostView.storageBlobs) = {};

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
    const auto blobs = hostView.storageBlobs.size();
    for(std::size_t i = 0; i < blobs; i++)
        checkError(cudaMemcpy(
            accView.storageBlobs[i],
            hostView.storageBlobs[i].data(),
            hostView.mapping().blobSize(i),
            cudaMemcpyHostToDevice));
    if constexpr(TRACE)
        cudaMemset(accView.storageBlobs[blobs], 0, accView.mapping().blobSize(blobs)); // init trace count buffer
    std::cout << "copy H->D " << stop() << " s\n";

    const auto blocks = PROBLEM_SIZE / THREADS_PER_BLOCK;

    double sumUpdate = 0;
    double sumMove = 0;
    for(int s = 0; s < STEPS; ++s)
    {
        if constexpr(RUN_UPATE)
        {
            start();
            if(useSharedMemory)
                updateSM<MappingSM><<<blocks, THREADS_PER_BLOCK>>>(accView);
            else
                update<<<blocks, THREADS_PER_BLOCK>>>(accView);
            const auto secondsUpdate = stop();
            std::cout << "update " << secondsUpdate << " s\t";
            sumUpdate += secondsUpdate;
        }

        start();
        ::move<<<blocks, THREADS_PER_BLOCK>>>(accView);
        const auto secondsMove = stop();
        std::cout << "move " << secondsMove << " s\n";
        sumMove += secondsMove;
    }
    plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

    start();
    for(std::size_t i = 0; i < blobs; i++)
        checkError(cudaMemcpy(
            hostView.storageBlobs[i].data(),
            accView.storageBlobs[i],
            hostView.mapping().blobSize(i),
            cudaMemcpyDeviceToHost));
    std::cout << "copy D->H " << stop() << " s\n";

    if constexpr(TRACE)
        hostView.mapping().printFieldHits(hostView.storageBlobs);

    for(std::size_t i = 0; i < accView.storageBlobs.size(); i++)
        checkError(cudaFree(accView.storageBlobs[i]));
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
        FP distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
        FP distSixth = distSqr * distSqr * distSqr;
        FP invDistCube = ALLOW_RSQRT ? rsqrt(distSixth) : (1.0f / sqrtf(distSixth));
        FP s = bj.w * invDistCube;
        ai.x += r.x * s * +TIMESTEP;
        ai.y += r.y * s * +TIMESTEP;
        ai.z += r.z * s * +TIMESTEP;
        return ai;
    }

    __shared__ FP4 shPosition[SHARED_ELEMENTS_PER_BLOCK];

    __device__ auto tile_calculation(FP4 myPosition, FP3 accel) -> FP3
    {
        for(auto& i : shPosition)
            accel = bodyBodyInteraction(myPosition, i, accel);
        return accel;
    }

    __global__ void calculate_forces(const FP4* globalX, FP4* globalA)
    {
        FP3 acc = {0.0f, 0.0f, 0.0f};
        const unsigned gtid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
        const FP4 myPosition = globalX[gtid];
        for(unsigned i = 0, tile = 0; i < PROBLEM_SIZE; i += SHARED_ELEMENTS_PER_BLOCK, tile++)
        {
            for(unsigned j = threadIdx.x; j < SHARED_ELEMENTS_PER_BLOCK; j += THREADS_PER_BLOCK)
                shPosition[j] = globalX[tile * SHARED_ELEMENTS_PER_BLOCK + j];
            __syncthreads();
            acc = tile_calculation(myPosition, acc);
            __syncthreads();
        }
        globalA[gtid] = {acc.x, acc.y, acc.z, 0.0f};
    }

    __global__ void move(FP4* globalX, const FP4* globalA)
    {
        const unsigned gtid = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
        FP4 pos = globalX[gtid];
        const FP4 vel = globalA[gtid];
        pos.x += vel.x * +TIMESTEP;
        pos.y += vel.y * +TIMESTEP;
        pos.z += vel.z * +TIMESTEP;
        globalX[gtid] = pos;
    }

    void run(std::ostream& plotFile)
    try
    {
        const auto* title = "GPU gems";
        std::cout << '\n' << title << '\n';

        Stopwatch watch;

        auto hostPositions = std::vector<FP4>(PROBLEM_SIZE);
        auto hostVelocities = std::vector<FP4>(PROBLEM_SIZE);

        FP4* accPositions = nullptr;
        checkError(cudaMalloc(&accPositions, PROBLEM_SIZE * sizeof(FP4)));
        FP4* accVelocities = nullptr;
        checkError(cudaMalloc(&accVelocities, PROBLEM_SIZE * sizeof(FP4)));

        watch.printAndReset("alloc");

        std::default_random_engine engine;
        std::normal_distribution<FP> distribution(FP(0), FP(1));
        for(int i = 0; i < PROBLEM_SIZE; ++i)
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
        checkError(cudaMemcpy(accPositions, hostPositions.data(), PROBLEM_SIZE * sizeof(FP4), cudaMemcpyHostToDevice));
        checkError(
            cudaMemcpy(accVelocities, hostVelocities.data(), PROBLEM_SIZE * sizeof(FP4), cudaMemcpyHostToDevice));
        std::cout << "copy H->D " << stop() << " s\n";

        const auto blocks = PROBLEM_SIZE / THREADS_PER_BLOCK;

        double sumUpdate = 0;
        double sumMove = 0;
        for(int s = 0; s < STEPS; ++s)
        {
            if constexpr(RUN_UPATE)
            {
                start();
                calculate_forces<<<blocks, THREADS_PER_BLOCK>>>(accPositions, accVelocities);
                const auto secondsUpdate = stop();
                std::cout << "update " << secondsUpdate << " s\t";
                sumUpdate += secondsUpdate;
            }

            start();
            move<<<blocks, THREADS_PER_BLOCK>>>(accPositions, accVelocities);
            const auto secondsMove = stop();
            std::cout << "move " << secondsMove << " s\n";
            sumMove += secondsMove;
        }
        plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

        start();
        checkError(cudaMemcpy(hostPositions.data(), accPositions, PROBLEM_SIZE * sizeof(FP4), cudaMemcpyDeviceToHost));
        checkError(
            cudaMemcpy(hostVelocities.data(), accVelocities, PROBLEM_SIZE * sizeof(FP4), cudaMemcpyDeviceToHost));
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
    std::cout << PROBLEM_SIZE / 1024 << "ki particles (" << PROBLEM_SIZE * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << SHARED_ELEMENTS_PER_BLOCK << " particles ("
              << SHARED_ELEMENTS_PER_BLOCK * llama::sizeOf<SharedMemoryParticle> / 1024 << " kiB) in shared memory\n"
              << "Using " << THREADS_PER_BLOCK << " per block\n";
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
        PROBLEM_SIZE / 1024,
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

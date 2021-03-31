#include "../../common/Stopwatch.hpp"

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <string>
#include <utility>

using FP = float;

constexpr auto PROBLEM_SIZE = 64 * 1024; ///< total number of particles
constexpr auto SHARED_ELEMENTS_PER_BLOCK = 512;
constexpr auto STEPS = 5; ///< number of steps to calculate
constexpr auto ALLOW_RSQRT = true; // rsqrt can be way faster, but less accurate
constexpr FP TIMESTEP = 0.0001f;

constexpr auto THREADS_PER_BLOCK = 256;
constexpr auto AOSOA_LANES = 32; // coalesced memory access

// makes our life easier for now
static_assert(PROBLEM_SIZE % SHARED_ELEMENTS_PER_BLOCK == 0);
static_assert(SHARED_ELEMENTS_PER_BLOCK % THREADS_PER_BLOCK == 0);

constexpr FP EPS2 = 0.01;

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

template <typename VirtualParticleI, typename VirtualParticleJ>
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

template <std::size_t ProblemSize, bool UseAccumulator, std::size_t BlockSize, int MappingSM, typename View>
__global__ void updateSM(View particles)
{
    // FIXME: removing this lambda makes nvcc 11 segfault
    auto sharedView = [] {
        constexpr auto sharedMapping = [] {
            constexpr auto arrayDomain = llama::ArrayDomain{BlockSize};
            if constexpr (MappingSM == 0)
                return llama::mapping::AoS{arrayDomain, SharedMemoryParticle{}};
            if constexpr (MappingSM == 1)
                return llama::mapping::SoA{arrayDomain, SharedMemoryParticle{}};
            if constexpr (MappingSM == 2)
                return llama::mapping::SoA{arrayDomain, SharedMemoryParticle{}, std::true_type{}};
            if constexpr (MappingSM == 3)
                return llama::mapping::AoSoA<decltype(arrayDomain), SharedMemoryParticle, AOSOA_LANES>{arrayDomain};
        }();

        llama::Array<std::byte*, decltype(sharedMapping)::blobCount> sharedMems{};
        boost::mp11::mp_for_each<boost::mp11::mp_iota_c<decltype(sharedMapping)::blobCount>>([&](auto i) {
            __shared__ std::byte sharedMem[sharedMapping.blobSize(i)];
            sharedMems[i] = &sharedMem[0];
        });
        return llama::View{sharedMapping, sharedMems};
    }();

    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;
    const auto tbi = blockIdx.x;

    llama::One<Particle> pi;
    if constexpr (UseAccumulator)
        pi = particles(ti);
    for (std::size_t blockOffset = 0; blockOffset < ProblemSize; blockOffset += BlockSize)
    {
        LLAMA_INDEPENDENT_DATA
        for (auto j = tbi; j < BlockSize; j += THREADS_PER_BLOCK)
            sharedView(j) = particles(blockOffset + j);
        __syncthreads();

        LLAMA_INDEPENDENT_DATA
        for (auto j = std::size_t{0}; j < BlockSize; ++j)
        {
            if constexpr (UseAccumulator)
                pPInteraction(pi, sharedView(j));
            else
                pPInteraction(particles(ti), sharedView(j));
        }
        __syncthreads();
    }
    if constexpr (UseAccumulator)
        particles(ti) = pi;
}

template <std::size_t ProblemSize, bool UseAccumulator, typename View>
__global__ void update(View particles)
{
    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;

    llama::One<Particle> pi;
    if constexpr (UseAccumulator)
        pi = particles(ti);
    LLAMA_INDEPENDENT_DATA
    for (auto j = std::size_t{0}; j < ProblemSize; ++j)
    {
        if constexpr (UseAccumulator)
            pPInteraction(pi, particles(j));
        else
            pPInteraction(particles(ti), particles(j));
    }
    if constexpr (UseAccumulator)
        particles(ti) = pi;
}

template <std::size_t ProblemSize, typename View>
__global__ void move(View particles)
{
    const auto ti = threadIdx.x + blockIdx.x * blockDim.x;
    particles(ti)(tag::Pos()) += particles(ti)(tag::Vel()) * +TIMESTEP;
}

void checkError(cudaError_t code)
{
    if (code != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(code));
}

template <int Mapping, int MappingSM, bool UseAccumulator>
void run(std::ostream& plotFile, bool useSharedMemory)
try
{
    auto mappingName = [](int m) -> std::string {
        if (m == 0)
            return "AoS";
        if (m == 1)
            return "SoA";
        if (m == 2)
            return "SoA MB";
        if (m == 3)
            return "AoSoA" + std::to_string(AOSOA_LANES);
        if (m == 4)
            return "Split SoA";
    };
    auto title = "GM " + mappingName(Mapping);
    if (useSharedMemory)
        title += " SM " + mappingName(MappingSM);
    if (UseAccumulator)
        title += " Acc";
    std::cout << '\n' << title << '\n';

    auto mapping = [] {
        const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};
        if constexpr (Mapping == 0)
            return llama::mapping::AoS{arrayDomain, Particle{}};
        if constexpr (Mapping == 1)
            return llama::mapping::SoA{arrayDomain, Particle{}};
        if constexpr (Mapping == 2)
            return llama::mapping::SoA{arrayDomain, Particle{}, std::true_type{}};
        if constexpr (Mapping == 3)
            return llama::mapping::AoSoA<decltype(arrayDomain), Particle, AOSOA_LANES>{arrayDomain};
        if constexpr (Mapping == 4)
            return llama::mapping::Split<
                decltype(arrayDomain),
                Particle,
                llama::RecordCoord<1>,
                llama::mapping::SoA,
                llama::mapping::SoA,
                true>{arrayDomain};
    }();

    Stopwatch watch;

    auto hostView = llama::allocView(mapping);
    auto accView = llama::allocView(mapping, [](std::size_t size) {
        std::byte* p;
        checkError(cudaMalloc(&p, size));
        return p;
    });

    watch.printAndReset("alloc");

    std::mt19937_64 generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        llama::One<Particle> p;
        p(tag::Pos(), tag::X()) = distribution(generator);
        p(tag::Pos(), tag::Y()) = distribution(generator);
        p(tag::Pos(), tag::Z()) = distribution(generator);
        p(tag::Vel(), tag::X()) = distribution(generator) / FP(10);
        p(tag::Vel(), tag::Y()) = distribution(generator) / FP(10);
        p(tag::Vel(), tag::Z()) = distribution(generator) / FP(10);
        p(tag::Mass()) = distribution(generator) / FP(100);
        hostView(i) = p;
    }

    watch.printAndReset("init");

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    auto start = [&] { checkError(cudaEventRecord(startEvent)); };
    auto stop = [&] {
        checkError(cudaEventRecord(stopEvent));
        checkError(cudaEventSynchronize(stopEvent));
        float milliseconds = 0;
        checkError(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
        return milliseconds / 1000;
    };

    start();
    for (auto i = 0; i < accView.storageBlobs.rank; i++)
        checkError(cudaMemcpy(
            accView.storageBlobs[i],
            hostView.storageBlobs[i].data(),
            mapping.blobSize(i),
            cudaMemcpyHostToDevice));
    std::cout << "copy H->D " << stop() << " s\n";

    const auto blocks = PROBLEM_SIZE / THREADS_PER_BLOCK;

    double sumUpdate = 0;
    double sumMove = 0;
    for (std::size_t s = 0; s < STEPS; ++s)
    {
        start();
        if (useSharedMemory)
            updateSM<PROBLEM_SIZE, UseAccumulator, SHARED_ELEMENTS_PER_BLOCK, MappingSM>
                <<<blocks, THREADS_PER_BLOCK>>>(accView);
        else
            update<PROBLEM_SIZE, UseAccumulator><<<blocks, THREADS_PER_BLOCK>>>(accView);
        const auto secondsUpdate = stop();
        std::cout << "update " << secondsUpdate << " s\t";
        sumUpdate += secondsUpdate;

        start();
        move<PROBLEM_SIZE><<<blocks, THREADS_PER_BLOCK>>>(accView);
        const auto secondsMove = stop();
        std::cout << "move " << secondsMove << " s\n";
        sumMove += secondsMove;
    }
    if (!UseAccumulator)
        plotFile << std::quoted(title) << "\t" << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\t';
    else
        plotFile << sumUpdate / STEPS << '\t' << sumMove / STEPS << '\n';

    start();
    for (auto i = 0; i < accView.storageBlobs.rank; i++)
        checkError(cudaMemcpy(
            hostView.storageBlobs[i].data(),
            accView.storageBlobs[i],
            mapping.blobSize(i),
            cudaMemcpyDeviceToHost));
    std::cout << "copy D->H " << stop() << " s\n";

    for (auto i = 0; i < accView.storageBlobs.rank; i++)
        checkError(cudaFree(accView.storageBlobs[i]));
    checkError(cudaEventDestroy(startEvent));
    checkError(cudaEventDestroy(stopEvent));
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << std::endl;
}

int main()
try
{
    std::cout << PROBLEM_SIZE / 1000 << "k particles (" << PROBLEM_SIZE * llama::sizeOf<Particle> / 1024 << "kiB)\n"
              << "Caching " << SHARED_ELEMENTS_PER_BLOCK << " particles ("
              << SHARED_ELEMENTS_PER_BLOCK * llama::sizeOf<SharedMemoryParticle> / 1024 << " kiB) in shared memory\n"
              << "Using " << THREADS_PER_BLOCK << " per block\n";
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Running on " << prop.name << " " << prop.sharedMemPerBlock / 1024 << "kiB SM\n";
    std::cout << std::fixed;

    std::ofstream plotFile{"nbody.tsv"};
    plotFile.exceptions(std::ios::badbit | std::ios::failbit);
    plotFile << "\"\"\t\"update\"\t\"move\"\t\"update with acc\"\t\"move with acc\"\n";

    using namespace boost::mp11;
    mp_for_each<mp_iota_c<5>>([&](auto i) {
        mp_for_each<mp_list_c<bool, false, true>>(
            [&](auto useAccumulator) { run<decltype(i)::value, 0, decltype(useAccumulator)::value>(plotFile, false); });
    });
    mp_for_each<mp_iota_c<5>>([&](auto i) {
        mp_for_each<mp_iota_c<4>>([&](auto j) {
            mp_for_each<mp_list_c<bool, false, true>>([&](auto useAccumulator) {
                run<decltype(i)::value, decltype(j)::value, decltype(useAccumulator)::value>(plotFile, true);
            });
        });
    });

    std::cout << "Plot with: ./nbody.sh\n";
    std::ofstream{"nbody.sh"} << fmt::format(
        R"(#!/usr/bin/gnuplot -p
set title "nbody CUDA {0}k particles"
set style data histograms
set style fill solid
set xtics rotate by 45 right
set key out top center maxrows 3
set yrange [0:*]
plot 'nbody.tsv' using 2:xtic(1) ti col, "" using 4 ti col
)",
        PROBLEM_SIZE / 1000);

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

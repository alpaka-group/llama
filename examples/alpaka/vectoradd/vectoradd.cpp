/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../../common/Stopwatch.hpp"
#include "../../common/alpakaHelpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto MAPPING
    = 1; ///< 0 native AoS, 1 native SoA, 2 native SoA (separate blobs, does not work yet), 3 tree AoS, 4 tree SoA
constexpr auto PROBLEM_SIZE = 64 * 1024 * 1024;
constexpr auto BLOCK_SIZE = 256;
constexpr auto STEPS = 10;

using FP = float;

// clang-format off
namespace tag
{
    struct X{};
    struct Y{};
    struct Z{};
} // namespace tag

using Vector = llama::DS<
    llama::DE<tag::X, FP>,
    llama::DE<tag::Y, FP>,
    llama::DE<tag::Z, FP>>;
// clang-format on

template <std::size_t ProblemSize, std::size_t Elems>
struct AddKernel
{
    template <typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View a, View b) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);

        LLAMA_INDEPENDENT_DATA
        for (auto i = start; i < end; ++i)
        {
            a(i)(tag::X{}) += b(i)(tag::X{});
            a(i)(tag::Y{}) -= b(i)(tag::Y{});
            a(i)(tag::Z{}) *= b(i)(tag::Z{});
        }
    }
};

auto main() -> int
try
{
    // ALPAKA
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;

    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    // using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::AccCpuSerial<Dim, Size>;

    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;
    const DevAcc devAcc(alpaka::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    // LLAMA
    const auto arrayDomain = llama::ArrayDomain{PROBLEM_SIZE};

    const auto mapping = [&] {
        if constexpr (MAPPING == 0)
            return llama::mapping::AoS{arrayDomain, Vector{}};
        if constexpr (MAPPING == 1)
            return llama::mapping::SoA{arrayDomain, Vector{}};
        if constexpr (MAPPING == 2)
            return llama::mapping::SoA<decltype(arrayDomain), Vector, true>{arrayDomain};
        if constexpr (MAPPING == 3)
            return llama::mapping::tree::Mapping{arrayDomain, llama::Tuple{}, Vector{}};
        if constexpr (MAPPING == 4)
            return llama::mapping::tree::Mapping{
                arrayDomain,
                llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()},
                Vector{}};
    }();

    std::cout << PROBLEM_SIZE / 1000 / 1000 << " million vectors\n"
              << PROBLEM_SIZE * llama::sizeOf<Vector> * 2 / 1000 / 1000 << " MB on device\n";

    Stopwatch chrono;

    const auto bufferSize = Size(mapping.blobSize(0));

    // allocate buffers
    auto hostBufferA = alpaka::allocBuf<std::byte, Size>(devHost, bufferSize);
    auto hostBufferB = alpaka::allocBuf<std::byte, Size>(devHost, bufferSize);
    auto devBufferA = alpaka::allocBuf<std::byte, Size>(devAcc, bufferSize);
    auto devBufferB = alpaka::allocBuf<std::byte, Size>(devAcc, bufferSize);

    chrono.printAndReset("Alloc");

    // create LLAMA views
    auto hostA = llama::View{mapping, llama::Array{alpaka::getPtrNative(hostBufferA)}};
    auto hostB = llama::View{mapping, llama::Array{alpaka::getPtrNative(hostBufferB)}};
    auto devA = llama::View{mapping, llama::Array{alpaka::getPtrNative(devBufferA)}};
    auto devB = llama::View{mapping, llama::Array{alpaka::getPtrNative(devBufferB)}};

    chrono.printAndReset("Views");

    std::default_random_engine generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    auto seed = distribution(generator);
    LLAMA_INDEPENDENT_DATA
    for (std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        hostA(i) = seed + i;
        hostB(i) = seed - i;
    }
    chrono.printAndReset("Init");

    alpaka::memcpy(queue, devBufferA, hostBufferA, bufferSize);
    alpaka::memcpy(queue, devBufferB, hostBufferB, bufferSize);

    chrono.printAndReset("Copy H->D");

    constexpr std::size_t hardwareThreads = 2; // relevant for OpenMP2Threads
    using Distribution = common::ThreadsElemsDistribution<Acc, BLOCK_SIZE, hardwareThreads>;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    const alpaka::Vec<Dim, Size> elems(static_cast<Size>(elemCount));
    const alpaka::Vec<Dim, Size> threads(static_cast<Size>(threadCount));
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::Vec<Dim, Size> blocks(static_cast<Size>((PROBLEM_SIZE + innerCount - 1) / innerCount));

    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{blocks, threads, elems};

    for (std::size_t s = 0; s < STEPS; ++s)
    {
        alpaka::exec<Acc>(queue, workdiv, AddKernel<PROBLEM_SIZE, elemCount>{}, devA, devB);
        chrono.printAndReset("Add kernel");
    }

    alpaka::memcpy(queue, hostBufferA, devBufferA, bufferSize);
    alpaka::memcpy(queue, hostBufferB, devBufferB, bufferSize);

    chrono.printAndReset("Copy D->H");

    return 0;
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

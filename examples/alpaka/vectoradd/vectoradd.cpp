/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

#include "../../common/Chrono.hpp"
#include "../../common/alpakaHelpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <llama/llama.hpp>
#include <random>
#include <utility>

constexpr auto MAPPING
    = 0; /// 0 native AoS, 1 native SoA, 2 tree AoS, 3 tree SoA
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
}

using Vector = llama::DS<
    llama::DE<tag::X, FP>,
    llama::DE<tag::Y, FP>,
    llama::DE<tag::Z, FP>>;
// clang-format on

template<std::size_t ProblemSize, std::size_t Elems>
struct AddKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void
    operator()(const Acc & acc, View a, View b) const
    {
        const auto ti
            = alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        const auto start = ti * Elems;
        const auto end = alpaka::math::min(acc, start + Elems, ProblemSize);

        LLAMA_INDEPENDENT_DATA
        for(auto i = start; i < end; ++i)
        {
            a(i)(tag::X{}) += b(i)(tag::X{});
            a(i)(tag::Y{}) -= b(i)(tag::Y{});
            a(i)(tag::Z{}) *= b(i)(tag::Z{});
        }
    }
};

int main(int argc, char ** argv)
{
    // ALPAKA
    using Dim = alpaka::dim::DimInt<1>;
    using Size = std::size_t;

    using Acc = alpaka::example::ExampleDefaultAcc<Dim, Size>;
    // using Acc = alpaka::acc::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;

    using DevHost = alpaka::dev::DevCpu;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using Queue = alpaka::queue::Queue<DevAcc, alpaka::queue::Blocking>;
    const DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    const DevHost devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));
    Queue queue(devAcc);

    // LLAMA
    using UserDomain = llama::UserDomain<1>;
    const UserDomain userDomain{PROBLEM_SIZE};

    const auto mapping = [&] {
        if constexpr(MAPPING == 0)
        {
            using Mapping = llama::mapping::AoS<UserDomain, Vector>;
            return Mapping(userDomain);
        }
        if constexpr(MAPPING == 1)
        {
            using Mapping = llama::mapping::SoA<UserDomain, Vector>;
            return Mapping(userDomain);
        }
        if constexpr(MAPPING == 2)
        {
            auto treeOperationList = llama::Tuple{};
            using Mapping = llama::mapping::tree::
                Mapping<UserDomain, Vector, decltype(treeOperationList)>;
            return Mapping(userDomain, treeOperationList);
        }
        if constexpr(MAPPING == 3)
        {
            auto treeOperationList
                = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
            using Mapping = llama::mapping::tree::
                Mapping<UserDomain, Vector, decltype(treeOperationList)>;
            return Mapping(userDomain, treeOperationList);
        }
    }();
    using Mapping = decltype(mapping);

    std::cout << PROBLEM_SIZE / 1000 / 1000 << " million vectors\n"
              << PROBLEM_SIZE * llama::sizeOf<Vector> * 2 / 1000 / 1000
              << " MB on device\n";

    Chrono chrono;

    const auto bufferSize = Size(mapping.getBlobSize(0));

    // allocate buffers
    auto hostBufferA
        = alpaka::mem::buf::alloc<std::byte, Size>(devHost, bufferSize);
    auto hostBufferB
        = alpaka::mem::buf::alloc<std::byte, Size>(devHost, bufferSize);
    auto devBufferA
        = alpaka::mem::buf::alloc<std::byte, Size>(devAcc, bufferSize);
    auto devBufferB
        = alpaka::mem::buf::alloc<std::byte, Size>(devAcc, bufferSize);

    chrono.printAndReset("Alloc");

    // create LLAMA views
    auto hostA = llama::View<Mapping, std::byte *>{
        mapping, {alpaka::mem::view::getPtrNative(hostBufferA)}};
    auto hostB = llama::View<Mapping, std::byte *>{
        mapping, {alpaka::mem::view::getPtrNative(hostBufferB)}};
    auto devA = llama::View<Mapping, std::byte *>{
        mapping, {alpaka::mem::view::getPtrNative(devBufferA)}};
    auto devB = llama::View<Mapping, std::byte *>{
        mapping, {alpaka::mem::view::getPtrNative(devBufferB)}};

    chrono.printAndReset("Views");

    std::default_random_engine generator;
    std::normal_distribution<FP> distribution(FP(0), FP(1));
    auto seed = distribution(generator);
    LLAMA_INDEPENDENT_DATA
    for(std::size_t i = 0; i < PROBLEM_SIZE; ++i)
    {
        hostA(i) = seed + i;
        hostB(i) = seed - i;
    }
    chrono.printAndReset("Init");

    alpaka::mem::view::copy(queue, devBufferA, hostBufferA, bufferSize);
    alpaka::mem::view::copy(queue, devBufferB, hostBufferB, bufferSize);

    chrono.printAndReset("Copy H->D");

    constexpr std::size_t hardwareThreads = 2; // relevant for OpenMP2Threads
    using Distribution
        = common::ThreadsElemsDistribution<Acc, BLOCK_SIZE, hardwareThreads>;
    constexpr std::size_t elemCount = Distribution::elemCount;
    constexpr std::size_t threadCount = Distribution::threadCount;
    const alpaka::vec::Vec<Dim, Size> elems(static_cast<Size>(elemCount));
    const alpaka::vec::Vec<Dim, Size> threads(static_cast<Size>(threadCount));
    constexpr auto innerCount = elemCount * threadCount;
    const alpaka::vec::Vec<Dim, Size> blocks(
        static_cast<Size>((PROBLEM_SIZE + innerCount - 1) / innerCount));

    const auto workdiv
        = alpaka::workdiv::WorkDivMembers<Dim, Size>{blocks, threads, elems};

    for(std::size_t s = 0; s < STEPS; ++s)
    {
        alpaka::kernel::exec<Acc>(
            queue, workdiv, AddKernel<PROBLEM_SIZE, elemCount>{}, devA, devB);
        chrono.printAndReset("Add kernel");
    }

    alpaka::mem::view::copy(queue, hostBufferA, devBufferA, bufferSize);
    alpaka::mem::view::copy(queue, hostBufferB, devBufferB, bufferSize);

    chrono.printAndReset("Copy D->H");

    return 0;
}

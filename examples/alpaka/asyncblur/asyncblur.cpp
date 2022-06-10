/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file asynccopy.cpp
 *  \brief Asynchronous bluring example for LLAMA using ALPAKA.
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../../common/Stopwatch.hpp"
#include "../../common/alpakaHelpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <iostream>
#include <list>
#include <llama/llama.hpp>
#include <random>
#include <stb_image.h>
#include <stb_image_write.h>
#include <utility>

constexpr auto ASYNC = true; ///< defines whether the data shall be processed asynchronously
constexpr auto SHARED = true; ///< defines whether shared memory shall be used
constexpr auto SAVE = true; ///< defines whether the resultion image shall be saved
constexpr auto CHUNK_COUNT = 4;

constexpr auto DEFAULT_IMG_X = 4096; /// width of the default image if no png is loaded
constexpr auto DEFAULT_IMG_Y = 4096; /// height of the default image if no png is loaded
constexpr auto KERNEL_SIZE = 8; /// radius of the blur kernel, the diameter is this times two plus one
constexpr auto CHUNK_SIZE = 512; /// size of each chunk to be processed per alpaka kernel
constexpr auto ELEMS_PER_BLOCK = 16; /// number of elements per direction(!) every block should process

using FP = float;

// clang-format off
namespace tag
{
    struct R{};
    struct G{};
    struct B{};
} // namespace tag

/// real record dimension of the image pixel used on the host for loading and saving
using Pixel = llama::Record<
    llama::Field<tag::R, FP>,
    llama::Field<tag::G, FP>,
    llama::Field<tag::B, FP>>;

/// record dimension used in the kernel to modify the image
using PixelOnAcc = llama::Record<
    llama::Field<tag::R, FP>, // you can remove one here if you want to checkout the difference of the result image ;)
    llama::Field<tag::G, FP>,
    llama::Field<tag::B, FP>>;
// clang-format on

/** Alpaka kernel functor used to blur a small image living in the device memory
 *  using the \ref PixelOnAcc record dimension
 */
template<int Elems, int KernelSize, int ElemsPerBlock>
struct BlurKernel
{
    template<typename Acc, typename View>
    LLAMA_FN_HOST_ACC_INLINE void operator()(const Acc& acc, View oldImage, View newImage) const
    {
        const auto ti = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        [[maybe_unused]] auto sharedView = [&]
        {
            if constexpr(SHARED)
            {
                // Using SoA for the shared memory
                constexpr auto sharedChunkSize = ElemsPerBlock + 2 * KernelSize;
                constexpr auto sharedMapping = llama::mapping::
                    SoA<llama::ArrayExtents<int, sharedChunkSize, sharedChunkSize>, typename View::RecordDim, false>{};
                auto& sharedMem = alpaka::declareSharedVar<std::byte[sharedMapping.blobSize(0)], __COUNTER__>(acc);
                return llama::View(sharedMapping, llama::Array<std::byte*, 1>{&sharedMem[0]});
            }
            else
                return int{}; // dummy
        }();

        [[maybe_unused]] const auto bi = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        if constexpr(SHARED)
        {
            constexpr auto threadsPerBlock = ElemsPerBlock / Elems;
            const auto threadIdxInBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);

            const int bStart[2]
                = {bi[0] * ElemsPerBlock + threadIdxInBlock[0], bi[1] * ElemsPerBlock + threadIdxInBlock[1]};
            const int bEnd[2] = {
                alpaka::math::min(acc, bStart[0] + ElemsPerBlock + 2 * KernelSize, oldImage.mapping().extents()[0]),
                alpaka::math::min(acc, bStart[1] + ElemsPerBlock + 2 * KernelSize, oldImage.mapping().extents()[1]),
            };
            LLAMA_INDEPENDENT_DATA
            for(auto y = bStart[0]; y < bEnd[0]; y += threadsPerBlock)
                LLAMA_INDEPENDENT_DATA
            for(auto x = bStart[1]; x < bEnd[1]; x += threadsPerBlock)
                sharedView(y - bi[0] * ElemsPerBlock, x - bi[1] * ElemsPerBlock) = oldImage(y, x);

            alpaka::syncBlockThreads(acc);
        }

        const int start[2] = {ti[0] * Elems, ti[1] * Elems};
        const int end[2] = {
            alpaka::math::min(acc, start[0] + Elems, oldImage.mapping().extents()[0] - 2 * KernelSize),
            alpaka::math::min(acc, start[1] + Elems, oldImage.mapping().extents()[1] - 2 * KernelSize),
        };

        LLAMA_INDEPENDENT_DATA
        for(auto y = start[0]; y < end[0]; ++y)
            LLAMA_INDEPENDENT_DATA
        for(auto x = start[1]; x < end[1]; ++x)
        {
            llama::One<PixelOnAcc> sum{0};

            using ItType = std::int64_t;
            const ItType iBStart = SHARED ? ItType(y) - ItType(bi[0] * ElemsPerBlock) : y;
            const ItType iAStart = SHARED ? ItType(x) - ItType(bi[1] * ElemsPerBlock) : x;
            const ItType i_b_end
                = SHARED ? ItType(y + 2 * KernelSize + 1) - ItType(bi[0] * ElemsPerBlock) : y + 2 * KernelSize + 1;
            const ItType i_a_end
                = SHARED ? ItType(x + 2 * KernelSize + 1) - ItType(bi[1] * ElemsPerBlock) : x + 2 * KernelSize + 1;
            LLAMA_INDEPENDENT_DATA
            for(auto b = iBStart; b < i_b_end; ++b)
                LLAMA_INDEPENDENT_DATA
            for(auto a = iAStart; a < i_a_end; ++a)
            {
                if constexpr(SHARED)
                    sum += sharedView(b, a);
                else
                    sum += oldImage(b, a);
            }
            sum /= FP((2 * KernelSize + 1) * (2 * KernelSize + 1));
            newImage(y + KernelSize, x + KernelSize) = sum;
        }
    }
};

auto main(int argc, char** argv) -> int
try
{
    // ALPAKA
    using Dim = alpaka::DimInt<2>;

    using Acc = alpaka::ExampleDefaultAcc<Dim, int>;
    // using Acc = alpaka::AccGpuCudaRt<Dim, Size>;
    // using Acc = alpaka::AccCpuSerial<Dim, Size>;

    using Queue = alpaka::Queue<Acc, std::conditional_t<ASYNC, alpaka::NonBlocking, alpaka::Blocking>>;
    using DevHost = alpaka::DevCpu;
    using DevAcc = alpaka::Dev<Acc>;
    using PltfHost = alpaka::Pltf<DevHost>;
    using PltfAcc = alpaka::Pltf<DevAcc>;
    const DevAcc devAcc = alpaka::getDevByIdx<PltfAcc>(0);
    const DevHost devHost = alpaka::getDevByIdx<PltfHost>(0);
    std::vector<Queue> queue;
    for(int i = 0; i < CHUNK_COUNT; ++i)
        queue.emplace_back(devAcc);

    // ASYNCCOPY
    int img_x = DEFAULT_IMG_X;
    int img_y = DEFAULT_IMG_Y;
    int buffer_x = DEFAULT_IMG_X + 2 * KERNEL_SIZE;
    int buffer_y = DEFAULT_IMG_Y + 2 * KERNEL_SIZE;

    constexpr int hardwareThreads = 2; // relevant for OpenMP2Threads
    using Distribution = common::ThreadsElemsDistribution<Acc, ELEMS_PER_BLOCK, hardwareThreads>;
    constexpr int elemCount = Distribution::elemCount;
    constexpr int threadCount = Distribution::threadCount;

    std::vector<unsigned char> image;
    std::string out_filename = "output.png";

    if(argc > 1)
    {
        int x = 0;
        int y = 0;
        int n = 3;
        unsigned char* data = stbi_load(argv[1], &x, &y, &n, 0);
        image.resize(x * y * 3);
        std::copy(data, data + image.size(), begin(image));
        stbi_image_free(data);
        img_x = x;
        img_y = y;
        buffer_x = x + 2 * KERNEL_SIZE;
        buffer_y = y + 2 * KERNEL_SIZE;

        if(argc > 2)
            out_filename = std::string(argv[2]);
    }

    // LLAMA
    using ArrayIndex = llama::ArrayIndex<int, 2>;

    auto treeOperationList = llama::Tuple{llama::mapping::tree::functor::LeafOnlyRT()};
    const auto hostMapping
        = llama::mapping::tree::Mapping{llama::ArrayExtents{buffer_y, buffer_x}, treeOperationList, Pixel{}};
    const auto devMapping = llama::mapping::tree::Mapping{
        llama::ArrayExtents<int, CHUNK_SIZE + 2 * KERNEL_SIZE, CHUNK_SIZE + 2 * KERNEL_SIZE>{},
        treeOperationList,
        PixelOnAcc{}};
    using DevMapping = std::decay_t<decltype(devMapping)>;

    std::size_t hostBufferSize = 0;
    for(std::size_t i = 0; i < hostMapping.blobCount; i++)
        hostBufferSize += hostMapping.blobSize(i);
    std::cout << "Image size: " << img_x << ":" << img_y << '\n'
              << hostBufferSize * 2 / 1024 / 1024 << " MB on device\n";

    Stopwatch chrono;

    auto allocBlobHost = llama::bloballoc::AlpakaBuf<int, decltype(devHost)>{devHost};
    auto allocBlobAcc = llama::bloballoc::AlpakaBuf<int, decltype(devAcc)>{devAcc};
    auto hostView = llama::allocView(hostMapping, allocBlobHost);

    using HostChunkView = decltype(llama::allocView(devMapping, allocBlobHost));
    using AccChunkView = decltype(llama::allocView(devMapping, allocBlobAcc));
    std::vector<HostChunkView> hostChunkView;
    std::vector<AccChunkView> devOldView;
    std::vector<AccChunkView> devNewView;

    for(int i = 0; i < CHUNK_COUNT; ++i)
    {
        hostChunkView.push_back(llama::allocView(devMapping, allocBlobHost));
        devOldView.push_back(llama::allocView(devMapping, allocBlobAcc));
        devNewView.push_back(llama::allocView(devMapping, allocBlobAcc));
    }

    chrono.printAndReset("Alloc");

    if(image.empty())
    {
        image.resize(img_x * img_y * 3);
        std::default_random_engine generator;
        std::normal_distribution<FP> distribution{FP(0), FP(0.5)};
        for(int y = 0; y < buffer_y; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for(int x = 0; x < buffer_x; ++x)
            {
                hostView(y, x)(tag::R()) = std::abs(distribution(generator));
                hostView(y, x)(tag::G()) = std::abs(distribution(generator));
                hostView(y, x)(tag::B()) = std::abs(distribution(generator));
            }
        }
    }
    else
    {
        for(int y = 0; y < buffer_y; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for(int x = 0; x < buffer_x; ++x)
            {
                const auto X = std::clamp<int>(x, KERNEL_SIZE, img_x + KERNEL_SIZE - 1);
                const auto Y = std::clamp<int>(y, KERNEL_SIZE, img_y + KERNEL_SIZE - 1);
                const auto* pixel = &image[((Y - KERNEL_SIZE) * img_x + X - KERNEL_SIZE) * 3];
                hostView(y, x)(tag::R()) = FP(pixel[0]) / 255;
                hostView(y, x)(tag::G()) = FP(pixel[1]) / 255;
                hostView(y, x)(tag::B()) = FP(pixel[2]) / 255;
            }
        }
    }

    chrono.printAndReset("Init");
    const auto elems = alpaka::Vec<Dim, int>(elemCount, elemCount);
    const auto threads = alpaka::Vec<Dim, int>(threadCount, threadCount);
    const auto blocks = alpaka::Vec<Dim, int>(
        static_cast<int>((CHUNK_SIZE + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK),
        static_cast<int>((CHUNK_SIZE + ELEMS_PER_BLOCK - 1) / ELEMS_PER_BLOCK));
    const alpaka::Vec<Dim, int> chunks(
        static_cast<int>((img_y + CHUNK_SIZE - 1) / CHUNK_SIZE),
        static_cast<int>((img_x + CHUNK_SIZE - 1) / CHUNK_SIZE));

    const auto workdiv = alpaka::WorkDivMembers<Dim, int>{blocks, threads, elems};

    struct VirtualHostElement
    {
        llama::VirtualView<decltype(hostView)&> virtualHost;
        const llama::ArrayExtentsDynamic<int, 2> validMiniSize;
    };
    std::list<VirtualHostElement> virtualHostList;
    for(int chunk_y = 0; chunk_y < chunks[0]; ++chunk_y)
        for(int chunk_x = 0; chunk_x < chunks[1]; ++chunk_x)
        {
            // Create virtual view with size of mini view
            const auto validMiniSize = llama::ArrayExtents{
                ((chunk_y < chunks[0] - 1) ? CHUNK_SIZE : (img_y - 1) % CHUNK_SIZE + 1) + 2 * KERNEL_SIZE,
                ((chunk_x < chunks[1] - 1) ? CHUNK_SIZE : (img_x - 1) % CHUNK_SIZE + 1) + 2 * KERNEL_SIZE};
            llama::VirtualView virtualHost(hostView, {chunk_y * CHUNK_SIZE, chunk_x * CHUNK_SIZE});

            // Find free chunk stream
            int chunkNr = virtualHostList.size();
            if(virtualHostList.size() < CHUNK_COUNT)
                virtualHostList.push_back({virtualHost, validMiniSize});
            else
            {
                bool notFound = true;
                while(notFound)
                {
                    auto chunkIt = virtualHostList.begin();
                    for(chunkNr = 0; chunkNr < CHUNK_COUNT; ++chunkNr)
                    {
                        if(alpaka::empty(queue[chunkNr]))
                        {
                            // Copy data back

                            LLAMA_INDEPENDENT_DATA
                            for(int y = 0; y < chunkIt->validMiniSize[0] - 2 * KERNEL_SIZE; ++y)
                            {
                                LLAMA_INDEPENDENT_DATA
                                for(int x = 0; x < chunkIt->validMiniSize[1] - 2 * KERNEL_SIZE; ++x)
                                    chunkIt->virtualHost(y + KERNEL_SIZE, x + KERNEL_SIZE)
                                        = hostChunkView[chunkNr](y + KERNEL_SIZE, x + KERNEL_SIZE);
                            }
                            chunkIt = virtualHostList.erase(chunkIt);
                            virtualHostList.insert(chunkIt, {virtualHost, validMiniSize});
                            notFound = false;
                            break;
                        }
                        chunkIt++;
                    }
                    if(notFound)
                        std::this_thread::sleep_for(std::chrono::microseconds{1});
                }
            }

            // Copy data from virtual view to mini view
            for(int y = 0; y < validMiniSize[0]; ++y)
            {
                LLAMA_INDEPENDENT_DATA
                for(int x = 0; x < validMiniSize[1]; ++x)
                    hostChunkView[chunkNr](y, x) = virtualHost(y, x);
            }
            for(std::size_t i = 0; i < devMapping.blobCount; i++)
                alpaka::memcpy(
                    queue[chunkNr],
                    devOldView[chunkNr].storageBlobs[i],
                    hostChunkView[chunkNr].storageBlobs[i]);

            alpaka::exec<Acc>(
                queue[chunkNr],
                workdiv,
                BlurKernel<elemCount, KERNEL_SIZE, ELEMS_PER_BLOCK>{},
                llama::shallowCopy(devOldView[chunkNr]),
                llama::shallowCopy(devNewView[chunkNr]));

            for(std::size_t i = 0; i < devMapping.blobCount; i++)
                alpaka::memcpy(
                    queue[chunkNr],
                    hostChunkView[chunkNr].storageBlobs[i],
                    devNewView[chunkNr].storageBlobs[i]);
        }

    // Wait for not finished tasks on accelerator
    auto chunkIt = virtualHostList.begin();
    for(int chunkNr = 0; chunkNr < CHUNK_COUNT; ++chunkNr)
    {
        alpaka::wait(queue[chunkNr]);
        // Copy data back
        for(int y = 0; y < chunkIt->validMiniSize[0] - 2 * KERNEL_SIZE; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for(int x = 0; x < chunkIt->validMiniSize[1] - 2 * KERNEL_SIZE; ++x)
                chunkIt->virtualHost(y + KERNEL_SIZE, x + KERNEL_SIZE)
                    = hostChunkView[chunkNr](y + KERNEL_SIZE, x + KERNEL_SIZE);
        }
        chunkIt++;
    }
    chrono.printAndReset("Blur kernel");

    if(SAVE)
    {
        for(int y = 0; y < img_y; ++y)
        {
            LLAMA_INDEPENDENT_DATA
            for(int x = 0; x < img_x; ++x)
            {
                auto* pixel = &image[(y * img_x + x) * 3];
                pixel[0] = static_cast<unsigned char>(hostView(y + KERNEL_SIZE, x + KERNEL_SIZE)(tag::R()) * 255.);
                pixel[1] = static_cast<unsigned char>(hostView(y + KERNEL_SIZE, x + KERNEL_SIZE)(tag::G()) * 255.);
                pixel[2] = static_cast<unsigned char>(hostView(y + KERNEL_SIZE, x + KERNEL_SIZE)(tag::B()) * 255.);
            }
        }
        stbi_write_png(out_filename.c_str(), static_cast<int>(img_x), static_cast<int>(img_y), 3, image.data(), 0);
    }

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

#include "../../common/Stopwatch.hpp"
#include "../../common/hostname.hpp"
#include "../../common/ttjet_13tev_june2019.hpp"

#include <boost/functional/hash.hpp>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <fstream>
#include <llama/llama.hpp>
#include <numeric>

// This example only shows variations of naive, layout oblivious copy kernels between different data layouts.
// No specialized layout aware copy routine is implemented.

using Size = std::size_t;
constexpr auto repetitions = 5;
constexpr auto extents = llama::ArrayExtentsDynamic<Size, 3>{200u, 512u, 512u}; // z, y, x
constexpr auto threadsPerBlock = Size{256};

static_assert(extents.back() % threadsPerBlock == 0);

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
         llama::Field<tag::X, float>,
         llama::Field<tag::Y, float>,
         llama::Field<tag::Z, float>
     >>,
     llama::Field<tag::Vel, llama::Record<
         llama::Field<tag::X, float>,
         llama::Field<tag::Y, float>,
         llama::Field<tag::Z, float>
     >>,
     llama::Field<tag::Mass, float>
>;
// clang-format on

using RecordDim = Particle;
// using RecordDim = boost::mp11::mp_take_c<Event, 20>;
// using RecordDim = Event; // WARN: expect long compilation time

static_assert(
    llama::product(extents) * llama::sizeOf<RecordDim> < std::numeric_limits<decltype(extents)::value_type>::max(),
    "Array extents indexing type probably too small to hold all addresses");

template<typename Mapping, typename BlobType>
auto hash(const llama::View<Mapping, BlobType>& devView) -> std::size_t
{
    auto hostView = llama::allocViewUninitialized(devView.mapping());
    for(int i = 0; i < Mapping::blobCount; i++)
        cudaMemcpyAsync(
            &hostView.blobs()[i][0],
            &devView.blobs()[i][0],
            devView.mapping().blobSize(i),
            cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::size_t acc = 0;
    for(auto ai : llama::ArrayIndexRange{hostView.extents()})
        llama::forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc)
                                                             { boost::hash_combine(acc, hostView(ai)(rc)); });
    return acc;
}

auto toDim3(Size x, Size y, Size z)
{
    return dim3{static_cast<unsigned>(x), static_cast<unsigned>(y), static_cast<unsigned>(z)};
}

template<typename SrcView, typename DstView>
__global__ void fieldWiseCopy1DKernel(SrcView srcView, DstView dstView, unsigned arrayElements)
{
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= arrayElements)
        return;
    const auto ai = *(llama::ArrayIndexIterator{srcView.extents(), {}} + i);
    llama::forEachLeafCoord<typename SrcView::RecordDim>([&](auto rc) { dstView(ai)(rc) = srcView(ai)(rc); });
}

template<typename SrcView, typename DstView>
void fieldWiseCopy1D(const SrcView& srcView, DstView& dstView)
{
    static_assert(std::is_same_v<typename SrcView::ArrayExtents, typename DstView::ArrayExtents>);
    const auto arrayElements = llama::product(srcView.extents());
    const auto blocks = llama::divCeil(arrayElements, threadsPerBlock);
    fieldWiseCopy1DKernel<<<blocks, threadsPerBlock>>>(
        llama::shallowCopy(srcView),
        llama::shallowCopy(dstView),
        arrayElements);
}

template<typename SrcView, typename DstView>
__global__ void fieldWiseCopy3DKernel(SrcView srcView, DstView dstView)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    const auto z = blockIdx.z * blockDim.z + threadIdx.z;
    const auto [ez, ey, ex] = srcView.extents();
    if(x >= ex || y >= ey || z >= ez)
        return;

    const auto ai = typename SrcView::ArrayIndex{z, y, x};
    llama::forEachLeafCoord<typename SrcView::RecordDim>([&](auto rc) { dstView(ai)(rc) = srcView(ai)(rc); });
}

template<typename SrcView, typename DstView>
void fieldWiseCopy3D(const SrcView& srcView, DstView& dstView)
{
    static_assert(std::is_same_v<typename SrcView::ArrayExtents, typename DstView::ArrayExtents>);

    const auto [ez, ey, ex] = srcView.extents();
    fieldWiseCopy3DKernel<<<toDim3(ex / threadsPerBlock, ey, ez), toDim3(threadsPerBlock, 1, 1)>>>(
        llama::shallowCopy(srcView),
        llama::shallowCopy(dstView));
}

template<typename SrcView, typename DstView>
__global__ void fieldWiseCopyGridStrided1DKernel(SrcView srcView, DstView dstView, unsigned arrayElements)
{
    const auto gridSize = gridDim.x * blockDim.x;
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto it = llama::ArrayIndexIterator{srcView.extents(), {}};
    it += i;
    while(i < arrayElements)
    {
        llama::forEachLeafCoord<typename SrcView::RecordDim>([&](auto rc) { dstView (*it)(rc) = srcView(*it)(rc); });
        i += gridSize;
        it += gridSize;
    }
}

template<typename SrcView, typename DstView>
void fieldWiseCopyGridStrided1D(const SrcView& srcView, DstView& dstView, Size maxThreads)
{
    static_assert(std::is_same_v<typename SrcView::ArrayExtents, typename DstView::ArrayExtents>);

    const auto [ez, ey, ex] = srcView.extents();
    const auto arrayElements = llama::product(srcView.extents());
    const auto blocks
        = llama::divCeil(std::min(maxThreads, arrayElements), threadsPerBlock) * 2; // oversubscribe twice
    fieldWiseCopyGridStrided1DKernel<<<blocks, threadsPerBlock / 2>>>(
        llama::shallowCopy(srcView),
        llama::shallowCopy(dstView),
        arrayElements);
}

template<typename SrcView, typename DstView>
__global__ void fieldWiseCopyGridStrided3DKernel(SrcView srcView, DstView dstView)
{
    const auto gridSizeX = gridDim.x * blockDim.x;
    const auto gridSizeY = gridDim.y * blockDim.y;
    const auto gridSizeZ = gridDim.z * blockDim.z;
    const auto [ez, ey, ex] = srcView.extents();

    for(auto z = blockIdx.z * blockDim.z + threadIdx.z; z < ez; z += gridSizeZ)
        for(auto y = blockIdx.y * blockDim.y + threadIdx.y; y < ey; y += gridSizeY)
            for(auto x = blockIdx.x * blockDim.x + threadIdx.x; x < ex; x += gridSizeX)
            {
                const auto ai = typename SrcView::ArrayIndex{z, y, x};
                llama::forEachLeafCoord<typename SrcView::RecordDim>([&](auto rc)
                                                                     { dstView(ai)(rc) = srcView(ai)(rc); });
            }
}

template<typename SrcView, typename DstView>
void fieldWiseCopyGridStrided3D(const SrcView& srcView, DstView& dstView, Size maxThreads)
{
    static_assert(std::is_same_v<typename SrcView::ArrayExtents, typename DstView::ArrayExtents>);
    const auto arrayElements = llama::product(srcView.extents());
    const auto blocks
        = llama::divCeil(std::min(maxThreads, arrayElements), threadsPerBlock) * 2; // oversubscribe twice
    fieldWiseCopyGridStrided3DKernel<<<toDim3(blocks, 1, 1), toDim3(8, 8, 8)>>>(
        llama::shallowCopy(srcView),
        llama::shallowCopy(dstView));
}

template<typename View>
__global__ void initViewKernel(View view)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    const auto z = blockIdx.z * blockDim.z + threadIdx.z;
    const auto [ez, ey, ex] = view.extents();
    if(x >= ex || y >= ey || z >= ez)
        return;

    const auto ai = typename View::ArrayIndex{z, y, x};
    llama::forEachLeafCoord<typename View::RecordDim>([&, ex = ex, ey = ey](auto rc)
                                                      { view(ai)(rc) = z * ex * ey + y * ex + x; });
}

template<typename Mapping>
auto prepareViewAndHash(Mapping mapping)
{
    auto view = llama::allocViewUninitialized(mapping, llama::bloballoc::CudaMalloc{});
    const auto [ez, ey, ex] = view.extents();
    initViewKernel<<<toDim3(ex / threadsPerBlock, ey, ez), toDim3(threadsPerBlock, 1, 1)>>>(llama::shallowCopy(view));

    const auto checkSum = hash(view);
    return std::tuple{std::move(view), checkSum};
}

auto main() -> int
try
{
    const auto dataSize = llama::product(extents) * llama::sizeOf<RecordDim, false, false>; // no padding

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const auto [dsize, dunit] = llama::prettySize(dataSize);
    const auto [gmsize, gmunit] = llama::prettySize(prop.totalGlobalMem);
    // for bandwidth comp: https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
    const auto gibs = 2.0 * prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8) / (1024.0 * 1024.0 * 1024.0);
    fmt::print(
        "Dataset size:  {:5.1f}{} (x2)\nGMemory size:  {:5.1f}{}\nMax bandwidth: {:5.1f}GiB/s\nSMs: {}\nMax threads "
        "per SM: {}\n",
        dsize,
        dunit,
        gmsize,
        gmunit,
        gibs,
        prop.multiProcessorCount,
        prop.maxThreadsPerMultiProcessor);
    const Size maxThreads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;

    fmt::print("{:10} -> {:10} {:11} {:>10} {:>10} {:4}\n", "src", "dst", "alg", "ms", "GiB/s", "hash");

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // benchmark cudaMemcpy
    {
        std::byte* src = nullptr;
        std::byte* dst = nullptr;
        cudaMalloc(&src, dataSize);
        cudaMalloc(&dst, dataSize);
        cudaEventRecord(startEvent);
        for(auto i = 0; i < repetitions; i++)
            cudaMemcpyAsync(dst, src, dataSize, cudaMemcpyDeviceToDevice);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float ms = 0;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        ms /= repetitions;
        fmt::print(
            "{:10} -> {:10} {:11} {:10.3f} {:10.3f} {:>4}\n",
            "byte[]",
            "byte[]",
            "cudaMemcpy",
            ms,
            (2.0 * dataSize / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0),
            "OK");
        cudaFree(dst);
    }

    auto benchmarkAllCopies = [&](std::string_view srcName, std::string_view dstName, auto srcMapping, auto dstMapping)
    {
        const auto [srcView, srcHash] = prepareViewAndHash(srcMapping);

        auto benchmarkCopy = [&, srcView = &srcView, srcHash = srcHash](std::string_view algName, auto copy)
        {
            auto dstView = llama::allocViewUninitialized(dstMapping, llama::bloballoc::CudaMalloc{});
            cudaEventRecord(startEvent);
            for(auto i = 0; i < repetitions; i++)
                copy(*srcView, dstView);
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);
            float ms = 0;
            cudaEventElapsedTime(&ms, startEvent, stopEvent);
            ms /= repetitions;
            const auto dstHash = hash(dstView);
            fmt::print(
                "{:10} -> {:10} {:11} {:10.3f} {:10.3f} {:>4}\n",
                srcName,
                dstName,
                algName,
                ms,
                (2.0 * dataSize / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0),
                srcHash == dstHash ? "OK" : "BAD");
        };

        benchmarkCopy("naive 1D", [](const auto& srcView, auto& dstView) { ::fieldWiseCopy1D(srcView, dstView); });
        benchmarkCopy("naive 3D", [](const auto& srcView, auto& dstView) { ::fieldWiseCopy3D(srcView, dstView); });
        benchmarkCopy(
            "naive GS 1D",
            [&](const auto& srcView, auto& dstView) { ::fieldWiseCopyGridStrided1D(srcView, dstView, maxThreads); });
        benchmarkCopy(
            "naive GS 3D",
            [&](const auto& srcView, auto& dstView) { ::fieldWiseCopyGridStrided3D(srcView, dstView, maxThreads); });
    };

    using ArrayExtents = std::remove_const_t<decltype(extents)>;
    const auto alignedAoSMapping = llama::mapping::AlignedAoS<ArrayExtents, RecordDim>{extents};
    const auto multiBlobSoAMapping = llama::mapping::MultiBlobSoA<ArrayExtents, RecordDim>{extents};
    const auto aosoa8Mapping = llama::mapping::AoSoA<ArrayExtents, RecordDim, 8>{extents};
    const auto aosoa32Mapping = llama::mapping::AoSoA<ArrayExtents, RecordDim, 32>{extents};
    const auto aosoa64Mapping = llama::mapping::AoSoA<ArrayExtents, RecordDim, 64>{extents};
    //
    benchmarkAllCopies("A AoS", "SoA MB", alignedAoSMapping, multiBlobSoAMapping);
    benchmarkAllCopies("SoA MB", "A AoS", multiBlobSoAMapping, alignedAoSMapping);

    benchmarkAllCopies("SoA MB", "AoSoA32", multiBlobSoAMapping, aosoa32Mapping);
    benchmarkAllCopies("AoSoA32", "SoA MB", aosoa32Mapping, multiBlobSoAMapping);

    benchmarkAllCopies("AoSoA8", "AoSoA32", aosoa8Mapping, aosoa32Mapping);
    benchmarkAllCopies("AoSoA32", "AoSoA8", aosoa32Mapping, aosoa8Mapping);

    benchmarkAllCopies("AoSoA8", "AoSoA64", aosoa8Mapping, aosoa64Mapping);
    benchmarkAllCopies("AoSoA64", "AoSoA8", aosoa64Mapping, aosoa8Mapping);
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

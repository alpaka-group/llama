#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <cstddef>
#include <cuda_runtime.h>
#include <fmt/format.h>
#include <llama/llama.hpp>
#include <stb_image_write.h>
#include <stdexcept>

using namespace std::literals;

struct RGB
{
    unsigned char r, g, b;

    friend auto operator==(RGB a, RGB b) -> bool
    {
        return a.r == b.r && a.g == b.g && a.b == b.b;
    }
};

void checkError(cudaError_t code)
{
    if(code != cudaSuccess)
        throw std::runtime_error("CUDA Error: "s + cudaGetErrorString(code));
}

template<typename View, typename ArrayExtents>
__global__ void init(View view, ArrayExtents extents)
{
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y >= extents[0] || x >= extents[1])
        return;

    view(y, x).r = static_cast<float>(x) * 255 / static_cast<float>(blockDim.x * gridDim.x);
    view(y, x).g = static_cast<float>(y) * 255 / static_cast<float>(blockDim.y * gridDim.y);
    view(y, x).b = static_cast<float>(threadIdx.x + threadIdx.y) * 255 / static_cast<float>(blockDim.x + blockDim.y);
}

namespace llamaex
{
    using namespace llama;

    template<typename RecordDim, typename ArrayExtents>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto pitchesFromExtents(ArrayExtents extents)
    {
        constexpr std::size_t dim = ArrayExtents{}.size();
        Array<std::size_t, dim> pitches{};
        pitches[dim - 1] = sizeOf<RecordDim>;
        for(auto i = dim - 1; i > 0; --i)
            pitches[i - 1] = pitches[i] * extents[i - 1];
        return pitches;
    }

    template<
        typename TArrayExtents,
        typename TRecordDim,
        bool AlignAndPad = true,
        template<typename> typename FlattenRecordDim = mapping::FlattenRecordDimInOrder>
    struct PitchedAoS : mapping::MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        static constexpr std::size_t dim = TArrayExtents{}.size();

        using Base = mapping::MappingBase<TArrayExtents, TRecordDim>;
        using Flattener = FlattenRecordDim<TRecordDim>;

        Array<std::size_t, dim> pitches;

    public:
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE constexpr PitchedAoS(TArrayExtents extents, Array<std::size_t, dim> pitches)
            : Base(extents)
            , pitches(pitches)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr PitchedAoS(TArrayExtents extents, std::size_t rowPitch)
            : Base(extents)
            , pitches(pitchesFromExtents<TRecordDim>(extents))
        {
            static_assert(dim >= 2, "The rowPitch constructor is only available for 2D or higher dimensions");
            pitches[dim - 2] = rowPitch;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr PitchedAoS(
            TArrayExtents extents,
            std::size_t rowPitch,
            std::size_t slicePitch)
            : Base(extents)
            , pitches(pitchesFromExtents<TRecordDim>(extents))
        {
            static_assert(
                dim >= 3,
                "The rowPitch/slicePitch constructor is only available for 3D or higher dimensions");
            pitches[dim - 2] = rowPitch;
            pitches[dim - 3] = slicePitch;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return pitches[0] * Base::extents()[0];
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            typename Base::ArrayIndex ai,
            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<std::size_t>
        {
            constexpr std::size_t flatFieldIndex =
#ifdef __NVCC__
                *& // mess with nvcc compiler state to workaround bug
#endif
                 Flattener::template flatIndex<RecordCoords...>;
            const auto offset
                = dot(pitches, ai) + flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, AlignAndPad>;
            return {0, offset};
        }
    };
} // namespace llamaex

auto main() -> int
try
{
    int device = 0;
    checkError(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    checkError(cudaGetDeviceProperties(&prop, device));
    fmt::print(
        "Running on {}, {}MiB GM, {}kiB SM\n",
        prop.name,
        prop.totalGlobalMem / 1024 / 1024,
        prop.sharedMemPerBlock / 1024);

    const auto extents = llama::ArrayExtents<std::size_t, llama::dyn, llama::dyn>{600, 800}; // height, width
    const auto widthBytes = extents[1] * sizeof(RGB);

    const auto blockDim = dim3{16, 32, 1};
    const auto gridDim = dim3{
        llama::divCeil(static_cast<unsigned>(extents[1]), blockDim.x),
        llama::divCeil(static_cast<unsigned>(extents[0]), blockDim.y),
        1};

    auto host1 = std::vector<RGB>(llama::product(extents));
    auto host2 = std::vector<RGB>(llama::product(extents));
    {
        std::byte* mem = nullptr;
        std::size_t rowPitch = 0;
        checkError(cudaMallocPitch(&mem, &rowPitch, widthBytes, extents[0]));
        fmt::print("Row pitch: {} B ({} B padding)\n", rowPitch, rowPitch - widthBytes);

        auto mapping = llamaex::PitchedAoS<llama::ArrayExtentsDynamic<std::size_t, 2>, RGB>{extents, rowPitch};
        assert(mapping.blobSize(0) == rowPitch * extents[0]);
        auto view = llama::View{mapping, llama::Array{mem}};

        init<<<gridDim, blockDim>>>(view, extents);

        checkError(cudaMemcpy2D(host1.data(), widthBytes, mem, rowPitch, widthBytes, extents[0], cudaMemcpyDefault));
        checkError(cudaFree(mem));

        stbi_write_png("pitch1.png", extents[1], extents[0], 3, host1.data(), 0);
    }

    {
        std::byte* mem = nullptr;
        checkError(cudaMalloc(&mem, widthBytes * extents[0]));

        auto mapping = llama::mapping::AoS{extents, RGB{}};
        auto view = llama::View{mapping, llama::Array{mem}};

        init<<<gridDim, blockDim>>>(view, extents);

        checkError(cudaMemcpy(host2.data(), mem, widthBytes * extents[0], cudaMemcpyDefault));
        checkError(cudaFree(mem));

        stbi_write_png("pitch2.png", extents[1], extents[0], 3, host2.data(), 0);
    }

    if(host1 != host2)
        fmt::print("ERROR: produced two different images");

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

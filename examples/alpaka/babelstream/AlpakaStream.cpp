// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code
//
// Cupla version created by Jeff Young in 2021
// Ported from cupla to alpaka by Bernhard Manfred Gruber in 2022

#include "AlpakaStream.h"

#include <numeric>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#    include <cuda_runtime.h>
#endif

namespace
{
    constexpr auto blockSize = 1024;
    constexpr auto dotBlockSize = 256;

    struct StreamingAccessor
    {
        template<typename T>
        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
        struct Reference : llama::ProxyRefOpMixin<Reference<T>, T>
        {
            T& ref;

            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
            LLAMA_ACC LLAMA_FORCE_INLINE auto operator=(const Reference& r) -> Reference&
            {
                *this = static_cast<T>(r);
                return *this;
            }

            LLAMA_ACC LLAMA_FORCE_INLINE auto operator=(T t) -> Reference&
            {
                __stcs(&ref, t);
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_ACC LLAMA_FORCE_INLINE operator T() const
            {
                return __ldcs(&ref);
            }
        };

        template<typename T>
        LLAMA_ACC LLAMA_FORCE_INLINE auto operator()(T& ref) const -> Reference<T>
        {
            return Reference<T>{{}, ref};
        }
    };
    // using Accessor = StreamingAccessor;
    using Accessor = llama::accessor::Default;
} // namespace

template<typename T>
AlpakaStream<T>::AlpakaStream(Idx arraySize, Idx deviceIndex)
    : mapping({arraySize})
    , arraySize(arraySize)
    , devHost(alpaka::getDevByIdx(platformHost, 0))
    , devAcc(alpaka::getDevByIdx(platformAcc, deviceIndex))
    , sums(alpaka::allocBuf<T, Idx>(devHost, dotBlockSize))
    , d_a(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_b(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_c(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_sum(alpaka::allocBuf<T, Idx>(devAcc, dotBlockSize))
    , queue(devAcc)
{
    if(arraySize % blockSize != 0)
        throw std::runtime_error("Array size must be a multiple of " + std::to_string(blockSize));
    std::cout << "Using alpaka device " << alpaka::getName(devAcc) << std::endl;
}

struct InitKernel
{
    template<typename TAcc, typename T>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, T* a, T* b, T* c, T initA, T initB, T initC) const
    {
        const auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] = initA;
        b[i] = initB;
        c[i] = initC;
    }
};

template<typename T>
void AlpakaStream<T>::init_arrays(T initA, T initB, T initC)
{
    const auto workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);
    alpaka::exec<Acc>(
        queue,
        workdiv,
        InitKernel{},
        alpaka::getPtrNative(d_a),
        alpaka::getPtrNative(d_b),
        alpaka::getPtrNative(d_c),
        initA,
        initB,
        initC);
    alpaka::wait(queue);
}

template<typename T>
void AlpakaStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
    // TODO(bgruber): avoid temporary alpaka views when we upgrade to alpaka 1.0.0
    auto va = alpaka::createView(devHost, a, arraySize);
    alpaka::memcpy(queue, va, d_a);
    auto vb = alpaka::createView(devHost, b, arraySize);
    alpaka::memcpy(queue, vb, d_b);
    auto vc = alpaka::createView(devHost, c, arraySize);
    alpaka::memcpy(queue, vc, d_c);
}

struct CopyKernel
{
    template<typename TAcc, typename ViewA, typename ViewB>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, ViewA a, ViewB c) const
    {
        const auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        c[i] = a[i];
    }
};

template<typename T>
void AlpakaStream<T>::copy()
{
    const auto workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_a))}, Accessor{}};
    auto viewC
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_c))}, Accessor{}};

    alpaka::exec<Acc>(queue, workdiv, CopyKernel{}, viewA, viewC);
    alpaka::wait(queue);
}

struct MulKernel
{
    template<typename TAcc, typename ViewB, typename ViewC>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, ViewB b, ViewC c) const
    {
        const typename ViewB::RecordDim scalar = startScalar;
        const auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        b[i] = scalar * c[i];
    }
};

template<typename T>
void AlpakaStream<T>::mul()
{
    const auto workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewB
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_b))}, Accessor{}};
    auto viewC
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_c))}, Accessor{}};

    alpaka::exec<Acc>(queue, workdiv, MulKernel{}, viewB, viewC);
    alpaka::wait(queue);
}

struct AddKernel
{
    template<typename TAcc, typename ViewA, typename ViewB, typename ViewC>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, ViewA a, ViewB b, ViewC c) const
    {
        const auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        c[i] = a[i] + b[i];
    }
};

template<typename T>
void AlpakaStream<T>::add()
{
    const auto workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_a))}, Accessor{}};
    auto viewB
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_b))}, Accessor{}};
    auto viewC
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_c))}, Accessor{}};

    alpaka::exec<Acc>(queue, workdiv, AddKernel{}, viewA, viewB, viewC);
    alpaka::wait(queue);
}

struct TriadKernel
{
    template<typename TAcc, typename ViewA, typename ViewB, typename ViewC>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, ViewA a, ViewB b, ViewC c) const
    {
        const typename ViewB::RecordDim scalar = startScalar;
        const auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] = b[i] + scalar * c[i];
    }
};

template<typename T>
void AlpakaStream<T>::triad()
{
    const auto workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_a))}, Accessor{}};
    auto viewB
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_b))}, Accessor{}};
    auto viewC
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_c))}, Accessor{}};

    alpaka::exec<Acc>(queue, workdiv, TriadKernel{}, viewA, viewB, viewC);
    alpaka::wait(queue);
}

struct NstreamKernel
{
    template<typename TAcc, typename ViewA, typename ViewB, typename ViewC>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, ViewA a, ViewB b, ViewC c) const
    {
        const typename ViewB::RecordDim scalar = startScalar;
        const auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        a[i] += b[i] + scalar * c[i];
    }
};

template<typename T>
void AlpakaStream<T>::nstream()
{
    const auto workdiv = WorkDiv{arraySize / blockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_a))}, Accessor{}};
    auto viewB
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_b))}, Accessor{}};
    auto viewC
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_c))}, Accessor{}};

    alpaka::exec<Acc>(queue, workdiv, NstreamKernel{}, viewA, viewB, viewC);
    alpaka::wait(queue);
}

struct DotKernel
{
    template<typename TAcc, typename ViewA, typename ViewB, typename ViewSum>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, ViewA a, ViewB b, ViewSum sum, int arraySize) const
    {
        using T = typename ViewA::RecordDim;

        // TODO(Jeff Young) - test if sharedMem bug is affecting performance here
        auto& tbSum = alpaka::declareSharedVar<T[blockSize], __COUNTER__>(acc);

        auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        const auto [local_i] = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        const auto [totalThreads] = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        tbSum[local_i] = 0.0;
        for(; i < arraySize; i += totalThreads) // NOLINT(bugprone-infinite-loop)
            tbSum[local_i] += a[i] * b[i];

        const auto [blockDim] = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        for(int offset = blockDim / 2; offset > 0; offset /= 2)
        {
            alpaka::syncBlockThreads(acc);
            if(local_i < offset)
                tbSum[local_i] += tbSum[local_i + offset];
        }

        const auto [blockIdx] = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        if(local_i == 0)
            sum[blockIdx] = tbSum[local_i];
    }
};

template<typename T>
auto AlpakaStream<T>::dot() -> T
{
    const auto workdiv = WorkDiv{dotBlockSize, blockSize, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, dotBlockSize * blockSize);

    auto viewA
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_a))}, Accessor{}};
    auto viewB
        = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_b))}, Accessor{}};
    auto viewSum = llama::View{mapping, llama::Array{reinterpret_cast<std::byte*>(alpaka::getPtrNative(d_sum))}};

    alpaka::exec<Acc>(queue, workdiv, DotKernel{}, viewA, viewB, viewSum, arraySize);
    alpaka::wait(queue);

    alpaka::memcpy(queue, sums, d_sum);
    const T* sumPtr = alpaka::getPtrNative(sums);
    return std::reduce(sumPtr, sumPtr + dotBlockSize);
}

void listDevices()
{
    const auto platform = alpaka::Platform<Acc>{};
    const auto count = alpaka::getDevCount(platform);
    std::cout << "Devices:" << std::endl;
    for(int i = 0; i < count; i++)
        std::cout << i << ": " << getDeviceName(i) << std::endl;
}

auto getDeviceName(int deviceIndex) -> std::string
{
    const auto platform = alpaka::Platform<Acc>{};
    return alpaka::getName(alpaka::getDevByIdx(platform, deviceIndex));
}

auto getDeviceDriver([[maybe_unused]] int device) -> std::string
{
    return "Not supported";
}

template class AlpakaStream<float>;
template class AlpakaStream<double>;

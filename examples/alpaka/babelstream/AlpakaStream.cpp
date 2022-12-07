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

constexpr auto TBSIZE = 1024;
constexpr auto DOT_NUM_BLOCKS = 256;

template<typename T>
AlpakaStream<T>::AlpakaStream(Idx arraySize, Idx deviceIndex)
    : mapping({arraySize})
    , arraySize(arraySize)
    , devHost(alpaka::getDevByIdx<DevHost>(0u))
    , devAcc(alpaka::getDevByIdx<Acc>(deviceIndex))
    , sums(alpaka::allocBuf<T, Idx>(devHost, DOT_NUM_BLOCKS))
    , d_a(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_b(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_c(alpaka::allocBuf<T, Idx>(devAcc, arraySize))
    , d_sum(alpaka::allocBuf<T, Idx>(devAcc, DOT_NUM_BLOCKS))
    , queue(devAcc)
{
    if(arraySize % TBSIZE != 0)
        throw std::runtime_error("Array size must be a multiple of " + std::to_string(TBSIZE));
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
    const auto workdiv = WorkDiv{arraySize / TBSIZE, TBSIZE, 1};
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
    const auto workdiv = WorkDiv{arraySize / TBSIZE, TBSIZE, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_a)}};
    auto viewC = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_c)}};

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
    const auto workdiv = WorkDiv{arraySize / TBSIZE, TBSIZE, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewB = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_b)}};
    auto viewC = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_c)}};

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
    const auto workdiv = WorkDiv{arraySize / TBSIZE, TBSIZE, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_a)}};
    auto viewB = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_b)}};
    auto viewC = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_c)}};

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
    const auto workdiv = WorkDiv{arraySize / TBSIZE, TBSIZE, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_a)}};
    auto viewB = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_b)}};
    auto viewC = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_c)}};

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
    const auto workdiv = WorkDiv{arraySize / TBSIZE, TBSIZE, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, arraySize);

    auto viewA = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_a)}};
    auto viewB = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_b)}};
    auto viewC = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_c)}};

    alpaka::exec<Acc>(queue, workdiv, NstreamKernel{}, viewA, viewB, viewC);
    alpaka::wait(queue);
}

struct DotKernel
{
    template<typename TAcc, typename ViewA, typename ViewB, typename ViewSum>
    ALPAKA_FN_ACC void operator()(const TAcc& acc, ViewA a, ViewB b, ViewSum sum, int arraySize) const
    {
        using T = typename ViewA::RecordDim;

        // TODO - test if sharedMem bug is affecting performance here
        auto& tb_sum = alpaka::declareSharedVar<T[TBSIZE], __COUNTER__>(acc);

        auto [i] = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        const auto [local_i] = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        const auto [totalThreads] = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        tb_sum[local_i] = 0.0;
        for(; i < arraySize; i += totalThreads)
            tb_sum[local_i] += a[i] * b[i];

        const auto [blockDim] = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        for(int offset = blockDim / 2; offset > 0; offset /= 2)
        {
            alpaka::syncBlockThreads(acc);
            if(local_i < offset)
                tb_sum[local_i] += tb_sum[local_i + offset];
        }

        const auto [blockIdx] = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        if(local_i == 0)
            sum[blockIdx] = tb_sum[local_i];
    }
};

template<typename T>
T AlpakaStream<T>::dot()
{
    const auto workdiv = WorkDiv{DOT_NUM_BLOCKS, TBSIZE, 1};
    // auto const workdiv = alpaka::getValidWorkDiv(devAcc, DOT_NUM_BLOCKS * TBSIZE);

    auto viewA = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_a)}};
    auto viewB = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_b)}};
    auto viewSum = llama::View{mapping, llama::Array{alpaka::getPtrNative(d_sum)}};

    alpaka::exec<Acc>(queue, workdiv, DotKernel{}, viewA, viewB, viewSum, arraySize);
    alpaka::wait(queue);

    alpaka::memcpy(queue, sums, d_sum);
    const T* sumPtr = alpaka::getPtrNative(sums);
    return std::reduce(sumPtr, sumPtr + DOT_NUM_BLOCKS);
}

void listDevices()
{
    const auto count = alpaka::getDevCount<Acc>();
    std::cout << "Devices:" << std::endl;
    for(int i = 0; i < count; i++)
        std::cout << i << ": " << getDeviceName(i) << std::endl;
}

std::string getDeviceName(int deviceIndex)
{
    return alpaka::getName(alpaka::getDevByIdx<Acc>(deviceIndex));
}

std::string getDeviceDriver(int device)
{
    return "Not supported";
}

template class AlpakaStream<float>;
template class AlpakaStream<double>;

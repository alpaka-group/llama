// Copyright 2020 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: CC0-1.0

#pragma once

#include <alpaka/alpaka.hpp>
#include <cstddef>

namespace common
{
    constexpr auto threadElemDistMinElem = 2;

    /** Returns a good guess for an optimal number of threads and elements in a
     *  block based on the total number of elements in the block.
     */
    template<typename T_Acc, std::size_t BlockSize, std::size_t HardwareThreads>
    struct ThreadsElemsDistribution
    {
        /// number of elements per thread
        static constexpr std::size_t elemCount = BlockSize;
        /// number of threads per block
        static constexpr std::size_t threadCount = 1u;
    };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<std::size_t BlockSize, std::size_t HardwareThreads, typename Dim, typename Size>
    struct ThreadsElemsDistribution<alpaka::AccGpuCudaRt<Dim, Size>, BlockSize, HardwareThreads>
    {
        static constexpr std::size_t elemCount = threadElemDistMinElem;
        static constexpr std::size_t threadCount = BlockSize / threadElemDistMinElem;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<std::size_t BlockSize, std::size_t HardwareThreads, typename Dim, typename Size>
    struct ThreadsElemsDistribution<alpaka::AccCpuOmp2Threads<Dim, Size>, BlockSize, HardwareThreads>
    {
        static constexpr std::size_t elemCount = (BlockSize + HardwareThreads - 1u) / HardwareThreads;
        static constexpr std::size_t threadCount = HardwareThreads;
    };
#endif

} // namespace common

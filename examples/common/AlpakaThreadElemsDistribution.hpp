/* To the extent possible under law, Alexander Matthes has waived all
 * copyright and related or neighboring rights to this example of LLAMA using
 * the CC0 license, see https://creativecommons.org/publicdomain/zero/1.0 .
 *
 * This example is meant to be "stolen" from to learn how to use LLAMA, which
 * itself is not under the public domain but LGPL3+.
 */

/** \file AlpakaThreadElemsDistribution.hpp
 *  \brief common helper class for getting the right amount of elements per
 *  thread and threads per block based on total number of elements in the block
 *  and the accelator type.
 */

#pragma once

#define THREADELEMDIST_MIN_ELEM 2

namespace common
{
    /** Returns a good guess for an optimal number of threads and elements in a
     *  block based on the total number of elements in the block.
     */
    template<typename T_Acc, std::size_t blockSize, std::size_t hardwareThreads>
    struct ThreadsElemsDistribution
    {
        /// number of elements per thread
        static constexpr std::size_t elemCount = blockSize;
        /// number of threads per block
        static constexpr std::size_t threadCount = 1u;
    };

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size>
    struct ThreadsElemsDistribution<
        alpaka::acc::AccGpuCudaRt<T_Dim, T_Size>,
        blockSize,
        hardwareThreads>
    {
        static constexpr std::size_t elemCount = THREADELEMDIST_MIN_ELEM;
        static constexpr std::size_t threadCount
            = blockSize / THREADELEMDIST_MIN_ELEM;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
    template<
        std::size_t blockSize,
        std::size_t hardwareThreads,
        typename T_Dim,
        typename T_Size>
    struct ThreadsElemsDistribution<
        alpaka::acc::AccCpuOmp2Threads<T_Dim, T_Size>,
        blockSize,
        hardwareThreads>
    {
        static constexpr std::size_t elemCount
            = (blockSize + hardwareThreads - 1u) / hardwareThreads;
        static constexpr std::size_t threadCount = hardwareThreads;
    };
#endif

} // namspace common

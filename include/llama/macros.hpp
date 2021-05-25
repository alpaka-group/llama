// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#if defined(__GNUC__)
#    define LLAMA_INDEPENDENT_DATA _Pragma("GCC ivdep")
#elif defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#    define LLAMA_INDEPENDENT_DATA _Pragma("ivdep")
#elif defined(__clang__)
#    define LLAMA_INDEPENDENT_DATA _Pragma("clang loop vectorize(enable) interleave(enable) distribute(enable)")
#elif defined(_MSC_VER)
#    define LLAMA_INDEPENDENT_DATA __pragma(loop(ivdep))
#else
/// May be put in front of a loop statement. Indicates that all (!) data access
/// inside the loop is indepent, so the loop can be safely vectorized. Example:
/// \code{.cpp}
///     LLAMA_INDEPENDENT_DATA
///     for(int i = 0; i < N; ++i)
///         // because of LLAMA_INDEPENDENT_DATA the compiler knows that a and b
///         // do not overlap and the operation can safely be vectorized
///         a[i] += b[i];
/// \endcode
#    define LLAMA_INDEPENDENT_DATA
#endif

#ifndef LLAMA_FN_HOST_ACC_INLINE
#    if defined(__NVCC__)
#        define LLAMA_FN_HOST_ACC_INLINE __host__ __device__ __forceinline__
#    elif defined(__GNUC__) || defined(__clang__)
#        define LLAMA_FN_HOST_ACC_INLINE inline __attribute__((always_inline))
#    elif defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#        define LLAMA_FN_HOST_ACC_INLINE __forceinline
#    else
/// Some offloading parallelization language extensions such a CUDA, OpenACC or
/// OpenMP 4.5 need to specify whether a class, struct, function or method
/// "resides" on the host, the accelerator (the offloading device) or both.
/// LLAMA supports this with marking every function needed on an accelerator
/// with `LLAMA_FN_HOST_ACC_INLINE`. When using such a language (or e.g. <a
/// href="https://github.com/alpaka-group/alpaka">alpaka</a>)
/// this macro should be defined on the compiler's command line. E.g. for
/// alpaka: -D'LLAMA_FN_HOST_ACC_INLINE=ALPAKA_FN_HOST_ACC'
#        define LLAMA_FN_HOST_ACC_INLINE inline
#        warning LLAMA_FN_HOST_ACC_INLINE not defined for this compiler
#    endif
#endif

#ifndef LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS
#    if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __attribute__((always_inline)) __VA_ARGS__
#    elif defined(__GNUC__) || defined(__INTEL_COMPILER) || defined(__NVCC__)
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__ __attribute__((always_inline))
#    elif defined(_MSC_VER)
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...)                                                               \
            __VA_ARGS__ /* FIXME: MSVC cannot combine constexpr and [[msvc::forceinline]] */
#    else
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__
#        warning LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS not defined for this compiler
#    endif
#endif
#ifndef LLAMA_LAMBDA_INLINE
#    define LLAMA_LAMBDA_INLINE LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS()
#endif

/// Suppresses nvcc warning: 'calling a __host__ function from __host__ __device__ function.'
#if defined(__NVCC__) && !defined(__clang__)
#    define LLAMA_SUPPRESS_HOST_DEVICE_WARNING _Pragma("nv_exec_check_disable")
#else
#    define LLAMA_SUPPRESS_HOST_DEVICE_WARNING
#endif

#if defined(__INTEL_COMPILER) /*|| defined(__INTEL_LLVM_COMPILER)*/
#    define LLAMA_FORCE_INLINE_RECURSIVE _Pragma("forceinline recursive")
#elif defined(_MSC_VER)
#    define LLAMA_FORCE_INLINE_RECURSIVE __pragma(inline_depth(255))
#else
/// Forces the compiler to recursively inline the call hiearchy started by the
/// subsequent function call.
#    define LLAMA_FORCE_INLINE_RECURSIVE
#endif

/// Forces a copy of a value. This is useful to prevent ODR usage of constants
/// when compiling for GPU targets.
#define LLAMA_COPY(x) decltype(x)(x)

// TODO: clang 10 and 11 fail to compile this currently with the issue described here:
// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
// let's try again with clang 12
// Intel LLVM compiler is also using the clang frontend
#if (__has_include(<ranges>) && defined(__cpp_concepts) && !defined(__clang__) && !defined(__INTEL_LLVM_COMPILER))
#    define CAN_USE_RANGES 1
#else
#    define CAN_USE_RANGES 0
#endif

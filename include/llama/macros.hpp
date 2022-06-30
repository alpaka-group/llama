// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#ifdef __INTEL_COMPILER
#    error LLAMA has stopped supporting the Intel Classic Compiler after Intel announced its planned deprecation and \
 replacement by the Intel LLVM-based compiler. Please migrate to the Intel LLVM-based compiler.
#endif

#if defined(__INTEL_LLVM_COMPILER)
// icx supports #pragma ivdep, but it will issue a diagnostic that it needs vectorize(assume_safety) to vectorize.
// Let's keep both pragmas for now.
#    define LLAMA_INDEPENDENT_DATA                                                                                    \
        _Pragma("ivdep") _Pragma("clang loop vectorize(assume_safety) interleave(assume_safety)")
#elif defined(__clang__)
#    define LLAMA_INDEPENDENT_DATA _Pragma("clang loop vectorize(assume_safety) interleave(assume_safety)")
#elif defined(__NVCOMPILER)
#    define LLAMA_INDEPENDENT_DATA _Pragma("ivdep")
#elif defined(__GNUC__)
#    define LLAMA_INDEPENDENT_DATA _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
#    define LLAMA_INDEPENDENT_DATA __pragma(loop(ivdep))
#else
/// May be put in front of a loop statement. Indicates that all (!) data access inside the loop is indepent, so the
/// loop can be safely vectorized. Example: \code{.cpp}
///     LLAMA_INDEPENDENT_DATA
///     for(int i = 0; i < N; ++i)
///         // because of LLAMA_INDEPENDENT_DATA the compiler knows that a and b
///         // do not overlap and the operation can safely be vectorized
///         a[i] += b[i];
/// \endcode
#    define LLAMA_INDEPENDENT_DATA
#endif

#ifndef LLAMA_FORCE_INLINE
#    if defined(__NVCC__)
#        define LLAMA_FORCE_INLINE __forceinline__
#    elif defined(__GNUC__) || defined(__clang__)
#        define LLAMA_FORCE_INLINE inline __attribute__((always_inline))
#    elif defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
#        define LLAMA_FORCE_INLINE __forceinline
#    else
/// Forces the compiler to inline a function annotated with this macro
#        define LLAMA_FORCE_INLINE inline
#        warning LLAMA_FORCE_INLINE is only defined to "inline" for this compiler
#    endif
#endif

#ifndef LLAMA_PRAGMA
#    define LLAMA_PRAGMA(tokens) _Pragma(#    tokens)
#endif

#ifndef LLAMA_UNROLL
#    if defined(__NVCC__) || defined(__NVCOMPILER) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#        define LLAMA_UNROLL(...) LLAMA_PRAGMA(unroll __VA_ARGS__)
#    elif defined(__GNUG__)
#        define LLAMA_UNROLL(...) LLAMA_PRAGMA(GCC unroll __VA_ARGS__)
#    elif defined(_MSC_VER)
// MSVC does not support a pragma for unrolling
#        define LLAMA_UNROLL(...)
#    else
/// Requests the compiler to unroll the loop following this directive. An optional unrolling count may be provided as
/// argument, which must be a constant expression.
#        define LLAMA_UNROLL(...)
#        warning LLAMA_UNROLL is not implemented for your compiler
#    endif
#endif

#ifndef LLAMA_HOST_ACC
#    if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#        define LLAMA_HOST_ACC __host__ __device__
#    elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
#        define LLAMA_HOST_ACC
#    else
/// Some offloading parallelization language extensions such a CUDA, OpenACC or OpenMP 4.5 need to specify whether a
/// class, struct, function or method "resides" on the host, the accelerator (the offloading device) or both. LLAMA
/// supports this with marking every function needed on an accelerator with `LLAMA_HOST_ACC`.
#        define LLAMA_HOST_ACC
#        warning LLAMA_HOST_ACC is only defined empty for this compiler
#    endif
#endif

#define LLAMA_FN_HOST_ACC_INLINE LLAMA_FORCE_INLINE LLAMA_HOST_ACC

#ifndef LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS
#    if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __attribute__((always_inline)) __VA_ARGS__
#    elif defined(__GNUC__) || (defined(__NVCC__) && !defined(_MSC_VER))
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__ __attribute__((always_inline))
#    elif defined(_MSC_VER)
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...)                                                              \
            __VA_ARGS__ /* FIXME: MSVC cannot combine constexpr and [[msvc::forceinline]] */
#    else
#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__
#        warning LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS not defined for this compiler
#    endif
#endif
#ifndef LLAMA_LAMBDA_INLINE
/// Gives strong indication to the compiler to inline the attributed lambda.
#    define LLAMA_LAMBDA_INLINE LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS()
#endif

/// Suppresses nvcc warning: 'calling a __host__ function from __host__ __device__ function.'
#if defined(__NVCC__) && !defined(__clang__)
#    define LLAMA_SUPPRESS_HOST_DEVICE_WARNING _Pragma("nv_exec_check_disable")
#else
#    define LLAMA_SUPPRESS_HOST_DEVICE_WARNING
#endif

#if defined(_MSC_VER)
#    define LLAMA_FORCE_INLINE_RECURSIVE __pragma(inline_depth(255))
#else
/// Forces the compiler to recursively inline the call hiearchy started by the subsequent function call.
#    define LLAMA_FORCE_INLINE_RECURSIVE
#endif

/// Forces a copy of a value. This is useful to prevent ODR usage of constants when compiling for GPU targets.
#define LLAMA_COPY(x) decltype(x)(x)

// https://devblogs.microsoft.com/cppblog/optimizing-the-layout-of-empty-base-classes-in-vs2015-update-2-3/
#if defined(_MSC_VER)
#    define LLAMA_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
#else
#    define LLAMA_DECLSPEC_EMPTY_BASES
#endif

/// Expands to likely if [[likely]] supported by the compiler. Use as [[LLAMA_LIKELY]].
#if __has_cpp_attribute(likely)
#    define LLAMA_LIKELY likely
#else
#    define LLAMA_LIKELY
#endif

/// Expands to unlikely if [[unlikely]] supported by the compiler. Use as [[LLAMA_UNLIKELY]].
#if __has_cpp_attribute(unlikely)
#    define LLAMA_UNLIKELY unlikely
#else
#    define LLAMA_UNLIKELY
#endif

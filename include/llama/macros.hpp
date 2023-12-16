// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

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
#    if defined(__NVCC__) || defined(__HIP__)
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
#    define LLAMA_PRAGMA(tokens) _Pragma(#tokens)
#endif

#ifndef LLAMA_UNROLL
#    if defined(__HIP__) || defined(__NVCC__) || defined(__NVCOMPILER) || defined(__clang__)                          \
        || defined(__INTEL_LLVM_COMPILER)
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

#ifndef LLAMA_ACC
#    if defined(__HIP__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#        define LLAMA_ACC __device__
#    elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
#        define LLAMA_ACC
#    else
#        define LLAMA_ACC
#        warning LLAMA_HOST_ACC is only defined empty for this compiler
#    endif
#endif

#ifndef LLAMA_HOST_ACC
#    if defined(__HIP__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
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

#ifndef LLAMA_SUPPRESS_HOST_DEVICE_WARNING
#    if defined(__NVCC__) && !defined(__clang__)
#        define LLAMA_SUPPRESS_HOST_DEVICE_WARNING _Pragma("nv_exec_check_disable")
#    else
/// Suppresses the nvcc warning: 'calling a __host__ function from __host__ __device__ function.'
/// This macro can be applied to function declarations
#        define LLAMA_SUPPRESS_HOST_DEVICE_WARNING
#    endif
#endif

#ifndef LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
#    ifdef __NVCC__
#        ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#            define LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING                                                          \
                _Pragma("nv_diag_suppress 20011") _Pragma("nv_diag_suppress 20014")
#        else
#            define LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING                                                          \
                _Pragma("diag_suppress 20011") _Pragma("diag_suppress 20014")
#        endif
#    else
/// Suppresses the nvcc warnings:
/// 'calling a __host__ function from a __host__ __device__ function is not allowed' and
/// 'calling a __host__ function("...") from a __host__ __device__ function("...") is not allowed'
/// This macro can be applied before the concerned code block, which then needs to be ended with \ref
/// LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING.
#        define LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
#    endif
#endif
#ifndef LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
#    ifdef __NVCC__
#        ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#            define LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING                                                            \
                _Pragma("nv_diag_default 20011") _Pragma("nv_diag_default 20014")
#        else
#            define LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING _Pragma("diag_default 20011") _Pragma("diag_default 20014")
#        endif
#    else
#        define LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
#    endif
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

/// Expands to consteval if the compilers supports the keyword, otherwise to constexpr.
// TODO(bgruber): reevalute with nvhpc in the future or find workaround
#if defined(__cpp_consteval) && !defined(__NVCOMPILER)
#    define LLAMA_CONSTEVAL consteval
#else
#    define LLAMA_CONSTEVAL constexpr
#endif

#ifndef LLAMA_EXPORT
/// Annotation of all LLAMA public APIs. Expands to nothing by default. Can be defined to 'export' when building LLAMA
/// as a C++20 module.
#    define LLAMA_EXPORT
#endif

// TODO(bgruber): clang 12-15 (libstdc++ from gcc 11.2 or gcc 12.1) fail to compile this currently with the issue
// described here:
// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
// Intel LLVM compiler is also using the clang frontend
#define CAN_USE_RANGES 0
#if __has_include(<version>)
#    include <version>
#    if defined(__cpp_concepts) && defined(__cpp_lib_ranges) && (!defined(__clang__) || __clang_major__ >= 16)        \
        && !defined(__INTEL_LLVM_COMPILER) && (!defined(_MSC_VER) || _MSC_VER > 1932) && !defined(__NVCOMPILER)
#        undef CAN_USE_RANGES
#        define CAN_USE_RANGES 1
#    endif
#endif

// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <boost/predef.h>

#if BOOST_COMP_INTEL == 0 // Work around for broken intel detection
#if defined(__INTEL_COMPILER)
#ifdef BOOST_COMP_INTEL_DETECTION
#undef BOOST_COMP_INTEL_DETECTION
#endif
#define BOOST_COMP_INTEL_DETECTION BOOST_PREDEF_MAKE_10_VVRR(__INTEL_COMPILER)
#if defined(BOOST_COMP_INTEL)
#undef BOOST_COMP_INTEL
#endif
#define BOOST_COMP_INTEL BOOST_COMP_INTEL_DETECTION
#endif
#endif

#if BOOST_COMP_GNUC != 0
#define LLAMA_INDEPENDENT_DATA _Pragma("GCC ivdep")
#elif BOOST_COMP_INTEL != 0
#define LLAMA_INDEPENDENT_DATA _Pragma("ivdep")
#elif BOOST_COMP_CLANG
#define LLAMA_INDEPENDENT_DATA \
    _Pragma("clang loop vectorize(enable)") \
        _Pragma("clang loop interleave(enable)") \
            _Pragma("clang loop distribute(enable)")
#elif defined(_MSC_VER)
#define LLAMA_INDEPENDENT_DATA __pragma(loop(ivdep))
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
#define LLAMA_INDEPENDENT_DATA
#endif

#ifndef LLAMA_FN_HOST_ACC_INLINE
#if BOOST_COMP_NVCC != 0
#define LLAMA_FN_HOST_ACC_INLINE __forceinline__
#elif BOOST_COMP_GNUC != 0
#define LLAMA_FN_HOST_ACC_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define LLAMA_FN_HOST_ACC_INLINE __forceinline
#else
/// Some offloading parallelization language extensions such a CUDA, OpenACC or
/// OpenMP 4.5 need to specify whether a class, struct, function or method
/// "resides" on the host, the accelerator (the offloading device) or both.
/// LLAMA supports this with marking every function needed on an accelerator
/// with `LLAMA_FN_HOST_ACC_INLINE`. When using such a language (or e.g. <a
/// href="https://github.com/ComputationalRadiationPhysics/alpaka">alpaka</a>)
/// this macro should be defined on the compiler's command line. E.g. for
/// alpaka: -D'LLAMA_FN_HOST_ACC_INLINE=ALPAKA_FN_HOST_ACC'
#define LLAMA_FN_HOST_ACC_INLINE inline
#endif
#endif

#if BOOST_COMP_INTEL != 0
#define LLAMA_FORCE_INLINE_RECURSIVE _Pragma("forceinline recursive")
#elif defined(_MSC_VER)
#define LLAMA_FORCE_INLINE_RECURSIVE __pragma(inline_depth(255))
#else
/// Forces the compiler to recursively inline the call hiearchy started by the
/// subsequent function call.
#define LLAMA_FORCE_INLINE_RECURSIVE
#endif

/// Forces a copy of a value. This is useful to prevent ODR usage of constants
/// when compiling for GPU targets.
#define LLAMA_COPY(x) decltype(x)(x)

/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

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
#else
/** Shows that all (!) data access inside inside of a loop is indepent, so the
 *  loop can safely be vectorized although the compiler may not know the data
 *  dependencies completely. Usage looks like this
 * \code{.cpp}
 *  LLAMA_INDEPENDENT_DATA
 *  for (int i = 0; i < N; ++i)
 *      // because of LLAMA_INDEPENDENT_DATA the compiler knows that a and b do
 *      // not overlap and the operation can safely be vectorized
 *      a[i] += b[i];
 * \endcode
 */
#define LLAMA_INDEPENDENT_DATA
#endif

#ifndef LLAMA_FN_HOST_ACC_INLINE
/** Some offloading parallelization language extensions such a CUDA, OpenACC or
 *  OpenMP 4.5 need to specify whether a class, struct, function or method
 *  "resides" on the host, the accelerator (the offloading device) or both.
 *  LLAMA supports this with marking every function wich would be needed on an
 *  accelerator with `LLAMA_FN_HOST_ACC_INLINE`. When using such a language (or
 *  e.g.
 *  <a href="https://github.com/ComputationalRadiationPhysics/alpaka">alpaka</a>
 *  ) the define can be redefined before including LLAMA, e.g. for alpaka:
 * \code{.cpp}
 *  #include <alpaka/alpaka.hpp>
 *  #ifdef __CUDACC__
 *      #define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC __forceinline__
 *  #else
 *      #define LLAMA_FN_HOST_ACC_INLINE ALPAKA_FN_ACC inline
 *  #endif
 *  #include <llama/llama.hpp>
 * \endcode
 */
#if BOOST_COMP_GNUC != 0
#define LLAMA_FN_HOST_ACC_INLINE inline __attribute__((always_inline))
#else
#define LLAMA_FN_HOST_ACC_INLINE inline
#endif
#endif

#ifndef LLAMA_NO_HOST_ACC_WARNING
#if __NVCC__ != 0
#if BOOST_COMP_MSVC != 0
#define LLAMA_NO_HOST_ACC_WARNING __pragma(hd_warning_disable)
#else
#define LLAMA_NO_HOST_ACC_WARNING _Pragma("hd_warning_disable")
#endif
#else
/** Deactivates (wrong negative) warnings about calling host function in an
 *  offloading device (e.g. for CUDA).
 */
#define LLAMA_NO_HOST_ACC_WARNING
#endif
#endif

#if BOOST_COMP_INTEL != 0
#define LLAMA_FORCE_INLINE_RECURSIVE _Pragma("forceinline recursive")
#else
/** If possible forces the compiler to recursively inline the following function
 *  and all child function calls. Should be use carefully as at least the
 *  Intel compiler implementation seems to be buggy.
 */
#define LLAMA_FORCE_INLINE_RECURSIVE
#endif

#define LLAMA_DEREFERENCE(x) decltype(x)(x)

#ifndef LLAMA_IGNORE_LITERAL
#define LLAMA_IGNORE_LITERAL(x)
#endif

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
#   if defined(__INTEL_COMPILER)
#       ifdef BOOST_COMP_INTEL_DETECTION
#           undef BOOST_COMP_INTEL_DETECTION
#       endif
#       define BOOST_COMP_INTEL_DETECTION BOOST_PREDEF_MAKE_10_VVRR(__INTEL_COMPILER)
#       if defined(BOOST_COMP_INTEL)
#           undef BOOST_COMP_INTEL
#       endif
#       define BOOST_COMP_INTEL BOOST_COMP_INTEL_DETECTION
#   endif
#endif

#if BOOST_COMP_GNUC != 0
#   define LLAMA_INDEPENDENT_DATA _Pragma ("GCC ivdep")
#elif BOOST_COMP_INTEL != 0
#   define LLAMA_INDEPENDENT_DATA _Pragma ("ivdep")
#elif BOOST_COMP_CLANG
#   define LLAMA_INDEPENDENT_DATA                                              \
        _Pragma ("clang loop vectorize(enable)")                               \
        _Pragma ("clang loop interleave(enable)")                              \
        _Pragma ("clang loop distribute(enable)")
#else
#   define LLAMA_INDEPENDENT_DATA
#endif

#ifndef LLAMA_FN_HOST_ACC_INLINE
#   define LLAMA_FN_HOST_ACC_INLINE inline
#endif

#ifndef LLAMA_NO_HOST_ACC_WARNING
#   if __NVCC__ != 0
#       if BOOST_COMP_MSVC != 0
#           define LLAMA_NO_HOST_ACC_WARNING __pragma(hd_warning_disable)
#       else
#           define LLAMA_NO_HOST_ACC_WARNING _Pragma("hd_warning_disable")
#       endif
#   else
#       define LLAMA_NO_HOST_ACC_WARNING
#   endif
#endif

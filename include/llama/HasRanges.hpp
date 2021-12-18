// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// TODO(bgruber): clang 10-13 (libstdc++ from gcc 11.2) fail to compile this currently with the issue described here:
// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
// Intel LLVM compiler is also using the clang frontend
#define CAN_USE_RANGES 0
#if __has_include(<version>)
#    include <version>
#    if defined(__cpp_concepts) && defined(__cpp_lib_ranges) && (!defined(__clang__) || __clang_major__ >= 14)        \
        && !defined(__INTEL_LLVM_COMPILER)
#        undef CAN_USE_RANGES
#        define CAN_USE_RANGES 1
#    endif
#endif
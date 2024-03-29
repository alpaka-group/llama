# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18.3)
project(llama-nbody CXX)

option(LLAMA_NBODY_OPENMP OFF)

find_package(fmt CONFIG REQUIRED)
if (LLAMA_NBODY_OPENMP)
	find_package(OpenMP REQUIRED)
endif()
find_package(xsimd QUIET)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()

add_executable(${PROJECT_NAME} nbody.cpp ../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama fmt::fmt)
if (LLAMA_NBODY_OPENMP)
	target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()
if (xsimd_FOUND)
	target_link_libraries(${PROJECT_NAME} PRIVATE xsimd)
else()
	message(WARNING "xsimd not found: n-body built without explicit SIMD support")
endif()

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2) # /fp:fast
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native -fno-math-errno) # -ffast-math
endif()

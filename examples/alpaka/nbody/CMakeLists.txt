# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18.3)
project(llama-alpaka-nbody CXX)

find_package(fmt CONFIG REQUIRED)
find_package(xsimd QUIET)
if (NOT xsimd_FOUND)
	message(WARNING "xsimd not found: alpaka n-body example is disabled")
	return()
endif()
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 1.0 REQUIRED)
alpaka_add_executable(${PROJECT_NAME} nbody.cpp ../../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama fmt::fmt alpaka::alpaka xsimd)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2) # /fp:fast
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native -fno-math-errno) # -ffast-math
endif()

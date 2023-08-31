# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: LGPL-3.0-or-later

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
if (alpaka_ACC_GPU_CUDA_ENABLE AND (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND
	(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.6))
	# nvcc <= 11.5 chokes on `pis(tag::Pos{}, tag::X{})` inside `pPInteraction()` and segfauls
	message(WARNING "nvcc < 11.6 fails to compile the alpaka n-body example, so it is disabled.")
	return()
endif()
alpaka_add_executable(${PROJECT_NAME} nbody.cpp ../../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama fmt::fmt alpaka::alpaka xsimd)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2 /fp:fast>)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native $<IF:$<CXX_COMPILER_ID:NVHPC>,-fast,-ffast-math>)
endif()

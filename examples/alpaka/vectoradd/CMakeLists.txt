# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required (VERSION 3.18.3)
project(llama-alpaka-vectoradd CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 0.9.0 REQUIRED)
if (alpaka_ACC_GPU_CUDA_ENABLE AND (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND
	(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.3) AND (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.4))
	message(WARNING "nvcc 11.3 fails to compile the alpaka vectoradd example, so it is disabled.")
	return()
endif()
alpaka_add_executable(${PROJECT_NAME} vectoradd.cpp ../../common/alpakaHelpers.hpp ../../common/Stopwatch.hpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka)

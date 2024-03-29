# Copyright 2020 Bernhard Manfred Gruber
# SPDX-License-Identifier: ISC

cmake_minimum_required (VERSION 3.18.3)
project(llama-heatequation CXX)

find_package(xsimd QUIET)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()

add_executable(${PROJECT_NAME} heatequation.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
if (xsimd_FOUND)
	target_link_libraries(${PROJECT_NAME} PRIVATE xsimd)
else()
	message(WARNING "xsimd not found: heatequation built without explicit SIMD support")
endif()

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2 /fp:fast)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native $<IF:$<CXX_COMPILER_ID:NVHPC>,-fast,-ffast-math>)
endif()

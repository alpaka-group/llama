# Copyright 2021 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18.3)
project(llama-vectoradd CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} vectoradd.cpp ../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2 /fp:fast)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native $<IF:$<CXX_COMPILER_ID:NVHPC>,-fast,-ffast-math>)
endif()

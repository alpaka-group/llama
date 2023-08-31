# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required (VERSION 3.18.3)
project(llama-alpaka-pic)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(alpaka 1.0 REQUIRED)
alpaka_add_executable(${PROJECT_NAME} pic.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama fmt::fmt alpaka::alpaka OpenMP::OpenMP_CXX)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2 /fp:fast)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native $<IF:$<CXX_COMPILER_ID:NVHPC>,-fast,-ffast-math>)
endif()

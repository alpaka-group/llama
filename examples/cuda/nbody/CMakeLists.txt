# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.18.3)
project(llama-cuda-nbody CUDA)

find_package(CUDAToolkit) # for include directories
find_package(fmt CONFIG REQUIRED)
if (NOT TARGET llama::llama)
    find_package(llama REQUIRED)
endif ()
add_executable(${PROJECT_NAME} nbody.cu ../../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17)
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(${PROJECT_NAME} PUBLIC
            --expt-relaxed-constexpr --use_fast_math
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:--compiler-options -Wall,-Wextra>
            )
else ()
    target_compile_options(${PROJECT_NAME} PUBLIC -ffast-math -Wall -Wextra)
endif ()
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama CUDA::cudart fmt::fmt)

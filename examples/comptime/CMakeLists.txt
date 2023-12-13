# Copyright 2023 Bernhard Manfred Gruber
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required (VERSION 3.18.3)
project(llama-comptime CXX)

set(LLAMA_COMPTIME_RECORD_DIM_SIZE 20 CACHE STRING "comptime example record dimension size")

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} comptime.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DLLAMA_COMPTIME_RECORD_DIM_SIZE=${LLAMA_COMPTIME_RECORD_DIM_SIZE})

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
	target_compile_options(${PROJECT_NAME} PRIVATE -fbracket-depth=1500)
endif()

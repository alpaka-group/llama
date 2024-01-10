# Copyright 2024 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18.3)

project(llama-nbody-baseline CXX)
add_executable(${PROJECT_NAME} nbody_baseline.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

project(llama-nbody-ported CXX)
if (NOT TARGET llama::llama)
	find_package(llama CONFIG REQUIRED)
endif()
add_executable(${PROJECT_NAME} nbody_ported.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

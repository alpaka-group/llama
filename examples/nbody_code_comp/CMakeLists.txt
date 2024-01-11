# Copyright 2024 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18.3)

project(llama-nbody-aos-baseline CXX)
add_executable(${PROJECT_NAME} nbody-AoS-baseline.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

project(llama-nbody-soa CXX)
add_executable(${PROJECT_NAME} nbody-SoA.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

project(llama-nbody-aosoa CXX)
add_executable(${PROJECT_NAME} nbody-AoSoA.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)

project(llama-nbody-ported CXX)
if (NOT TARGET llama::llama)
	find_package(llama CONFIG REQUIRED)
endif()
add_executable(${PROJECT_NAME} nbody-ported.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

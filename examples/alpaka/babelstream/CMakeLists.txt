# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required(VERSION 3.22)
project(llama-alpaka-babelstream CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 1.0 REQUIRED)
alpaka_add_executable(${PROJECT_NAME} main.cpp Stream.h AlpakaStream.cpp AlpakaStream.h)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PUBLIC ALPAKA)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka)

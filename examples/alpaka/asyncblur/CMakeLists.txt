# Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required (VERSION 3.18.3)
project(llama-alpaka-asyncblur CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 1.0 REQUIRED)
alpaka_add_executable(${PROJECT_NAME} asyncblur.cpp ../../common/Stopwatch.hpp)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ../../../thirdparty/stb/include)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka)

# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18.3)
project(llama-memmap CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(Boost COMPONENTS iostreams)
add_executable(${PROJECT_NAME} memmap.cpp)
if (MSVC)
	# MSVC has a constexpr bug when printing the triangle in C++17 mode
	target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
else()
	target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama Boost::iostreams)

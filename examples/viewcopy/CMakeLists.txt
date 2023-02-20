# Copyright 2020 Bernhard Manfred Gruber
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required (VERSION 3.18.3)
project(llama-viewcopy CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
	message(WARNING "nvc++ cannot compile the viewcopy example")
	return()
endif()

find_package(OpenMP REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} viewcopy.cpp ../common/Stopwatch.hpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama OpenMP::OpenMP_CXX)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
	if (MSVC_VERSION VERSION_GREATER_EQUAL 1930 AND MSVC_VERSION VERSION_LESS 1932)
		# VS 2022 has a new lambda processor that has troubles with OpenMP pragmas:
		# https://developercommunity.visualstudio.com/t/OpenMP-in-lambda-expression-compile-erro/1501041
		# And the workaround flag /Zc:lambda- needs to come after the CXX standard,
		# so we have to workaround CMake as well and not use cxx_std_20
		target_compile_options(${PROJECT_NAME} PRIVATE /std:c++20 /Zc:lambda-)
	else()
		target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
	endif()
else()
	target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
endif()

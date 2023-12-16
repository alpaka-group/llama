# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18)
project(llama-root-lhcb_analysis)

find_package(ROOT REQUIRED)
find_package(OpenMP REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} lhcb.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE
		ROOT::Hist ROOT::Graf ROOT::Gpad ROOT::ROOTNTuple llama::llama OpenMP::OpenMP_CXX)
if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
endif()

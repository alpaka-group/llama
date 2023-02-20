# Copyright 2022 Bernhard Manfred Gruber
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required (VERSION 3.18.3)
project(llama-bufferguard CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} bufferguard.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

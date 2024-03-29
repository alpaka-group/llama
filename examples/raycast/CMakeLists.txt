# Copyright 2020 Bernhard Manfred Gruber
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required (VERSION 3.18.3)
project (llama-raycast CXX)

find_package(tinyobjloader CONFIG)
if (NOT tinyobjloader_FOUND)
	message(WARNING "tinyobjloader not found: raycast example will not be built")
	return()
endif()

if (NOT TARGET llama::llama)
	find_package(llama CONFIG REQUIRED)
endif()
add_executable(${PROJECT_NAME} raycast.cpp)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ../../thirdparty/stb/include)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyobjloader::tinyobjloader llama::llama)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2 /fp:fast)
	target_compile_definitions(${PROJECT_NAME} PRIVATE _USE_MATH_DEFINES )
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native $<IF:$<CXX_COMPILER_ID:NVHPC>,-fast,-ffast-math>)
endif()

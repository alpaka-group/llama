cmake_minimum_required (VERSION 3.15)
project(llama-alpaka-nbody)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED on)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 0.5.0 REQUIRED)
alpaka_add_executable(${PROJECT_NAME} nbody.cpp ../../common/alpakaHelpers.hpp ../../common/Stopwatch.hpp)
target_compile_definitions(${PROJECT_NAME} PUBLIC LLAMA_FN_HOST_ACC_INLINE=ALPAKA_FN_HOST_ACC)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka)

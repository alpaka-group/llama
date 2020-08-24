cmake_minimum_required (VERSION 3.3)
project(llama-alpaka-nbody)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED on)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 0.5.0 REQUIRED)
ALPAKA_ADD_EXECUTABLE(${PROJECT_NAME} nbody.cpp ../../common/alpakaHelpers.hpp ../../common/Chrono.hpp)
target_compile_definitions(${PROJECT_NAME} PUBLIC LLAMA_FN_HOST_ACC_INLINE=ALPAKA_FN_HOST_ACC)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka)

install(FILES "${PROJECT_BINARY_DIR}/llama-alpaka-nbody" DESTINATION "bin")


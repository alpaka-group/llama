cmake_minimum_required (VERSION 3.3)
project(llama-vectoradd)

set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} vectoradd.cpp)
#target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

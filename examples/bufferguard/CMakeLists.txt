cmake_minimum_required (VERSION 3.15)
project(llama-bufferguard)

set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} bufferguard.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

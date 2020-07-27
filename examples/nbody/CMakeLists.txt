cmake_minimum_required (VERSION 3.3)
project(llama-nbody)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED on)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} nbody.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(${PROJECT_NAME} PRIVATE -fno-math-errno) # sqrt prevents vectorization otherwise
endif()

install(FILES "${PROJECT_BINARY_DIR}/llama-nbody" DESTINATION "bin")


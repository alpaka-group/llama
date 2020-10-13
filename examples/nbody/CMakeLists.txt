cmake_minimum_required (VERSION 3.3)
project(llama-nbody)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} nbody.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-fno-math-errno # sqrt prevents vectorization otherwise
		-march=native
		-ffast-math
	)
endif()

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" DESTINATION "bin")


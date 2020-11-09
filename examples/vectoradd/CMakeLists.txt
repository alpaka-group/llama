cmake_minimum_required (VERSION 3.15)
project(llama-vectoradd)

set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} vectoradd.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-march=native
		-ffast-math
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
		/fp:fast
	)
endif()

cmake_minimum_required (VERSION 3.15)
project(llama-viewcopy)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} viewcopy.cpp ../common/Stopwatch.hpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama OpenMP::OpenMP_CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-march=native
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
	)
endif()

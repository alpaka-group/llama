cmake_minimum_required (VERSION 3.15)
project(llama-viewcopy)

find_package(OpenMP REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} viewcopy.cpp ../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama OpenMP::OpenMP_CXX)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-march=native
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
	)
endif()

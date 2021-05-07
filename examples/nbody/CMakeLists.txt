cmake_minimum_required (VERSION 3.15)
project(llama-nbody)

find_package(fmt CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Vc QUIET)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()

add_executable(${PROJECT_NAME} nbody.cpp ../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama fmt::fmt OpenMP::OpenMP_CXX)
if (Vc_FOUND)
	target_link_libraries(${PROJECT_NAME} PRIVATE Vc::Vc)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	target_compile_options(${PROJECT_NAME} PRIVATE
		#-fno-math-errno # sqrt prevents vectorization otherwise
		-march=native
		-ffast-math
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
		/fp:fast
	)
endif()

cmake_minimum_required (VERSION 3.15)
project(llama-nbody_benchmark CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(fmt CONFIG REQUIRED)
add_executable(${PROJECT_NAME} nbody.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama fmt::fmt)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR
	CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		#-fno-math-errno # sqrt prevents vectorization otherwise
		-march=native
		-ffast-math
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
		/fp:fast
		/bigobj
	)
endif()

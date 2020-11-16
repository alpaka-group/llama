cmake_minimum_required (VERSION 3.15)
project(llama-alpaka-nbody)

find_package(Vc REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 0.5.0 REQUIRED)
alpaka_add_executable(${PROJECT_NAME} nbody.cpp ../../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_definitions(${PROJECT_NAME} PUBLIC "LLAMA_FN_HOST_ACC_INLINE=ALPAKA_FN_HOST_ACC inline")
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka Vc::Vc)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-fno-math-errno # sqrt prevents vectorization otherwise
		-march=native
		-ffast-math
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
		/fp:fast
	)
endif()

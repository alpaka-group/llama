cmake_minimum_required (VERSION 3.18.3)
project(llama-cuda-viewcopy CUDA)

find_package(CUDAToolkit) # for include directories
find_package(fmt CONFIG REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} viewcopy.cu ../../common/Stopwatch.hpp)
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
	target_compile_options(${PROJECT_NAME} PRIVATE --expt-relaxed-constexpr)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama CUDA::cudart fmt::fmt)
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17)

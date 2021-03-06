cmake_minimum_required (VERSION 3.18.3)
project(llama-cuda-nbody CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "35" CACHE STRING "CUDA architectures to compile for")

find_package(CUDAToolkit) # for include directories
find_package(fmt CONFIG REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} nbody.cu ../../common/Stopwatch.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17)
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama CUDA::cudart fmt::fmt)

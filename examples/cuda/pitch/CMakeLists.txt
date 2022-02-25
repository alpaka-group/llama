cmake_minimum_required (VERSION 3.18.3)
project(llama-cuda-pitch CXX CUDA)

find_package(CUDAToolkit) # for include directories
find_package(fmt CONFIG REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} pitch.cu)
target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17)
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ../../../thirdparty/stb/include)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama CUDA::cudart fmt::fmt)
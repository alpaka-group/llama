cmake_minimum_required (VERSION 3.15)
project(llama-thrust CXX CUDA)

set (Thrust_DIR "../../thirdparty/thrust/thrust/cmake/")
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust FROM_OPTIONS)
find_package(OpenMP REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} thrust.cu ../common/Stopwatch.hpp ../common/hostname.hpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:
    -forward-unknown-to-host-compiler
    --expt-extended-lambda
    --expt-relaxed-constexpr
    --use_fast_math
    --ftemplate-backtrace-limit 500>)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama Thrust OpenMP::OpenMP_CXX)

# OpenMP is not added to the compiler commandline by cmake when using nvcc ..
if (NOT MSVC)
	find_package(TBB REQUIRED)
	target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb)
	target_compile_options(${PROJECT_NAME} PUBLIC -march=native -ffast-math -fopenmp)
else()
	target_compile_options(${PROJECT_NAME} PUBLIC /openmp)
endif()

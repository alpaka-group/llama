cmake_minimum_required (VERSION 3.15)
project(llama-daxpy CXX)

#find_package(Vc QUIET)
find_package(OpenMP REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()

add_executable(${PROJECT_NAME} daxpy.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama OpenMP::OpenMP_CXX)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2 /fp:fast)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native -ffast-math)
endif()

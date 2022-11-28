cmake_minimum_required (VERSION 3.18.3)

find_package(OpenMP REQUIRED)
set(STREAM_ARRAY_SIZE 80000000) # if we make this too large, we will get linker errors due to ELF/PE limits

project(stream C)

add_executable(${PROJECT_NAME} stream.cpp)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DSTREAM_ARRAY_SIZE=${STREAM_ARRAY_SIZE})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
endif()

project(llama-stream CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} stream.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PRIVATE -DSTREAM_ARRAY_SIZE=${STREAM_ARRAY_SIZE} -DTUNED)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama OpenMP::OpenMP_CXX)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE /arch:AVX2)
else()
	target_compile_options(${PROJECT_NAME} PRIVATE -march=native)
endif()

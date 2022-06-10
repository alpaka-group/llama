cmake_minimum_required (VERSION 3.18.3)
project(llama-bytesplit CXX)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} bytesplit.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

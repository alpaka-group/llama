cmake_minimum_required (VERSION 3.3)
project(llama-heatequation)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED on)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} heatequation.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

install(FILES "${PROJECT_BINARY_DIR}/llama-heatequation" DESTINATION "bin")


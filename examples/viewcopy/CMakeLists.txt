cmake_minimum_required (VERSION 3.3)
project(llama-viewcopy)

set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} viewcopy.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" DESTINATION "bin")


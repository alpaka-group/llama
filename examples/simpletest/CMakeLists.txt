cmake_minimum_required (VERSION 3.3)
project (llama-simpletest)

if (NOT TARGET llama::llama)
	find_package(llama CONFIG REQUIRED)
endif()
add_executable(${PROJECT_NAME} simpletest.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

install(FILES "${PROJECT_BINARY_DIR}/llama-simpletest" DESTINATION "bin")

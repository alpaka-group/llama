cmake_minimum_required (VERSION 3.3)
project (llama-virtualviewtest)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} virtualviewtest.cpp ../common/Dummy.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

install(FILES "${PROJECT_BINARY_DIR}/llama-virtualviewtest" DESTINATION "bin")

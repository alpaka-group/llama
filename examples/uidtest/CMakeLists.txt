cmake_minimum_required (VERSION 3.3)
project (llama-uidtest)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} uidtest.cpp)
target_link_libraries(${PROJECT_NAME} llama::llama)

install(FILES "${PROJECT_BINARY_DIR}/llama-uidtest" DESTINATION "bin")

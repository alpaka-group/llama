cmake_minimum_required (VERSION 3.3)
project(llama-nbody)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 0.4.0 REQUIRED)
ALPAKA_ADD_EXECUTABLE(${PROJECT_NAME} nbody.cpp ../common/Dummy.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka)

install(FILES "${PROJECT_BINARY_DIR}/llama-nbody" DESTINATION "bin")


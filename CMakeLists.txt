cmake_minimum_required (VERSION 3.3)
project (llama)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

ADD_SUBDIRECTORY("examples")

configure_file (
	"${PROJECT_SOURCE_DIR}/cmake/llama-config-version.cmake.in"
	"${PROJECT_BINARY_DIR}/cmake/llama-config-version.cmake"
	@ONLY
	)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/llama" DESTINATION "include" )
install( FILES "${CMAKE_CURRENT_SOURCE_DIR}/llama-config.cmake" DESTINATION "lib/cmake/llama" )
install( FILES "${PROJECT_BINARY_DIR}/cmake/llama-config-version.cmake" DESTINATION "lib/cmake/llama" )

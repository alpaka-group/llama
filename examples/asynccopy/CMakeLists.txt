cmake_minimum_required (VERSION 3.3)
project(llama-asynccopy)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka 0.5.0 REQUIRED)
ALPAKA_ADD_EXECUTABLE(${PROJECT_NAME} asynccopy.cpp ../common/Dummy.cpp stb_image.h stb_image_write.h)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama alpaka::alpaka)

install(FILES "${PROJECT_BINARY_DIR}/llama-asynccopy" DESTINATION "bin" )

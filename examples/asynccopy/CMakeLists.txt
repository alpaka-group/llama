cmake_minimum_required (VERSION 3.3)
project(llama-asynccopy)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(alpaka REQUIRED)
find_package(png++ REQUIRED)
message(STATUS ${PNG_LIBRARIES})
ALPAKA_ADD_EXECUTABLE(${PROJECT_NAME} asynccopy.cpp ../common/Dummy.cpp)
target_link_libraries(${PROJECT_NAME} PUBLIC llama::llama alpaka::alpaka)

install(FILES "${PROJECT_BINARY_DIR}/llama-asynccopy" DESTINATION "bin" )

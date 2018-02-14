cmake_minimum_required (VERSION 3.3)
project (llama-simpletest)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

SET(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE STRING "The location of the llama library")
LIST(APPEND CMAKE_PREFIX_PATH "${LLAMA_ROOT}")

find_package(llama 0.1.0 REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${llama_INCLUDE_DIR})
add_definitions(${llama_DEFINITIONS})

include_directories(llama-simpletest ${INCLUDE_DIRS})
add_executable(llama-simpletest ${CMAKE_CURRENT_LIST_DIR}/simpletest.cpp)
target_link_libraries(llama-simpletest ${LIBRARIES})

install( FILES "${PROJECT_BINARY_DIR}/llama-simpletest" DESTINATION "bin" )

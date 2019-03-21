cmake_minimum_required (VERSION 3.3)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
SET(_TARGET_NAME "llama-nbody")

project(${_TARGET_NAME})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

SET(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE STRING "The location of the llama library")
LIST(APPEND CMAKE_PREFIX_PATH "${LLAMA_ROOT}")

find_package(llama 0.1.0 REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${llama_INCLUDE_DIR})
add_definitions(${llama_DEFINITIONS})

################################################################################
# Alpaka LIB
################################################################################
find_package(alpaka REQUIRED)
set(LIBRARIES ${LIBRARIES} ${alpaka_LIBRARIES})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
set(DEFINITIONS ${DEFINITIONS} ${alpaka_DEFINITIONS})
set(DEFINITIONS ${DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})

include_directories(${_TARGET_NAME} ${INCLUDE_DIRS})
ALPAKA_ADD_EXECUTABLE(
	${_TARGET_NAME}
	${CMAKE_CURRENT_LIST_DIR}/nbody.cpp
	${CMAKE_CURRENT_LIST_DIR}/../common/Dummy.cpp
)
TARGET_COMPILE_OPTIONS( ${_TARGET_NAME} PRIVATE ${DEFINITIONS})
TARGET_LINK_LIBRARIES( ${_TARGET_NAME} PUBLIC "alpaka")

install( FILES "${PROJECT_BINARY_DIR}/llama-nbody" DESTINATION "bin" )


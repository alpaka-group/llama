cmake_minimum_required (VERSION 3.3)

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)
SET(_TARGET_NAME "llama-asynccopy")

project(${_TARGET_NAME})

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/../Modules/")

SET(LLAMA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." CACHE STRING "The location of the llama library")
LIST(APPEND CMAKE_PREFIX_PATH "${LLAMA_ROOT}")

find_package(llama 0.1.0 REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${llama_INCLUDE_DIR})
add_definitions(${llama_DEFINITIONS})

################################################################################
# Alpaka LIB
################################################################################
set(ALPAKA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../alpaka/" CACHE STRING  "The location of the alpaka library")
LIST(APPEND CMAKE_PREFIX_PATH "${ALPAKA_ROOT}")
find_package(alpaka REQUIRED)
set(LIBRARIES ${LIBRARIES} ${alpaka_LIBRARIES})
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
set(DEFINITIONS ${DEFINITIONS} ${alpaka_DEFINITIONS})
set(DEFINITIONS ${DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})

find_package(png++ REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${png++_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${PNG_LIBRARIES})

message(STATUS ${PNG_LIBRARIES})


include_directories(${_TARGET_NAME} ${INCLUDE_DIRS})
ALPAKA_ADD_EXECUTABLE(
	${_TARGET_NAME}
	${CMAKE_CURRENT_LIST_DIR}/asynccopy.cpp
	${CMAKE_CURRENT_LIST_DIR}/../common/Dummy.cpp
)
TARGET_COMPILE_OPTIONS( ${_TARGET_NAME} PRIVATE ${DEFINITIONS})
TARGET_LINK_LIBRARIES( ${_TARGET_NAME} PUBLIC "alpaka" ${LIBRARIES})

install( FILES "${PROJECT_BINARY_DIR}/llama-asynccopy" DESTINATION "bin" )


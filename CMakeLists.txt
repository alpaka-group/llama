cmake_minimum_required (VERSION 3.3)
project (llama)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

# LLAMA include dir until a CMakeConfig.cmake is created
set(INCLUDE_DIRS ${INCLUDE_DIRS} "include/llama")

# set(ALPAKA_ROOT "${CMAKE_SOURCE_DIR}/../alpaka/" CACHE STRING  "The location of the alpaka library")
# set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${ALPAKA_ROOT})
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/Modules/")

# find_package(alpaka REQUIRED)
# set(LIBRARIES ${LIBRARIES} ${alpaka_LIBRARIES})
# set(INCLUDE_DIRS ${INCLUDE_DIRS} ${alpaka_INCLUDE_DIRS})
# set(DEFINITIONS ${DEFINITIONS} ${alpaka_DEFINITIONS})
# set(DEFINITIONS ${DEFINITIONS} ${ALPAKA_DEV_COMPILE_OPTIONS})

find_package(Boost MODULE REQUIRED)
set(INCLUDE_DIRS ${INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
add_definitions(-DBOOST_ALL_NO_LIB)

include_directories(llamatest1 ${INCLUDE_DIRS})
add_executable(llamatest1 ${CMAKE_SOURCE_DIR}/llamatest1.cpp)
target_link_libraries(llamatest1 ${LIBRARIES})

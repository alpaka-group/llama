cmake_minimum_required (VERSION 3.3)
project (llama)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/Modules/")

####################
# General examples #
####################

option(LLAMA_BUILD_EXAMPLE_SIMPLETEST "Building (and installing) the simpletest example" ON)
if (LLAMA_BUILD_EXAMPLE_SIMPLETEST)
	ADD_SUBDIRECTORY("simpletest")
endif()

option(LLAMA_BUILD_EXAMPLE_UIDTEST "Building (and installing) the uidtest example" ON)
if (LLAMA_BUILD_EXAMPLE_UIDTEST)
	ADD_SUBDIRECTORY("uidtest")
endif()

option(LLAMA_BUILD_EXAMPLE_TREEMAPTEST "Building (and installing) the treemaptest example" ON)
if (LLAMA_BUILD_EXAMPLE_TREEMAPTEST)
	ADD_SUBDIRECTORY("treemaptest")
endif()

option(LLAMA_BUILD_EXAMPLE_VIRTUALVIEWTEST "Building (and installing) the virtualviewtest example" ON)
if (LLAMA_BUILD_EXAMPLE_VIRTUALVIEWTEST)
	ADD_SUBDIRECTORY("virtualviewtest")
endif()

###################
# Alpaka examples #
###################

find_package(alpaka QUIET)

if (ALPAKA_FOUND)
	option(LLAMA_BUILD_EXAMPLE_NBODY "Building (and installing) the nbody example, needs alpaka" ON)
	option(LLAMA_BUILD_EXAMPLE_VECTORADD "Building (and installing) the vectoradd example, needs alpaka" ON)
	find_package(png++ QUIET)
	if (png++_FOUND)
		option(LLAMA_BUILD_EXAMPLE_ASYNCCOPY "Building (and installing) the asynccopy example, needs alpaka and png++" ON)
	else()
		option(LLAMA_BUILD_EXAMPLE_ASYNCCOPY "Building (and installing) the asynccopy example, needs alpaka and png++" OFF)
	endif()
else()
	option(LLAMA_BUILD_EXAMPLE_NBODY "Building (and installing) the nbody example, needs alpaka" OFF)
	option(LLAMA_BUILD_EXAMPLE_VECTORADD "Building (and installing) the vectoradd example, needs alpaka" OFF)
	option(LLAMA_BUILD_EXAMPLE_ASYNCCOPY "Building (and installing) the asynccopy example, needs alpaka and png++" OFF)
endif()

if (LLAMA_BUILD_EXAMPLE_NBODY)
	ADD_SUBDIRECTORY("nbody")
endif()

if (LLAMA_BUILD_EXAMPLE_VECTORADD)
	ADD_SUBDIRECTORY("vectoradd")
endif()

if (LLAMA_BUILD_EXAMPLE_ASYNCCOPY)
	ADD_SUBDIRECTORY("asynccopy")
endif()

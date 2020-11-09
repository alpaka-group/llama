cmake_minimum_required (VERSION 3.15)
project(llama-viewcopy)

set(CMAKE_CXX_STANDARD 20)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} viewcopy.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	# GCC will try to find TBB and if it is installed use it for the parallel STL
	# unfortunately, it does not link against TBB automatically
	find_package(TBB CONFIG QUIET)
	if (TBB_FOUND)
		# TBB needs pthreads
		find_package(Threads REQUIRED)
		target_link_libraries(${PROJECT_NAME} PRIVATE TBB::tbb Threads::Threads)
	endif()
endif()

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" DESTINATION "bin")


cmake_minimum_required (VERSION 3.3)
project(llama-heatequation)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED on)

find_package(Vc QUIET)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()

add_executable(${PROJECT_NAME} heatequation.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama)
if (Vc_FOUND)
	target_link_libraries(${PROJECT_NAME} PRIVATE Vc::Vc)
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-march=native
		-ffast-math
	)
elseif(MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
		/fp:fast
	)
endif()

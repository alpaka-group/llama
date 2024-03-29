cmake_minimum_required (VERSION 3.18.3)
project (llama CXX)

# llama
find_package(Boost 1.74.0 REQUIRED)
find_package(fmt CONFIG QUIET)
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_link_libraries(${PROJECT_NAME} INTERFACE Boost::headers)
add_compile_definitions(BOOST_ATOMIC_NO_LIB) # we don't need the compiled part in LLAMA or its examples
if (fmt_FOUND)
	target_link_libraries(${PROJECT_NAME} INTERFACE fmt::fmt)
	if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFMT_USE_NONTYPE_TEMPLATE_PARAMETERS=0 -DFMT_USE_NONTYPE_TEMPLATE_ARGS=0)
	endif()
else()
	message(WARNING "The fmt library was not found. You cannot use llama's dumping facilities.")
endif()

# llama::llama to make subdirectory projects work
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# llama IDE target to make source browsable/editable in IDEs
file(GLOB_RECURSE llamaSources "${CMAKE_CURRENT_SOURCE_DIR}/include/**")
add_custom_target("llamaIde" SOURCES ${llamaSources})
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/include/llama" FILES ${llamaSources})

# default build type, see: https://www.kitware.com/cmake-and-the-default-build-type/
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified.")
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY VALUE "Release")
endif()

if (MSVC)
	# FIXME(bgruber): alpaka uses M_PI, so we need to make it available on MSVC. This may be fixed in alpaka 1.0.0.
	target_compile_definitions(llama INTERFACE _USE_MATH_DEFINES)
	target_compile_options(${PROJECT_NAME} INTERFACE /Zc:lambda) # needed in C++17 mode, remove when upgrading to C++20
endif()

# CUDA
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	enable_language(CUDA)
	set(CMAKE_CUDA_ARCHITECTURES "35" CACHE STRING "CUDA architectures to compile for")

	if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
		target_compile_definitions(${PROJECT_NAME} INTERFACE -DFMT_USE_FLOAT128=0)

		# Workaround for clang as CUDA compiler with libstdc++ 12
		file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda_libstdc++12_workaround.hpp"
				"#include <__clang_cuda_runtime_wrapper.h>\n"
				"#if defined(__clang__) && defined(__CUDA__) && defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 12 && defined(__noinline__)\n"
				"#    undef __noinline__\n"
				"#endif\n")
		target_compile_options(${PROJECT_NAME} INTERFACE -include "${CMAKE_CURRENT_BINARY_DIR}/clang_cuda_libstdc++12_workaround.hpp")
	endif()
else()
	message(WARNING "Could not find CUDA. Try setting CMAKE_CUDA_COMPILER. CUDA tests and examples are disabled.")
endif()

# tests
include(CMakeDependentOption)
cmake_dependent_option(LLAMA_COMPILE_TESTS_AS_CUDA "Sets the language of all test code to CUDA." OFF "BUILD_TESTING;CMAKE_CUDA_COMPILER" OFF)

option(BUILD_TESTING "" OFF)
include(CTest)
if (BUILD_TESTING)
	option(LLAMA_SYSTEM_CATCH2 "Use the system provided Catch2. This may result in a build failure, if Catch2 was compiled with a different C++ version as the LLAMA tests." ON)
	if (LLAMA_SYSTEM_CATCH2)
		find_package(Catch2 3.0.1 REQUIRED)
		include(Catch)
	else()
		# get Catch2 v3 and build it from source with the same C++ standard as the tests
		Include(FetchContent)
		FetchContent_Declare(Catch2 GIT_REPOSITORY https://github.com/catchorg/Catch2.git GIT_TAG v3.0.1)
		FetchContent_MakeAvailable(Catch2)
		target_compile_features(Catch2 PUBLIC cxx_std_20)
		include(Catch)

		# hide Catch2 cmake variables by default in cmake gui
		get_cmake_property(variables VARIABLES)
		foreach (var ${variables})
			if (var MATCHES "^CATCH_")
				mark_as_advanced(${var})
			endif()
		endforeach()
	endif()

	file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/tests/**")
	add_executable(tests ${testSources})
	catch_discover_tests(tests)
	source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/tests" FILES ${testSources})
	target_compile_features(tests PRIVATE cxx_std_20)

	if (LLAMA_COMPILE_TESTS_AS_CUDA)
		foreach(f ${testSources})
			set_source_files_properties(${f} PROPERTIES LANGUAGE CUDA)
		endforeach()
		target_compile_options(tests PRIVATE --extended-lambda --expt-relaxed-constexpr)
	endif()

	if (MSVC)
		target_compile_options(tests PRIVATE /permissive- /constexpr:steps10000000 /diagnostics:caret)
	else()
		target_compile_options(tests PRIVATE -Wall -Wextra $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror=narrowing> -march=native)
		if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
			target_compile_options(tests PRIVATE -Wno-missing-braces)
		endif()
	endif()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
		target_compile_options(tests PRIVATE -fconstexpr-steps=10000000)
	endif()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
		target_compile_options(tests PRIVATE -fp-model=precise)
	endif()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
		target_compile_options(tests PRIVATE --display_error_number -Wc,--pending_instantiations=0)
		target_compile_options(tests PRIVATE --diag_suppress=177) # disable: #177-D: variable "<unnamed>::autoRegistrar72" was declared but never referenced
	endif()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		if (LLAMA_ENABLE_ASAN_FOR_TESTS AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
			target_compile_options(tests PRIVATE -Wno-maybe-uninitialized) # triggered inside std::function by std::regex
		endif()
		if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0)
			target_compile_options(tests PRIVATE -Wno-dangling-reference) # triggered by an access involving RecordRef, so basically everywhere
		endif()
	endif()
	target_link_libraries(tests PRIVATE Catch2::Catch2WithMain llama::llama)

	option(LLAMA_ENABLE_ASAN_FOR_TESTS "Enables address sanitizer for tests" OFF)
	if (LLAMA_ENABLE_ASAN_FOR_TESTS)
		if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
			target_compile_options(tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
			target_link_options   (tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
		elseif(MSVC)
			target_compile_options(tests PRIVATE /fsanitize=address)
			target_link_options   (tests PRIVATE /fsanitize=address)
		endif()
	endif()

	option(LLAMA_ENABLE_COVERAGE_FOR_TESTS "Enables code coverage for tests" OFF)
	if (LLAMA_ENABLE_COVERAGE_FOR_TESTS)
		if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
			target_compile_options(tests PRIVATE --coverage)
			target_link_options(tests PRIVATE --coverage)
		endif()
	endif()
endif()

# examples
option(LLAMA_BUILD_EXAMPLES "Building (and installing) the examples" OFF)
if (LLAMA_BUILD_EXAMPLES)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

	# general examples
	add_subdirectory("examples/vectoradd")
	add_subdirectory("examples/nbody")
	add_subdirectory("examples/nbody_code_comp")
	add_subdirectory("examples/heatequation")
	add_subdirectory("examples/viewcopy")
	add_subdirectory("examples/bufferguard")
	add_subdirectory("examples/raycast")
	add_subdirectory("examples/bytesplit")
	add_subdirectory("examples/bitpackint")
	add_subdirectory("examples/bitpackfloat")
	add_subdirectory("examples/memmap")
	add_subdirectory("examples/stream")
	add_subdirectory("examples/falsesharing")
	add_subdirectory("examples/comptime")

	# alpaka examples
	find_package(alpaka 1.0)
	if (_alpaka_FOUND)
		add_subdirectory("examples/alpaka/nbody")
		add_subdirectory("examples/alpaka/vectoradd")
		add_subdirectory("examples/alpaka/asyncblur")
		add_subdirectory("examples/alpaka/pic")
		add_subdirectory("examples/alpaka/daxpy")
		add_subdirectory("examples/alpaka/babelstream")
	else()
		message(WARNING "Could not find alpaka. Alpaka examples are disabled.")
	endif()

	# ROOT examples
	find_package(ROOT QUIET)
	if (ROOT_FOUND)
		add_subdirectory("examples/root/lhcb_analysis")
	endif()

	# CUDA examples
	if (CMAKE_CUDA_COMPILER)
		add_subdirectory("examples/cuda/nbody")
		add_subdirectory("examples/cuda/pitch")
		add_subdirectory("examples/cuda/viewcopy")
	endif()

	# SYCL examples
	find_package(IntelSYCL)
	if (IntelSYCL_FOUND)
		add_subdirectory("examples/sycl/nbody")
	endif()
endif()

# install
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(_llama_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/llama")

configure_package_config_file (
	"${PROJECT_SOURCE_DIR}/cmake/llama-config.cmake.in"
	"${PROJECT_BINARY_DIR}/cmake/llama-config.cmake"
	INSTALL_DESTINATION "${_llama_INSTALL_CMAKEDIR}")

configure_file (
	"${PROJECT_SOURCE_DIR}/cmake/llama-config-version.cmake.in"
	"${PROJECT_BINARY_DIR}/cmake/llama-config-version.cmake"
	@ONLY
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/llama" DESTINATION "include" )
install(
	FILES
		"${PROJECT_BINARY_DIR}/cmake/llama-config.cmake"
		"${PROJECT_BINARY_DIR}/cmake/llama-config-version.cmake"
	DESTINATION
		"${_llama_INSTALL_CMAKEDIR}"
)

cmake_minimum_required (VERSION 3.3)
project(llama-alpaka-pic)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package(fmt REQUIRED)
find_package(OpenMP REQUIRED)
find_package(alpaka 0.7.0 REQUIRED)
alpaka_add_executable(${PROJECT_NAME} pic.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_link_libraries(${PROJECT_NAME} PRIVATE llama::llama fmt::fmt alpaka::alpaka OpenMP::OpenMP_CXX)

if (MSVC)
	target_compile_options(${PROJECT_NAME} PRIVATE
		/arch:AVX2
		/fp:fast
	)
elseif()
	target_compile_options(${PROJECT_NAME} PRIVATE
		#-fno-math-errno # sqrt prevents vectorization otherwise
		-march=native
		-ffast-math
	)
endif()

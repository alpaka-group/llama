#pragma once

// ============================================================================
// == ./include/llama/View.hpp ==
// ==
// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// #pragma once
	// ============================================================================
	// == ./include/llama/Accessors.hpp ==
	// ==
	// Copyright 2023 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
		// ============================================================================
		// == ./include/llama/Concepts.hpp ==
		// ==
		// Copyright 2022 Bernhard Manfred Gruber
		// SPDX-License-Identifier: MPL-2.0

		// #pragma once
			// ============================================================================
			// == ./include/llama/Array.hpp ==
			// ==
			// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
			// SPDX-License-Identifier: MPL-2.0

			// #pragma once
				// ============================================================================
				// == ./include/llama/macros.hpp ==
				// ==
				// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
				// SPDX-License-Identifier: MPL-2.0

				// #pragma once
				#ifdef __INTEL_COMPILER
				#    error LLAMA has stopped supporting the Intel Classic Compiler after Intel announced its planned deprecation and \
				 replacement by the Intel LLVM-based compiler. Please migrate to the Intel LLVM-based compiler.
				#endif

				#if defined(__INTEL_LLVM_COMPILER)
				// icx supports #pragma ivdep, but it will issue a diagnostic that it needs vectorize(assume_safety) to vectorize.
				// Let's keep both pragmas for now.
				#    define LLAMA_INDEPENDENT_DATA                                                                                    \
				        _Pragma("ivdep") _Pragma("clang loop vectorize(assume_safety) interleave(assume_safety)")
				#elif defined(__clang__)
				#    define LLAMA_INDEPENDENT_DATA _Pragma("clang loop vectorize(assume_safety) interleave(assume_safety)")
				#elif defined(__NVCOMPILER)
				#    define LLAMA_INDEPENDENT_DATA _Pragma("ivdep")
				#elif defined(__GNUC__)
				#    define LLAMA_INDEPENDENT_DATA _Pragma("GCC ivdep")
				#elif defined(_MSC_VER)
				#    define LLAMA_INDEPENDENT_DATA __pragma(loop(ivdep))
				#else
				/// May be put in front of a loop statement. Indicates that all (!) data access inside the loop is indepent, so the
				/// loop can be safely vectorized. Example: \code{.cpp}
				///     LLAMA_INDEPENDENT_DATA
				///     for(int i = 0; i < N; ++i)
				///         // because of LLAMA_INDEPENDENT_DATA the compiler knows that a and b
				///         // do not overlap and the operation can safely be vectorized
				///         a[i] += b[i];
				/// \endcode
				#    define LLAMA_INDEPENDENT_DATA
				#endif

				#ifndef LLAMA_FORCE_INLINE
				#    if defined(__NVCC__) || defined(__HIP__)
				#        define LLAMA_FORCE_INLINE __forceinline__
				#    elif defined(__GNUC__) || defined(__clang__)
				#        define LLAMA_FORCE_INLINE inline __attribute__((always_inline))
				#    elif defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
				#        define LLAMA_FORCE_INLINE __forceinline
				#    else
				/// Forces the compiler to inline a function annotated with this macro
				#        define LLAMA_FORCE_INLINE inline
				#        warning LLAMA_FORCE_INLINE is only defined to "inline" for this compiler
				#    endif
				#endif

				#ifndef LLAMA_PRAGMA
				#    define LLAMA_PRAGMA(tokens) _Pragma(#tokens)
				#endif

				#ifndef LLAMA_UNROLL
				#    if defined(__HIP__) || defined(__NVCC__) || defined(__NVCOMPILER) || defined(__clang__)                          \
				        || defined(__INTEL_LLVM_COMPILER)
				#        define LLAMA_UNROLL(...) LLAMA_PRAGMA(unroll __VA_ARGS__)
				#    elif defined(__GNUG__)
				#        define LLAMA_UNROLL(...) LLAMA_PRAGMA(GCC unroll __VA_ARGS__)
				#    elif defined(_MSC_VER)
				// MSVC does not support a pragma for unrolling
				#        define LLAMA_UNROLL(...)
				#    else
				/// Requests the compiler to unroll the loop following this directive. An optional unrolling count may be provided as
				/// argument, which must be a constant expression.
				#        define LLAMA_UNROLL(...)
				#        warning LLAMA_UNROLL is not implemented for your compiler
				#    endif
				#endif

				#ifndef LLAMA_ACC
				#    if defined(__HIP__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
				#        define LLAMA_ACC __device__
				#    elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
				#        define LLAMA_ACC
				#    else
				#        define LLAMA_ACC
				#        warning LLAMA_HOST_ACC is only defined empty for this compiler
				#    endif
				#endif

				#ifndef LLAMA_HOST_ACC
				#    if defined(__HIP__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
				#        define LLAMA_HOST_ACC __host__ __device__
				#    elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
				#        define LLAMA_HOST_ACC
				#    else
				/// Some offloading parallelization language extensions such a CUDA, OpenACC or OpenMP 4.5 need to specify whether a
				/// class, struct, function or method "resides" on the host, the accelerator (the offloading device) or both. LLAMA
				/// supports this with marking every function needed on an accelerator with `LLAMA_HOST_ACC`.
				#        define LLAMA_HOST_ACC
				#        warning LLAMA_HOST_ACC is only defined empty for this compiler
				#    endif
				#endif

				#define LLAMA_FN_HOST_ACC_INLINE LLAMA_FORCE_INLINE LLAMA_HOST_ACC

				#ifndef LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS
				#    if defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
				#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __attribute__((always_inline)) __VA_ARGS__
				#    elif defined(__GNUC__) || (defined(__NVCC__) && !defined(_MSC_VER))
				#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__ __attribute__((always_inline))
				#    elif defined(_MSC_VER)
				#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...)                                                              \
				            __VA_ARGS__ /* FIXME: MSVC cannot combine constexpr and [[msvc::forceinline]] */
				#    else
				#        define LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(...) __VA_ARGS__
				#        warning LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS not defined for this compiler
				#    endif
				#endif
				#ifndef LLAMA_LAMBDA_INLINE
				/// Gives strong indication to the compiler to inline the attributed lambda.
				#    define LLAMA_LAMBDA_INLINE LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS()
				#endif

				#ifndef LLAMA_SUPPRESS_HOST_DEVICE_WARNING
				#    if defined(__NVCC__) && !defined(__clang__)
				#        define LLAMA_SUPPRESS_HOST_DEVICE_WARNING _Pragma("nv_exec_check_disable")
				#    else
				/// Suppresses the nvcc warning: 'calling a __host__ function from __host__ __device__ function.'
				/// This macro can be applied to function declarations
				#        define LLAMA_SUPPRESS_HOST_DEVICE_WARNING
				#    endif
				#endif

				#ifndef LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
				#    ifdef __NVCC__
				#        ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
				#            define LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING                                                          \
				                _Pragma("nv_diag_suppress 20011") _Pragma("nv_diag_suppress 20014")
				#        else
				#            define LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING                                                          \
				                _Pragma("diag_suppress 20011") _Pragma("diag_suppress 20014")
				#        endif
				#    else
				/// Suppresses the nvcc warnings:
				/// 'calling a __host__ function from a __host__ __device__ function is not allowed' and
				/// 'calling a __host__ function("...") from a __host__ __device__ function("...") is not allowed'
				/// This macro can be applied before the concerned code block, which then needs to be ended with \ref
				/// LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING.
				#        define LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
				#    endif
				#endif
				#ifndef LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
				#    ifdef __NVCC__
				#        ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
				#            define LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING                                                            \
				                _Pragma("nv_diag_default 20011") _Pragma("nv_diag_default 20014")
				#        else
				#            define LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING _Pragma("diag_default 20011") _Pragma("diag_default 20014")
				#        endif
				#    else
				#        define LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
				#    endif
				#endif

				/// Forces a copy of a value. This is useful to prevent ODR usage of constants when compiling for GPU targets.
				#define LLAMA_COPY(x) decltype(x)(x)

				// https://devblogs.microsoft.com/cppblog/optimizing-the-layout-of-empty-base-classes-in-vs2015-update-2-3/
				#if defined(_MSC_VER)
				#    define LLAMA_DECLSPEC_EMPTY_BASES __declspec(empty_bases)
				#else
				#    define LLAMA_DECLSPEC_EMPTY_BASES
				#endif

				/// Expands to likely if [[likely]] supported by the compiler. Use as [[LLAMA_LIKELY]].
				#if __has_cpp_attribute(likely)
				#    define LLAMA_LIKELY likely
				#else
				#    define LLAMA_LIKELY
				#endif

				/// Expands to unlikely if [[unlikely]] supported by the compiler. Use as [[LLAMA_UNLIKELY]].
				#if __has_cpp_attribute(unlikely)
				#    define LLAMA_UNLIKELY unlikely
				#else
				#    define LLAMA_UNLIKELY
				#endif

				/// Expands to consteval if the compilers supports the keyword, otherwise to constexpr.
				// TODO(bgruber): reevalute with nvhpc in the future or find workaround
				#if defined(__cpp_consteval) && !defined(__NVCOMPILER)
				#    define LLAMA_CONSTEVAL consteval
				#else
				#    define LLAMA_CONSTEVAL constexpr
				#endif

				#ifndef LLAMA_EXPORT
				/// Annotation of all LLAMA public APIs. Expands to nothing by default. Can be defined to 'export' when building LLAMA
				/// as a C++20 module.
				#    define LLAMA_EXPORT
				#endif

				// TODO(bgruber): clang 12-15 (libstdc++ from gcc 11.2 or gcc 12.1) fail to compile this currently with the issue
				// described here:
				// https://stackoverflow.com/questions/64300832/why-does-clang-think-gccs-subrange-does-not-satisfy-gccs-ranges-begin-functi
				// Intel LLVM compiler is also using the clang frontend
				#define CAN_USE_RANGES 0
				#if __has_include(<version>)
				#    include <version>
				#    if defined(__cpp_concepts) && defined(__cpp_lib_ranges) && (!defined(__clang__) || __clang_major__ >= 16)        \
				        && !defined(__INTEL_LLVM_COMPILER) && (!defined(_MSC_VER) || _MSC_VER > 1932) && !defined(__NVCOMPILER)
				#        undef CAN_USE_RANGES
				#        define CAN_USE_RANGES 1
				#    endif
				#endif
				// ==
				// == ./include/llama/macros.hpp ==
				// ============================================================================


			#include <ostream>
			#include <stdexcept>
			#include <tuple>

			namespace llama
			{
			    /// Array class like `std::array` but suitable for use with offloading devices like GPUs.
			    /// \tparam T type if array elements.
			    /// \tparam N rank of the array.
			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    // NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp,readability-identifier-naming)
			    struct Array
			    {
			        using value_type = T;
			        T element[N];

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto size() const
			        {
			            return N;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto empty() const -> bool
			        {
			            return N == 0;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto begin() -> T*
			        {
			            return &element[0];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto begin() const -> const T*
			        {
			            return &element[0];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto end() -> T*
			        {
			            return &element[0] + N;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto end() const -> const T*
			        {
			            return &element[0] + N;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto front() -> T&
			        {
			            return element[0];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto front() const -> const T&
			        {
			            return element[0];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto back() -> T&
			        {
			            return element[N - 1];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto back() const -> const T&
			        {
			            return element[N - 1];
			        }

			        template<typename IndexType>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&& idx) -> T&
			        {
			            return element[idx];
			        }

			        template<typename IndexType>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&& idx) const -> const T&
			        {
			            return element[idx];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto data() -> T*
			        {
			            return &element[0];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto data() const -> const T*
			        {
			            return &element[0];
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator==(const Array& a, const Array& b) -> bool
			        {
			            for(std::size_t i = 0; i < N; ++i)
			                if(a.element[i] != b.element[i])
			                    return false;
			            return true;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator!=(const Array& a, const Array& b) -> bool
			        {
			            return !(a == b);
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator+(const Array& a, const Array& b) -> Array
			        {
			            Array temp{};
			            for(std::size_t i = 0; i < N; ++i)
			                temp[i] = a[i] + b[i];
			            return temp;
			        }

			        template<std::size_t I>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto get() -> T&
			        {
			            return element[I];
			        }

			        template<std::size_t I>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto get() const -> const T&
			        {
			            return element[I];
			        }
			    };

			    LLAMA_EXPORT
			    template<typename T>
			    struct Array<T, 0>
			    {
			        using value_type = T;

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto size() const
			        {
			            return 0;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto empty() const -> bool
			        {
			            return true;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto begin() -> T*
			        {
			            return nullptr;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto begin() const -> const T*
			        {
			            return nullptr;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto end() -> T*
			        {
			            return nullptr;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto end() const -> const T*
			        {
			            return nullptr;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto front() -> T&
			        {
			            outOfRange();
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto front() const -> const T&
			        {
			            outOfRange();
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto back() -> T&
			        {
			            outOfRange();
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto back() const -> const T&
			        {
			            outOfRange();
			        }

			        template<typename IndexType>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&&) -> T&
			        {
			            outOfRange();
			        }

			        template<typename IndexType>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&&) const -> const T&
			        {
			            outOfRange();
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto data() -> T*
			        {
			            return nullptr;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr auto data() const -> const T*
			        {
			            return nullptr;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator==(const Array&, const Array&) -> bool
			        {
			            return true;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator!=(const Array&, const Array&) -> bool
			        {
			            return false;
			        }

			        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator+(const Array&, const Array&) -> Array
			        {
			            return {};
			        }

			        template<std::size_t I>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto get() -> T&
			        {
			            outOfRange();
			        }

			        template<std::size_t I>
			        LLAMA_FN_HOST_ACC_INLINE constexpr auto get() const -> const T&
			        {
			            outOfRange();
			        }

			    private:
			        [[noreturn]] void outOfRange() const
			        {
			            throw std::out_of_range{"Array has zero length"};
			        }
			    };

			    LLAMA_EXPORT
			    template<typename First, typename... Args>
			    Array(First, Args... args) -> Array<First, sizeof...(Args) + 1>;

			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    auto operator<<(std::ostream& os, const Array<T, N>& a) -> std::ostream&
			    {
			        os << "Array{";
			        bool first = true;
			        for(auto e : a)
			        {
			            if(first)
			                first = false;
			            else
			                os << ", ";
			            os << e;
			        }
			        os << "}";
			        return os;
			    }

			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    LLAMA_FN_HOST_ACC_INLINE constexpr auto pushFront([[maybe_unused]] Array<T, N> a, T v) -> Array<T, N + 1>
			    {
			        Array<T, N + 1> r{};
			        r[0] = v;
			        if constexpr(N > 0)
			            for(std::size_t i = 0; i < N; i++)
			                r[i + 1] = a[i];
			        return r;
			    }

			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    LLAMA_FN_HOST_ACC_INLINE constexpr auto pushBack([[maybe_unused]] Array<T, N> a, T v) -> Array<T, N + 1>
			    {
			        Array<T, N + 1> r{};
			        if constexpr(N > 0)
			            for(std::size_t i = 0; i < N; i++)
			                r[i] = a[i];
			        r[N] = v;
			        return r;
			    }

			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    LLAMA_FN_HOST_ACC_INLINE constexpr auto popBack([[maybe_unused]] Array<T, N> a)
			    {
			        static_assert(N > 0);
			        Array<T, N - 1> r{};
			        if constexpr(N > 1)
			            for(std::size_t i = 0; i < N - 1; i++)
			                r[i] = a[i];
			        return r;
			    }

			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    LLAMA_FN_HOST_ACC_INLINE constexpr auto popFront([[maybe_unused]] Array<T, N> a)
			    {
			        static_assert(N > 0);
			        Array<T, N - 1> r{};
			        if constexpr(N > 1)
			            for(std::size_t i = 0; i < N - 1; i++)
			                r[i] = a[i + 1];
			        return r;
			    }

			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    LLAMA_FN_HOST_ACC_INLINE constexpr auto product(Array<T, N> a) -> T
			    {
			        T prod = 1;
			        for(auto s : a)
			            prod *= s;
			        return prod;
			    }

			    LLAMA_EXPORT
			    template<typename T, std::size_t N>
			    LLAMA_FN_HOST_ACC_INLINE constexpr auto dot([[maybe_unused]] Array<T, N> a, [[maybe_unused]] Array<T, N> b) -> T
			    {
			        T r = 0;
			        if constexpr(N > 0)
			            for(std::size_t i = 0; i < N; i++)
			                r += a[i] * b[i];
			        return r;
			    }
			} // namespace llama

			LLAMA_EXPORT
			template<typename T, size_t N>
			struct std::tuple_size<llama::Array<T, N>> : std::integral_constant<size_t, N> // NOLINT(cert-dcl58-cpp)
			{
			};

			LLAMA_EXPORT
			template<size_t I, typename T, size_t N>
			struct std::tuple_element<I, llama::Array<T, N>> // NOLINT(cert-dcl58-cpp)
			{
			    using type = T;
			};
			// ==
			// == ./include/llama/Array.hpp ==
			// ============================================================================

			// ============================================================================
			// == ./include/llama/Core.hpp ==
			// ==
			// Copyright 2023 Alexander Matthes, Bernhard Manfred Gruber
			// SPDX-License-Identifier: MPL-2.0

			// #pragma once
				// ============================================================================
				// == ./include/llama/ArrayExtents.hpp ==
				// ==
				// Copyright 2022 Bernhard Manfred Gruber
				// SPDX-License-Identifier: MPL-2.0

				// #pragma once
				// #include "Array.hpp"    // amalgamate: file already inlined
					// ============================================================================
					// == ./include/llama/Meta.hpp ==
					// ==
					// Copyright 2022 Bernhard Manfred Gruber
					// SPDX-License-Identifier: MPL-2.0

					// #pragma once
					// #include "macros.hpp"    // amalgamate: file already inlined

					#include <boost/mp11.hpp>

					namespace llama
					{
					    // make mp11 directly available in the llama namespace
					    using namespace boost::mp11;

					    namespace internal
					    {
					        // adapted from boost::mp11, but with LLAMA_FN_HOST_ACC_INLINE
					        template<template<typename...> typename L, typename... T, typename F>
					        LLAMA_FN_HOST_ACC_INLINE constexpr void mpForEachInlined(L<T...>, F&& f)
					        {
					            using A = int[sizeof...(T)];
					            (void) A{((void) f(T{}), 0)...};
					        }

					        template<typename FromList, template<auto...> class ToList>
					        struct mp_unwrap_values_into_impl;

					        template<template<class...> class FromList, typename... Values, template<auto...> class ToList>
					        struct mp_unwrap_values_into_impl<FromList<Values...>, ToList>
					        {
					            using type = ToList<Values::value...>;
					        };

					        template<typename FromList, template<auto...> class ToList>
					        using mp_unwrap_values_into = typename mp_unwrap_values_into_impl<FromList, ToList>::type;

					        template<typename E, typename... Args>
					        struct ReplacePlaceholdersImpl
					        {
					            using type = E;
					        };
					        template<std::size_t I, typename... Args>
					        struct ReplacePlaceholdersImpl<mp_arg<I>, Args...>
					        {
					            using type = mp_at_c<mp_list<Args...>, I>;
					        };

					        template<template<typename...> typename E, typename... Ts, typename... Args>
					        struct ReplacePlaceholdersImpl<E<Ts...>, Args...>
					        {
					            using type = E<typename ReplacePlaceholdersImpl<Ts, Args...>::type...>;
					        };
					    } // namespace internal

					    LLAMA_EXPORT
					    template<typename Expression, typename... Args>
					    using ReplacePlaceholders = typename internal::ReplacePlaceholdersImpl<Expression, Args...>::type;
					} // namespace llama
					// ==
					// == ./include/llama/Meta.hpp ==
					// ============================================================================


				#include <limits>
				#include <type_traits>

				namespace llama
				{
				    // TODO(bgruber): make this an alias in C++20, when we have CTAD for aliases
				    /// Represents a run-time index into the array dimensions.
				    /// \tparam Dim Compile-time number of dimensions.
				    LLAMA_EXPORT
				    template<typename T, std::size_t Dim>
				    struct ArrayIndex : Array<T, Dim>
				    {
				        static constexpr std::size_t rank = Dim;
				    };

				    // allow comparing ArrayIndex with different size types:
				    LLAMA_EXPORT
				    template<std::size_t Dim, typename TA, typename TB>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(ArrayIndex<TA, Dim> a, ArrayIndex<TB, Dim> b) -> bool
				    {
				        for(std::size_t i = 0; i < Dim; ++i)
				            if(a[i] != b[i])
				                return false;
				        return true;
				    }

				    LLAMA_EXPORT
				    template<std::size_t Dim, typename TA, typename TB>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(ArrayIndex<TA, Dim> a, ArrayIndex<TB, Dim> b) -> bool
				    {
				        return !(a == b);
				    }

				    static_assert(
				        std::is_trivially_default_constructible_v<ArrayIndex<int, 1>>); // so ArrayIndex<1>{} will produce a zeroed
				                                                                        // index. Should hold for all dimensions,
				                                                                        // but just checking for <1> here.
				    static_assert(std::is_trivially_copy_constructible_v<ArrayIndex<int, 1>>);
				    static_assert(std::is_trivially_move_constructible_v<ArrayIndex<int, 1>>);
				    static_assert(std::is_trivially_copy_assignable_v<ArrayIndex<int, 1>>);
				    static_assert(std::is_trivially_move_assignable_v<ArrayIndex<int, 1>>);

				    namespace internal
				    {
				        template<typename Default, typename... Ints>
				        struct IndexTypeFromArgs
				        {
				            using type = Default;
				        };

				        template<typename Default, typename FirstInt, typename... Ints>
				        struct IndexTypeFromArgs<Default, FirstInt, Ints...>
				        {
				            static_assert(std::conjunction_v<std::is_same<FirstInt, Ints>...>, "All index types must be the same");
				            using type = FirstInt;
				        };
				    } // namespace internal

				    LLAMA_EXPORT
				    template<typename... Args>
				    ArrayIndex(Args...)
				        -> ArrayIndex<typename internal::IndexTypeFromArgs<std::size_t, Args...>::type, sizeof...(Args)>;
				} // namespace llama

				LLAMA_EXPORT
				template<typename V, size_t N>
				struct std::tuple_size<llama::ArrayIndex<V, N>> : std::integral_constant<size_t, N> // NOLINT(cert-dcl58-cpp)
				{
				};

				LLAMA_EXPORT
				template<size_t I, typename V, size_t N>
				struct std::tuple_element<I, llama::ArrayIndex<V, N>> // NOLINT(cert-dcl58-cpp)
				{
				    using type = V;
				};

				namespace llama
				{
				    namespace internal
				    {
				        struct Dyn
				        {
				            template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
				            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
				            LLAMA_FN_HOST_ACC_INLINE constexpr operator T() const
				            {
				                return static_cast<T>(-1);
				            }

				            template<typename T>
				            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator==(T i, Dyn) -> bool
				            {
				                return i == static_cast<T>(-1);
				            }

				            template<typename T>
				            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator==(Dyn d, T i) -> bool
				            {
				                return i == d;
				            }

				            template<typename T>
				            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator!=(T i, Dyn d) -> bool
				            {
				                return !(i == d);
				            }

				            template<typename T>
				            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator!=(Dyn d, T i) -> bool
				            {
				                return !(i == d);
				            }
				        };
				    } // namespace internal

				    LLAMA_EXPORT
				    /// Used as a template argument to \ref ArrayExtents to mark a dynamic extent.
				    inline constexpr auto dyn = internal::Dyn{};

				    /// ArrayExtents holding compile and runtime indices. This is conceptually equivalent to the std::extent of
				    /// std::mdspan (@see: https://wg21.link/P0009) including the changes to make the size_type controllable (@see:
				    /// https://wg21.link/P2553).
				    LLAMA_EXPORT
				    template<typename T = std::size_t, T... Sizes>
				    struct ArrayExtents : Array<T, ((Sizes == dyn) + ... + 0)>
				    {
				        static constexpr std::size_t rank = sizeof...(Sizes);
				        static constexpr auto rankDynamic = ((Sizes == dyn) + ... + 0);
				        static constexpr auto rankStatic = rank - rankDynamic;

				        using Index = ArrayIndex<T, rank>;
				        using value_type = T;

				        template<std::size_t I>
				        LLAMA_FN_HOST_ACC_INLINE constexpr auto get() const -> value_type
				        {
				            using TypeList = mp_list_c<T, Sizes...>;
				            constexpr auto extent = mp_at_c<TypeList, I>::value;
				            if constexpr(extent != dyn)
				                return extent;
				            else
				                return static_cast<const Array<value_type, rankDynamic>&>(
				                    *this)[+mp_count<mp_take_c<TypeList, I>, std::integral_constant<T, dyn>>::value];
				        }

				        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](T i) const -> value_type
				        {
				            return mp_with_index<rank>(i, [&](auto ic) LLAMA_LAMBDA_INLINE { return get<decltype(ic)::value>(); });
				        }

				    private:
				        template<std::size_t... Is>
				        LLAMA_FN_HOST_ACC_INLINE constexpr auto toArray(std::index_sequence<Is...>) const -> Index
				        {
				            return {get<Is>()...};
				        }

				    public:
				        LLAMA_FN_HOST_ACC_INLINE constexpr auto toArray() const -> Index
				        {
				            return toArray(std::make_index_sequence<rank>{});
				        }
				    };

				    LLAMA_EXPORT
				    template<typename T>
				    struct ArrayExtents<T>
				    {
				        static constexpr std::size_t rank = 0;
				        static constexpr auto rankDynamic = 0;
				        static constexpr auto rankStatic = 0;

				        using Index = ArrayIndex<T, 0>;
				        using value_type = T;

				        LLAMA_FN_HOST_ACC_INLINE constexpr auto toArray() const -> Index
				        {
				            return {};
				        }
				    };

				    LLAMA_EXPORT
				    template<typename... Args>
				    ArrayExtents(Args...) -> ArrayExtents<
				        typename internal::IndexTypeFromArgs<std::size_t, Args...>::type,
				        (Args{}, dyn)...>; // we just use Args to repeat dyn as often as Args is big

				    static_assert(std::is_trivially_default_constructible_v<ArrayExtents<std::size_t, 1>>);
				    static_assert(std::is_trivially_copy_constructible_v<ArrayExtents<std::size_t, 1>>);
				    static_assert(std::is_trivially_move_constructible_v<ArrayExtents<std::size_t, 1>>);
				    static_assert(std::is_trivially_copy_assignable_v<ArrayExtents<std::size_t, 1>>);
				    static_assert(std::is_trivially_move_assignable_v<ArrayExtents<std::size_t, 1>>);
				    static_assert(std::is_empty_v<ArrayExtents<std::size_t, 1>>);

				    LLAMA_EXPORT
				    template<typename SizeTypeA, SizeTypeA... SizesA, typename SizeTypeB, SizeTypeB... SizesB>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(
				        ArrayExtents<SizeTypeA, SizesA...> a,
				        ArrayExtents<SizeTypeB, SizesB...> b) -> bool
				    {
				        return a.toArray() == b.toArray();
				    }

				    LLAMA_EXPORT
				    template<typename SizeTypeA, SizeTypeA... SizesA, typename SizeTypeB, SizeTypeB... SizesB>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(
				        ArrayExtents<SizeTypeA, SizesA...> a,
				        ArrayExtents<SizeTypeB, SizesB...> b) -> bool
				    {
				        return !(a == b);
				    }

				    LLAMA_EXPORT
				    template<typename SizeType, SizeType... Sizes>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto product(ArrayExtents<SizeType, Sizes...> e) -> SizeType
				    {
				        return product(e.toArray());
				    }

				    namespace internal
				    {
				        template<typename SizeType, SizeType Extent, std::size_t... Is>
				        constexpr auto makeArrayExtents(std::index_sequence<Is...>)
				        {
				            return ArrayExtents<SizeType, (static_cast<void>(Is), Extent)...>{};
				        }
				    } // namespace internal

				    LLAMA_EXPORT
				    /// N-dimensional ArrayExtents where all N extents are Extent.
				    template<typename SizeType, std::size_t N, SizeType Extent>
				    using ArrayExtentsNCube = decltype(internal::makeArrayExtents<SizeType, Extent>(std::make_index_sequence<N>{}));

				    LLAMA_EXPORT
				    /// N-dimensional ArrayExtents where all values are dynamic.
				    template<typename SizeType, std::size_t N>
				    using ArrayExtentsDynamic = ArrayExtentsNCube<SizeType, N, dyn>;

				    LLAMA_EXPORT
				    template<typename SizeType, std::size_t Dim, typename Func, typename... OuterIndices>
				    LLAMA_FN_HOST_ACC_INLINE void forEachArrayIndex(
				        [[maybe_unused]] const ArrayIndex<SizeType, Dim>& extents,
				        Func&& func,
				        OuterIndices... outerIndices)
				    {
				        constexpr auto fixedIndices = sizeof...(outerIndices);
				        LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
				        if constexpr(fixedIndices < Dim)
				        {
				            for(SizeType i = 0; i < extents[fixedIndices]; i++)
				                forEachArrayIndex(extents, std::forward<Func>(func), outerIndices..., i);
				        }
				        else
				        {
				            std::forward<Func>(func)(ArrayIndex<SizeType, fixedIndices>{outerIndices...});
				        }
				        LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
				    }

				    LLAMA_EXPORT
				    template<typename SizeType, SizeType... Sizes, typename Func>
				    LLAMA_FN_HOST_ACC_INLINE void forEachArrayIndex(ArrayExtents<SizeType, Sizes...> extents, Func&& func)
				    {
				        forEachArrayIndex(extents.toArray(), std::forward<Func>(func));
				    }
				} // namespace llama

				LLAMA_EXPORT
				template<typename SizeType, SizeType... Sizes>
				struct std::tuple_size<llama::ArrayExtents<SizeType, Sizes...>> // NOLINT(cert-dcl58-cpp)
				    : std::integral_constant<std::size_t, sizeof...(Sizes)>
				{
				};

				LLAMA_EXPORT
				template<typename SizeType, std::size_t I, SizeType... Sizes>
				struct std::tuple_element<I, llama::ArrayExtents<SizeType, Sizes...>> // NOLINT(cert-dcl58-cpp)
				{
				    using type = SizeType;
				};
				// ==
				// == ./include/llama/ArrayExtents.hpp ==
				// ============================================================================

			// #include "Meta.hpp"    // amalgamate: file already inlined
				// ============================================================================
				// == ./include/llama/RecordCoord.hpp ==
				// ==
				// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
				// SPDX-License-Identifier: MPL-2.0

				// #pragma once
				// #include "Meta.hpp"    // amalgamate: file already inlined
				// #include "macros.hpp"    // amalgamate: file already inlined

				#include <array>
				// #include <ostream>    // amalgamate: file already included
				// #include <type_traits>    // amalgamate: file already included

				namespace llama
				{
				    /// Represents a coordinate for a record inside the record dimension tree.
				    /// \tparam Coords... the compile time coordinate.
				    LLAMA_EXPORT
				    template<std::size_t... Coords>
				    struct RecordCoord
				    {
				        /// The list of integral coordinates as `mp_list`.
				        using List = mp_list_c<std::size_t, Coords...>;

				        static constexpr std::size_t front = mp_front<List>::value;
				        static constexpr std::size_t back = mp_back<List>::value;
				        static constexpr std::size_t size = sizeof...(Coords);
				    };

				    LLAMA_EXPORT
				    template<>
				    struct RecordCoord<>
				    {
				        using List = mp_list_c<std::size_t>;

				        static constexpr std::size_t size = 0;
				    };

				    LLAMA_EXPORT
				    template<std::size_t... CoordsA, std::size_t... CoordsB>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<CoordsA...>, RecordCoord<CoordsB...>)
				    {
				        return false;
				    }

				    LLAMA_EXPORT
				    template<std::size_t... Coords>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<Coords...>, RecordCoord<Coords...>)
				    {
				        return true;
				    }

				    LLAMA_EXPORT
				    template<std::size_t... CoordsA, std::size_t... CoordsB>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(RecordCoord<CoordsA...> a, RecordCoord<CoordsB...> b)
				    {
				        return !(a == b);
				    }

				    LLAMA_EXPORT
				    template<typename T>
				    inline constexpr bool isRecordCoord = false;

				    LLAMA_EXPORT
				    template<std::size_t... Coords>
				    inline constexpr bool isRecordCoord<RecordCoord<Coords...>> = true;

				    LLAMA_EXPORT
				    template<std::size_t... RCs>
				    auto operator<<(std::ostream& os, RecordCoord<RCs...>) -> std::ostream&
				    {
				        os << "RecordCoord<";
				        bool first = true;
				        for(auto rc : std::array<std::size_t, sizeof...(RCs)>{RCs...})
				        {
				            if(first)
				                first = false;
				            else
				                os << ", ";
				            os << rc;
				        }
				        os << ">";
				        return os;
				    }

				    inline namespace literals
				    {
				        /// Literal operator for converting a numeric literal into a \ref RecordCoord.
				        LLAMA_EXPORT
				        template<char... Digits>
				        constexpr auto operator"" _RC()
				        {
				            constexpr auto coord = []() constexpr
				            {
				                const char digits[] = {(Digits - 48)...};
				                std::size_t acc = 0;
				                std ::size_t powerOf10 = 1;
				                for(int i = sizeof...(Digits) - 1; i >= 0; i--)
				                {
				                    acc += digits[i] * powerOf10;
				                    powerOf10 *= 10;
				                }
				                return acc;
				            }();
				            return RecordCoord<coord>{};
				        }
				    } // namespace literals

				    /// Converts a type list of integral constants into a \ref RecordCoord.
				    LLAMA_EXPORT
				    template<typename L>
				    using RecordCoordFromList = internal::mp_unwrap_values_into<L, RecordCoord>;

				    /// Concatenate a set of \ref RecordCoord%s.
				    LLAMA_EXPORT
				    template<typename... RecordCoords>
				    using Cat = RecordCoordFromList<mp_append<typename RecordCoords::List...>>;

				    /// Concatenate a set of \ref RecordCoord%s instances.
				    LLAMA_EXPORT
				    template<typename... RecordCoords>
				    LLAMA_FN_HOST_ACC_INLINE constexpr auto cat(RecordCoords...)
				    {
				        return Cat<RecordCoords...>{};
				    }

				    /// RecordCoord without first coordinate component.
				    LLAMA_EXPORT
				    template<typename RecordCoord>
				    using PopFront = RecordCoordFromList<mp_pop_front<typename RecordCoord::List>>;

				    namespace internal
				    {
				        template<std::size_t... Coords1, std::size_t... Coords2>
				        constexpr auto recordCoordCommonPrefixIsBiggerImpl(RecordCoord<Coords1...>, RecordCoord<Coords2...>) -> bool
				        {
				            // CTAD does not work if Coords1/2 is an empty pack
				            std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
				            std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
				            for(std::size_t i = 0; i < std::min(a1.size(), a2.size()); i++)
				            {
				                if(a1[i] > a2[i])
				                    return true;
				                if(a1[i] < a2[i])
				                    return false;
				            }
				            return false;
				        };
				    } // namespace internal

				    /// Checks wether the first RecordCoord is bigger than the second.
				    LLAMA_EXPORT
				    template<typename First, typename Second>
				    inline constexpr auto recordCoordCommonPrefixIsBigger
				        = internal::recordCoordCommonPrefixIsBiggerImpl(First{}, Second{});

				    namespace internal
				    {
				        template<std::size_t... Coords1, std::size_t... Coords2>
				        constexpr auto recordCoordCommonPrefixIsSameImpl(RecordCoord<Coords1...>, RecordCoord<Coords2...>) -> bool
				        {
				            // CTAD does not work if Coords1/2 is an empty pack
				            std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
				            std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
				            for(std::size_t i = 0; i < std::min(a1.size(), a2.size()); i++)
				                if(a1[i] != a2[i])
				                    return false;
				            return true;
				        };
				    } // namespace internal

				    /// Checks whether two \ref RecordCoord%s are the same or one is the prefix of the other.
				    LLAMA_EXPORT
				    template<typename First, typename Second>
				    inline constexpr auto recordCoordCommonPrefixIsSame
				        = internal::recordCoordCommonPrefixIsSameImpl(First{}, Second{});
				} // namespace llama
				// ==
				// == ./include/llama/RecordCoord.hpp ==
				// ============================================================================


			#include <string>
			// #include <type_traits>    // amalgamate: file already included
			#if __has_include(<boost/describe/members.hpp>)
			#    include <boost/describe/members.hpp>
			#endif

			namespace llama
			{
			    /// Anonymous naming for a \ref Field.
			    LLAMA_EXPORT
			    struct NoName
			    {
			    };

			    /// @brief Tells whether the given type is allowed as a field type in LLAMA. Such types need to be trivially
			    /// constructible and trivially destructible.
			    LLAMA_EXPORT
			    template<typename T>
			    inline constexpr bool isAllowedFieldType = std::is_trivially_destructible_v<T>;

			    /// Record dimension tree node which may either be a leaf or refer to a child tree presented as another \ref
			    /// Record.
			    /// \tparam Tag Name of the node. May be any type (struct, class).
			    /// \tparam Type Type of the node. May be one of three cases. 1. another sub tree consisting of a nested \ref
			    /// Record. 2. an array of static size of any type, in which case a Record with as many \ref Field as the array
			    /// size is created, named \ref RecordCoord specialized on consecutive numbers I. 3. A scalar type different from
			    /// \ref Record, making this node a leaf of this type.
			    LLAMA_EXPORT
			    template<typename Tag, typename Type>
			    struct Field
			    {
			        static_assert(isAllowedFieldType<Type>, "This field's type is not allowed");
			    };

			    LLAMA_EXPORT
			    template<typename T>
			    inline constexpr bool isField = false;

			    LLAMA_EXPORT
			    template<typename Tag, typename Type>
			    inline constexpr bool isField<Field<Tag, Type>> = true;

			    /// A type list of \ref Field%s which may be used to define a record dimension.
			    LLAMA_EXPORT
			    template<typename... Fields>
			#if __cpp_concepts
			    // Cannot use a fold expression here, because clang/nvcc/icx cannot handle more than 256 arguments.
			    // If you get an error here, then you passed a type which is not llama::Field as argument to Record
			        requires(mp_all<mp_bool<isField<Fields>>...>::value)
			#endif
			    struct Record
			    {
			    };

			#if __cpp_nontype_template_args >= 201911L && !defined(__EDG__)
			    /// Defined when string literals are supported in the record dimension. See also \ref NamedField.
			#    define LLAMA_HAS_STRING_FIELDS

			    namespace internal
			    {
			        // N includes the char to store the null-terminator
			        template<std::size_t N>
			        struct FixedString
			        {
			            constexpr FixedString(const char* str)
			            {
			                std::copy(str, str + N, data);
			            }

			            char data[N];
			        };

			        template<std::size_t N>
			        FixedString(const char (&str)[N]) -> FixedString<N>;

			        template<FixedString Name>
			        struct StringTag
			        {
			        };
			    } // namespace internal

			    inline namespace literals
			    {
			        /// Literal operator for converting a string literal "abc"_Name to a StringTag<"Name">.
			        LLAMA_EXPORT
			        template<internal::FixedString Name>
			        auto operator"" _Name()
			        {
			            return internal::StringTag<Name>{};
			        }
			    } // namespace literals

			    /// Alternative to \ref Field. Use with string literals, e.g. NamedField<"x", float>. Access at the \ref View
			    /// requires to use "x"_Name then.
			    LLAMA_EXPORT
			    template<internal::FixedString Tag, typename Type>
			    using NamedField = Field<internal::StringTag<Tag>, Type>;

			#    if __has_include(<boost/describe/members.hpp>)
			    /// Defined when LLAMA has support to reflect a C++ struct into a record dimension using Boost.Describe.
			#        define LLAMA_CAN_REFLECT_RECORD_DIM
			    namespace internal
			    {
			        template<typename T>
			        auto reflectToRecordDim();

			        template<class C, typename T>
			        auto memberPointerPointeeType(T C::*) -> T;

			        constexpr auto constexpr_strlen(const char* s)
			        {
			            const char* end = s;
			            while(*end != 0)
			                end++;
			            return end - s;
			        }

			        template<typename Member>
			        using MakeFieldFromMemberDescriptor = NamedField<
			            FixedString<constexpr_strlen(Member::name) + 1>(Member::name),
			            decltype(reflectToRecordDim<decltype(memberPointerPointeeType(Member::pointer))>())>;

			        template<typename T>
			        auto reflectToRecordDim()
			        {
			            if constexpr(boost::describe::has_describe_members<T>::value)
			            {
			                using MemberList = boost::describe::describe_members<T, boost::describe::mod_public>;
			                return mp_rename<mp_transform<MakeFieldFromMemberDescriptor, MemberList>, llama::Record>{};
			            }
			            else
			                return T{};
			        }
			    } // namespace internal

			    /// Reflects the given type T using Boost.Describe and creates a record dimension for it.
			    LLAMA_EXPORT
			    template<typename T>
			    using ReflectToRecordDim = decltype(internal::reflectToRecordDim<T>());
			#    endif
			#endif

			    LLAMA_EXPORT
			    template<typename T>
			    struct NrAndOffset
			    {
			        T nr;
			        T offset;

			        friend auto operator<<(std::ostream& os, const NrAndOffset& value) -> std::ostream&
			        {
			            return os << "NrAndOffset{" << value.nr << ", " << value.offset << "}";
			        }
			    };

			    LLAMA_EXPORT
			    template<typename Int>
			    NrAndOffset(Int, Int) -> NrAndOffset<Int>;

			    LLAMA_EXPORT
			    template<typename TA, typename TB>
			    auto operator==(const NrAndOffset<TA>& a, const NrAndOffset<TB>& b) -> bool
			    {
			        return a.nr == b.nr && a.offset == b.offset;
			    }

			    LLAMA_EXPORT
			    template<typename TA, typename TB>
			    auto operator!=(const NrAndOffset<TA>& a, const NrAndOffset<TB>& b) -> bool
			    {
			        return !(a == b);
			    }

			    /// Get the tag from a \ref Field.
			    LLAMA_EXPORT
			    template<typename Field>
			    using GetFieldTag = mp_first<Field>;

			    /// Get the type from a \ref Field.
			    LLAMA_EXPORT
			    template<typename Field>
			    using GetFieldType = mp_second<Field>;

			    LLAMA_EXPORT
			    template<typename T>
			    inline constexpr auto isRecord = false;

			    LLAMA_EXPORT
			    template<typename... Fields>
			    inline constexpr auto isRecord<Record<Fields...>> = true;

			    namespace internal
			    {
			        template<typename RecordDim, typename RecordCoord>
			        struct GetTagsImpl;

			        template<typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
			        struct GetTagsImpl<Record<Fields...>, RecordCoord<FirstCoord, Coords...>>
			        {
			            using Field = mp_at_c<mp_list<Fields...>, FirstCoord>;
			            using ChildTag = GetFieldTag<Field>;
			            using ChildType = GetFieldType<Field>;
			            using type = mp_push_front<typename GetTagsImpl<ChildType, RecordCoord<Coords...>>::type, ChildTag>;
			        };

			        template<typename ChildType, std::size_t Count, std::size_t FirstCoord, std::size_t... Coords>
			        struct GetTagsImpl<ChildType[Count], RecordCoord<FirstCoord, Coords...>>
			        {
			            using ChildTag = RecordCoord<FirstCoord>;
			            using type = mp_push_front<typename GetTagsImpl<ChildType, RecordCoord<Coords...>>::type, ChildTag>;
			        };

			        template<typename T>
			        struct GetTagsImpl<T, RecordCoord<>>
			        {
			            using type = mp_list<>;
			        };
			    } // namespace internal

			    /// Get the tags of all \ref Field%s from the root of the record dimension tree until to the node identified by
			    /// \ref RecordCoord.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename RecordCoord>
			    using GetTags = typename internal::GetTagsImpl<RecordDim, RecordCoord>::type;

			    namespace internal
			    {
			        template<typename RecordDim, typename RecordCoord>
			        struct GetTagImpl
			        {
			            using type = mp_back<GetTags<RecordDim, RecordCoord>>;
			        };

			        template<typename RecordDim>
			        struct GetTagImpl<RecordDim, RecordCoord<>>
			        {
			            using type = NoName;
			        };
			    } // namespace internal

			    /// Get the tag of the \ref Field at a \ref RecordCoord inside the record dimension tree.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename RecordCoord>
			    using GetTag = typename internal::GetTagImpl<RecordDim, RecordCoord>::type;

			    /// Is true if, starting at two coordinates in two record dimensions, all subsequent nodes in the record dimension
			    /// tree have the same tag.
			    /// \tparam RecordDimA First record dimension.
			    /// \tparam RecordCoordA \ref RecordCoord based on RecordDimA along which the tags are compared.
			    /// \tparam RecordDimB second record dimension.
			    /// \tparam RecordCoordB \ref RecordCoord based on RecordDimB along which the tags are compared.
			    LLAMA_EXPORT
			    template<typename RecordDimA, typename RecordCoordA, typename RecordDimB, typename RecordCoordB>
			    inline constexpr auto hasSameTags = []() constexpr
			    {
			        if constexpr(RecordCoordA::size != RecordCoordB::size)
			            return false;
			        else if constexpr(RecordCoordA::size == 0 && RecordCoordB::size == 0)
			            return true;
			        else
			            return std::is_same_v<GetTags<RecordDimA, RecordCoordA>, GetTags<RecordDimB, RecordCoordB>>;
			    }();

			    namespace internal
			    {
			        template<typename FieldList, typename Tag>
			        struct FindFieldByTag
			        {
			            template<typename Field>
			            using HasTag = std::is_same<GetFieldTag<Field>, Tag>;

			            static constexpr auto value = mp_find_if<FieldList, HasTag>::value;
			        };

			        template<typename RecordDim, typename RecordCoord, typename... Tags>
			        struct GetCoordFromTagsImpl
			        {
			            static_assert(mp_size<RecordDim>::value != 0, "Tag combination is not valid");
			        };

			        template<typename... Fields, std::size_t... ResultCoords, typename FirstTag, typename... Tags>
			        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<ResultCoords...>, FirstTag, Tags...>
			        {
			            static constexpr auto tagIndex = FindFieldByTag<mp_list<Fields...>, FirstTag>::value;
			            static_assert(
			                tagIndex < sizeof...(Fields),
			                "FirstTag was not found inside this Record. Does your record dimension contain the tag you access "
			                "with?");

			            using ChildType = GetFieldType<mp_at_c<Record<Fields...>, tagIndex>>;

			            using type =
			                typename GetCoordFromTagsImpl<ChildType, RecordCoord<ResultCoords..., tagIndex>, Tags...>::type;
			        };

			        template<
			            typename ChildType,
			            std::size_t Count,
			            std::size_t... ResultCoords,
			            typename FirstTag,
			            typename... Tags>
			        struct GetCoordFromTagsImpl<ChildType[Count], RecordCoord<ResultCoords...>, FirstTag, Tags...>
			        {
			            static_assert(isRecordCoord<FirstTag>, "Please use a RecordCoord<I> to index into static arrays");
			            static_assert(FirstTag::size == 1, "Expected RecordCoord with 1 coordinate");
			            static_assert(FirstTag::front < Count, "Index out of bounds");

			            using type =
			                typename GetCoordFromTagsImpl<ChildType, RecordCoord<ResultCoords..., FirstTag::front>, Tags...>::type;
			        };

			        template<typename RecordDim, typename RecordCoord>
			        struct GetCoordFromTagsImpl<RecordDim, RecordCoord>
			        {
			            using type = RecordCoord;
			        };

			        // unpack a list of tags
			        template<typename... Fields, typename... Tags>
			        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<>, mp_list<Tags...>>
			            : GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<>, Tags...>
			        {
			        };

			        template<typename ChildType, std::size_t Count, typename... Tags>
			        struct GetCoordFromTagsImpl<ChildType[Count], RecordCoord<>, mp_list<Tags...>>
			            : GetCoordFromTagsImpl<ChildType[Count], RecordCoord<>, Tags...>
			        {
			        };

			        // pass through a RecordCoord
			        template<typename... Fields, std::size_t... RCs>
			        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<>, RecordCoord<RCs...>>
			        {
			            using type = RecordCoord<RCs...>;
			        };
			    } // namespace internal

			    /// Converts a series of tags, or a list of tags, navigating down a record dimension into a \ref RecordCoord. A
			    /// RecordCoord will be passed through unmodified.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename... TagsOrTagList>
			    using GetCoordFromTags = typename internal::GetCoordFromTagsImpl<RecordDim, RecordCoord<>, TagsOrTagList...>::type;

			    namespace internal
			    {
			        template<typename RecordDim, typename... RecordCoordOrTags>
			        struct GetTypeImpl
			        {
			            using type = typename GetTypeImpl<RecordDim, GetCoordFromTags<RecordDim, RecordCoordOrTags...>>::type;
			        };

			        template<typename... Children, std::size_t HeadCoord, std::size_t... TailCoords>
			        struct GetTypeImpl<Record<Children...>, RecordCoord<HeadCoord, TailCoords...>>
			        {
			            using ChildType = GetFieldType<mp_at_c<Record<Children...>, HeadCoord>>;
			            using type = typename GetTypeImpl<ChildType, RecordCoord<TailCoords...>>::type;
			        };

			        template<typename ChildType, std::size_t N, std::size_t HeadCoord, std::size_t... TailCoords>
			        struct GetTypeImpl<ChildType[N], RecordCoord<HeadCoord, TailCoords...>>
			        {
			            using type = typename GetTypeImpl<ChildType, RecordCoord<TailCoords...>>::type;
			        };

			        template<typename T>
			        struct GetTypeImpl<T, RecordCoord<>>
			        {
			            static_assert(isAllowedFieldType<T>);
			            using type = T;
			        };
			    } // namespace internal

			    /// Returns the type of a node in a record dimension tree identified by a given \ref RecordCoord or a series of
			    /// tags.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename... RecordCoordOrTags>
			    using GetType = typename internal::GetTypeImpl<RecordDim, RecordCoordOrTags...>::type;

			    namespace internal
			    {
			        template<typename RecordDim, typename RecordCoord>
			        struct LeafRecordCoordsImpl;

			        template<typename T, std::size_t... RCs>
			        struct LeafRecordCoordsImpl<T, RecordCoord<RCs...>>
			        {
			            using type = mp_list<RecordCoord<RCs...>>;
			        };

			        template<typename... Fields, std::size_t... RCs>
			        struct LeafRecordCoordsImpl<Record<Fields...>, RecordCoord<RCs...>>
			        {
			            template<std::size_t... Is>
			            static auto help(std::index_sequence<Is...>)
			            {
			                return mp_append<
			                    typename LeafRecordCoordsImpl<GetFieldType<Fields>, RecordCoord<RCs..., Is>>::type...>{};
			            }
			            using type = decltype(help(std::make_index_sequence<sizeof...(Fields)>{}));
			        };

			        template<typename Child, std::size_t N, std::size_t... RCs>
			        struct LeafRecordCoordsImpl<Child[N], RecordCoord<RCs...>>
			        {
			            template<std::size_t... Is>
			            static auto help(std::index_sequence<Is...>)
			            {
			                return mp_append<typename LeafRecordCoordsImpl<Child, RecordCoord<RCs..., Is>>::type...>{};
			            }
			            using type = decltype(help(std::make_index_sequence<N>{}));
			        };
			    } // namespace internal

			    /// Returns a flat type list containing all record coordinates to all leaves of the given record dimension.
			    LLAMA_EXPORT
			    template<typename RecordDim>
			    using LeafRecordCoords = typename internal::LeafRecordCoordsImpl<RecordDim, RecordCoord<>>::type;

			    /// Iterates over the record dimension tree and calls a functor on each element.
			    /// \param functor Functor to execute at each element of. Needs to have `operator()` with a template parameter for
			    /// the \ref RecordCoord in the record dimension tree.
			    /// \param baseCoord \ref RecordCoord at which the iteration should be started. The functor is called on elements
			    /// beneath this coordinate.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename Functor, std::size_t... Coords>
			    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafCoord(Functor&& functor, RecordCoord<Coords...> baseCoord)
			    {
			        LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
			        internal::mpForEachInlined(
			            LeafRecordCoords<GetType<RecordDim, RecordCoord<Coords...>>>{},
			            [&](auto innerCoord) LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr)
			            { std::forward<Functor>(functor)(cat(baseCoord, innerCoord)); });
			        LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
			    }

			    /// Iterates over the record dimension tree and calls a functor on each element.
			    /// \param functor Functor to execute at each element of. Needs to have `operator()` with a template parameter for
			    /// the \ref RecordCoord in the record dimension tree.
			    /// \param baseTags Tags used to define where the iteration should be started. The functor is called on elements
			    /// beneath this coordinate.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename Functor, typename... Tags>
			    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafCoord(Functor&& functor, Tags... /*baseTags*/)
			    {
			        forEachLeafCoord<RecordDim>(std::forward<Functor>(functor), GetCoordFromTags<RecordDim, Tags...>{});
			    }

			    namespace internal
			    {
			        template<typename T>
			        struct FlattenRecordDimImpl
			        {
			            using type = mp_list<T>;
			        };

			        template<typename... Fields>
			        struct FlattenRecordDimImpl<Record<Fields...>>
			        {
			            using type = mp_append<typename FlattenRecordDimImpl<GetFieldType<Fields>>::type...>;
			        };
			        template<typename Child, std::size_t N>
			        struct FlattenRecordDimImpl<Child[N]>
			        {
			            using type = mp_repeat_c<typename FlattenRecordDimImpl<Child>::type, N>;
			        };
			    } // namespace internal

			    /// Returns a flat type list containing all leaf field types of the given record dimension.
			    LLAMA_EXPORT
			    template<typename RecordDim>
			    using FlatRecordDim = typename internal::FlattenRecordDimImpl<RecordDim>::type;

			    /// The total number of fields in the recursively expanded record dimension.
			    LLAMA_EXPORT
			    template<typename RecordDim>
			    inline constexpr std::size_t flatFieldCount = 1;

			    LLAMA_EXPORT
			    template<typename... Children>
			    inline constexpr std::size_t flatFieldCount<Record<Children...>>
			        = (flatFieldCount<GetFieldType<Children>> + ... + 0);

			    LLAMA_EXPORT
			    template<typename Child, std::size_t N>
			    inline constexpr std::size_t flatFieldCount<Child[N]> = flatFieldCount<Child> * N;

			    namespace internal
			    {
			        template<std::size_t I, typename RecordDim>
			        inline constexpr std::size_t flatFieldCountBefore = 0;

			        template<typename... Children>
			        inline constexpr std::size_t flatFieldCountBefore<0, Record<Children...>> = 0;

			        // recursive formulation to benefit from template instantiation memoization
			        // this massively improves compilation time when this template is instantiated with a lot of different I
			        template<std::size_t I, typename... Children>
			        inline constexpr std::size_t flatFieldCountBefore<I, Record<Children...>>
			            = flatFieldCountBefore<I - 1, Record<Children...>>
			            + flatFieldCount<GetFieldType<mp_at_c<Record<Children...>, I - 1>>>;
			    } // namespace internal

			    /// The equivalent zero based index into a flat record dimension (\ref FlatRecordDim) of the given hierarchical
			    /// record coordinate.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename RecordCoord>
			    inline constexpr std::size_t flatRecordCoord = 0;

			    LLAMA_EXPORT
			    template<typename T>
			    inline constexpr std::size_t flatRecordCoord<T, RecordCoord<>> = 0;

			    LLAMA_EXPORT
			    template<typename... Children, std::size_t I, std::size_t... Is>
			    inline constexpr std::size_t flatRecordCoord<Record<Children...>, RecordCoord<I, Is...>>
			        = internal::flatFieldCountBefore<I, Record<Children...>>
			        + flatRecordCoord<GetFieldType<mp_at_c<Record<Children...>, I>>, RecordCoord<Is...>>;

			    LLAMA_EXPORT
			    template<typename Child, std::size_t N, std::size_t I, std::size_t... Is>
			    inline constexpr std::size_t flatRecordCoord<Child[N], RecordCoord<I, Is...>>
			        = flatFieldCount<Child> * I + flatRecordCoord<Child, RecordCoord<Is...>>;

			    namespace internal
			    {
			        template<typename TypeList>
			        constexpr auto flatAlignOfImpl()
			        {
			            std::size_t maxAlign = 0;
			            mp_for_each<mp_transform<mp_identity, TypeList>>(
			                [&](auto e) constexpr
			                {
			                    using T = typename decltype(e)::type;
			                    maxAlign = std::max(maxAlign, alignof(T));
			                });
			            return maxAlign;
			        }
			    } // namespace internal

			    /// The alignment of a type list if its elements would be in a normal struct. Effectively returns the maximum
			    /// alignment value in the type list.
			    LLAMA_EXPORT
			    template<typename TypeList>
			    inline constexpr std::size_t flatAlignOf = internal::flatAlignOfImpl<TypeList>();

			    /// The alignment of a type T.
			    LLAMA_EXPORT
			    template<typename T>
			    inline constexpr std::size_t alignOf = alignof(T);

			    /// The alignment of a record dimension if its fields would be in a normal struct. Effectively returns the maximum
			    /// alignment value in the type list.
			    LLAMA_EXPORT
			    template<typename... Fields>
			    inline constexpr std::size_t alignOf<Record<Fields...>> = flatAlignOf<FlatRecordDim<Record<Fields...>>>;

			    /// Returns the ceiling of a / b.
			    LLAMA_EXPORT
			    template<typename Integral>
			    [[nodiscard]] LLAMA_FN_HOST_ACC_INLINE constexpr auto divCeil(Integral a, Integral b) -> Integral
			    {
			        return (a + b - 1) / b;
			    }

			    /// Returns the integral n rounded up to be a multiple of mult.
			    LLAMA_EXPORT
			    template<typename Integral>
			    [[nodiscard]] LLAMA_FN_HOST_ACC_INLINE constexpr auto roundUpToMultiple(Integral n, Integral mult) -> Integral
			    {
			        return divCeil(n, mult) * mult;
			    }

			    namespace internal
			    {
			        template<typename TypeList, bool Align, bool IncludeTailPadding>
			        constexpr auto sizeOfImpl() -> std::size_t
			        {
			            std::size_t size = 0;
			            std::size_t maxAlign = 0; // NOLINT(misc-const-correctness)
			            mp_for_each<mp_transform<mp_identity, TypeList>>(
			                [&](auto e) constexpr
			                {
			                    using T = typename decltype(e)::type;
			                    if constexpr(Align)
			                    {
			                        size = roundUpToMultiple(size, alignof(T));
			                        maxAlign = std::max(maxAlign, alignof(T));
			                    }
			                    // NOLINTNEXTLINE(readability-misleading-indentation)
			                    size += sizeof(T);
			                });

			            // final padding, so next struct can start right away
			            if constexpr(Align && IncludeTailPadding)
			                if(maxAlign > 0)
			                    size = roundUpToMultiple(size, maxAlign); // TODO(bgruber): we could use flatAlignOf<TypeList>
			                                                              // here, at the cost of more template instantiations
			            return size;
			        }

			        template<typename TypeList, std::size_t I, bool Align>
			        constexpr auto offsetOfImplWorkaround() -> std::size_t;
			    } // namespace internal

			    /// The size of a type list if its elements would be in a normal struct.
			    LLAMA_EXPORT
			    template<typename TypeList, bool Align, bool IncludeTailPadding = true>
			    inline constexpr std::size_t flatSizeOf = internal::sizeOfImpl<TypeList, Align, IncludeTailPadding>();

			    /// The size of a type T.
			    LLAMA_EXPORT
			    template<typename T, bool Align = false, bool IncludeTailPadding = true>
			    inline constexpr std::size_t sizeOf = sizeof(T);

			    /// The size of a record dimension if its fields would be in a normal struct.
			    LLAMA_EXPORT
			    template<typename... Fields, bool Align, bool IncludeTailPadding>
			    inline constexpr std::size_t sizeOf<Record<Fields...>, Align, IncludeTailPadding>
			        = flatSizeOf<FlatRecordDim<Record<Fields...>>, Align, IncludeTailPadding>;

			    /// The byte offset of an element in a type list ifs elements would be in a normal struct.
			    LLAMA_EXPORT
			    template<typename TypeList, std::size_t I, bool Align>
			    inline constexpr std::size_t flatOffsetOf = internal::offsetOfImplWorkaround<TypeList, I, Align>();

			    namespace internal
			    {
			        // unfortunately, we cannot inline this function as an IILE, as MSVC complains:
			        // fatal error C1202: recursive type or function dependency context too complex
			        template<typename TypeList, std::size_t I, bool Align>
			        constexpr auto offsetOfImplWorkaround() -> std::size_t
			        {
			            if constexpr(I == 0)
			                return 0;
			            else
			            {
			                std::size_t offset // NOLINT(misc-const-correctness)
			                    = flatOffsetOf<TypeList, I - 1, Align> + sizeof(mp_at_c<TypeList, I - 1>);
			                if constexpr(Align)
			                    offset = roundUpToMultiple(offset, alignof(mp_at_c<TypeList, I>));
			                return offset;
			            }
			        }
			    } // namespace internal

			    /// The byte offset of an element in a record dimension if it would be a normal struct.
			    /// \tparam RecordDim Record dimension tree.
			    /// \tparam RecordCoord Record coordinate of an element inrecord dimension tree.
			    LLAMA_EXPORT
			    template<typename RecordDim, typename RecordCoord, bool Align = false>
			    inline constexpr std::size_t offsetOf
			        = flatOffsetOf<FlatRecordDim<RecordDim>, flatRecordCoord<RecordDim, RecordCoord>, Align>;

			    namespace internal
			    {
			        // Such a class is also known as arraw_proxy: https://quuxplusone.github.io/blog/2019/02/06/arrow-proxy/
			        template<typename T>
			        struct IndirectValue
			        {
			            T value;

			            LLAMA_FN_HOST_ACC_INLINE auto operator->() -> T*
			            {
			                return &value;
			            }

			            LLAMA_FN_HOST_ACC_INLINE auto operator->() const -> const T*
			            {
			                return &value;
			            }
			        };

			        // TODO(bgruber): replace in C++20
			        template<class T>
			        struct IsBoundedArray : std::false_type
			        {
			        };

			        template<class T, std::size_t N>
			        struct IsBoundedArray<T[N]> : std::true_type
			        {
			        };
			    } // namespace internal

			    /// True if the T is a record dimension. That is, T is either a llama::Record or a bounded array.
			    LLAMA_EXPORT
			    template<typename T>
			    inline constexpr bool isRecordDim = isRecord<T> || internal::IsBoundedArray<T>::value;

			    namespace internal
			    {
			        template<typename Coord, typename T, template<typename, typename> typename TypeFunctor>
			        struct TransformLeavesWithCoordImpl
			        {
			            using type = TypeFunctor<Coord, T>;
			        };

			        template<std::size_t... Is, typename... Fields, template<typename, typename> typename TypeFunctor>
			        struct TransformLeavesWithCoordImpl<RecordCoord<Is...>, Record<Fields...>, TypeFunctor>
			        {
			            template<std::size_t... Js>
			            static auto f(std::index_sequence<Js...>)
			            {
			                return Record<Field<
			                    GetFieldTag<Fields>,
			                    typename TransformLeavesWithCoordImpl<RecordCoord<Is..., Js>, GetFieldType<Fields>, TypeFunctor>::
			                        type>...>{};
			            }

			            using type = decltype(f(std::index_sequence_for<Fields...>{}));
			        };
			        template<std::size_t... Is, typename Child, std::size_t N, template<typename, typename> typename TypeFunctor>
			        struct TransformLeavesWithCoordImpl<RecordCoord<Is...>, Child[N], TypeFunctor>
			        {
			            template<std::size_t... Js>
			            static void f(std::index_sequence<Js...>)
			            {
			                static_assert(
			                    mp_same<typename TransformLeavesWithCoordImpl<RecordCoord<Is..., Js>, Child, TypeFunctor>::
			                                type...>::value,
			                    "Leave transformations beneath an array node must return the same type");
			            }
			            using dummy = decltype(f(std::make_index_sequence<N>{}));

			            using type = typename TransformLeavesWithCoordImpl<RecordCoord<Is..., 0>, Child, TypeFunctor>::type[N];
			        };

			        template<template<typename> typename F>
			        struct MakePassSecond
			        {
			            template<typename A, typename B>
			            using fn = F<B>;
			        };
			    } // namespace internal

			    /// Creates a new record dimension where each new leaf field's type is the result of applying FieldTypeFunctor to
			    /// the original leaf's \ref RecordCoord and field's type.
			    LLAMA_EXPORT
			    template<typename RecordDim, template<typename, typename> typename FieldTypeFunctor>
			    using TransformLeavesWithCoord =
			        typename internal::TransformLeavesWithCoordImpl<RecordCoord<>, RecordDim, FieldTypeFunctor>::type;

			    /// Creates a new record dimension where each new leaf field's type is the result of applying FieldTypeFunctor to
			    /// the original leaf field's type.
			    LLAMA_EXPORT
			    template<typename RecordDim, template<typename> typename FieldTypeFunctor>
			    using TransformLeaves
			        = TransformLeavesWithCoord<RecordDim, internal::MakePassSecond<FieldTypeFunctor>::template fn>;

			    namespace internal
			    {
			        // TODO(bgruber): we might implement this better by expanding a record dim into a list of tag lists and then
			        // computing a real set union of the two tag list lists

			        template<typename A, typename B>
			        auto mergeRecordDimsImpl(mp_identity<A> a, mp_identity<B>)
			        {
			            static_assert(std::is_same_v<A, B>, "Cannot merge record and non-record or fields with different types");
			            return a;
			        }

			        template<typename A, std::size_t NA, typename B, std::size_t NB>
			        auto mergeRecordDimsImpl([[maybe_unused]] mp_identity<A[NA]> a, [[maybe_unused]] mp_identity<B[NB]> b)
			        {
			            static_assert(std::is_same_v<A, B>, "Cannot merge arrays of different type");
			            if constexpr(NA < NB)
			                return b;
			            else
			                return a;
			        }

			        template<typename... FieldsA>
			        auto mergeRecordDimsImpl(mp_identity<Record<FieldsA...>> a, mp_identity<Record<>>)
			        {
			            return a;
			        }

			        template<
			            typename... FieldsA,
			            typename FieldB,
			            typename... FieldsB,
			            auto Pos = FindFieldByTag<Record<FieldsA...>, GetFieldTag<FieldB>>::value>
			        auto mergeRecordDimsImpl(mp_identity<Record<FieldsA...>>, mp_identity<Record<FieldB, FieldsB...>>)
			        {
			            if constexpr(Pos == sizeof...(FieldsA))
			            {
			                return mergeRecordDimsImpl(
			                    mp_identity<Record<FieldsA..., FieldB>>{},
			                    mp_identity<Record<FieldsB...>>{});
			            }
			            else
			            {
			                using OldFieldA = mp_at_c<Record<FieldsA...>, Pos>;
			                using NewFieldA = Field<
			                    GetFieldTag<OldFieldA>,
			                    typename decltype(mergeRecordDimsImpl(
			                        mp_identity<GetFieldType<OldFieldA>>{},
			                        mp_identity<GetFieldType<FieldB>>{}))::type>;
			                using NewRecordA = mp_replace_at_c<Record<FieldsA...>, Pos, NewFieldA>;
			                return mergeRecordDimsImpl(mp_identity<NewRecordA>{}, mp_identity<Record<FieldsB...>>{});
			            }
			        }
			    } // namespace internal

			    /// Creates a merged record dimension, where duplicated, nested fields are unified.
			    LLAMA_EXPORT
			    template<typename RecordDimA, typename RecordDimB>
			    using MergedRecordDims =
			        typename decltype(internal::mergeRecordDimsImpl(mp_identity<RecordDimA>{}, mp_identity<RecordDimB>{}))::type;

			    /// Alias for ToT, adding `const` if FromT is const qualified.
			    LLAMA_EXPORT
			    template<typename FromT, typename ToT>
			    using CopyConst = std::conditional_t<std::is_const_v<FromT>, const ToT, ToT>;

			    /// Used as template argument to specify a constant/compile-time value.
			    LLAMA_EXPORT
			    template<auto V>
			    using Constant = std::integral_constant<decltype(V), V>;

			    namespace internal
			    {
			        template<typename T>
			        struct IsConstant : std::false_type
			        {
			        };

			        template<typename T, T V>
			        struct IsConstant<std::integral_constant<T, V>> : std::true_type
			        {
			        };
			    } // namespace internal

			    LLAMA_EXPORT
			    template<typename T>
			    inline constexpr bool isConstant = internal::IsConstant<T>::value;

			    namespace internal
			    {
			        /// Holds a value of type T. Is useful as a base class. Is specialized for llama::Constant to not store the
			        /// value at runtime. \tparam T Type of value to store. \tparam I Is used to disambiguate multiple BoxedValue
			        /// base classes.
			        template<typename T, int I = 0>
			        struct BoxedValue
			        {
			            BoxedValue() = default;

			            // we don't make this ctor explicit so a Value appearing in a ctor list can just be created by passing a T
			            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
			            LLAMA_FN_HOST_ACC_INLINE BoxedValue(T value) : val(value)
			            {
			            }

			            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() const
			            {
			                return val;
			            }

			        private:
			            T val = {};
			        };

			        template<auto V, int I>
			        struct BoxedValue<Constant<V>, I>
			        {
			            BoxedValue() = default;

			            // we don't make this ctor explicit so a Value appearing in a ctor list can just be created by passing a T
			            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
			            LLAMA_FN_HOST_ACC_INLINE BoxedValue(Constant<V>)
			            {
			            }

			            LLAMA_FN_HOST_ACC_INLINE static constexpr auto value()
			            {
			                return V;
			            }
			        };
			    } // namespace internal

			    LLAMA_EXPORT
			    struct PrettySize
			    {
			        double size;
			        const char* unit;
			    };

			    /// Repeatedly divides the given size (in bytes) until it fits below 1000. Returns the new size and a string
			    /// literal with the corresponding unit.
			    LLAMA_EXPORT
			    inline auto prettySize(double size) -> PrettySize
			    {
			        static const char* unit[] = {"B ", "KB", "MB", "GB", "TB", "PB", "EB"};
			        unsigned unitIndex = 0;
			        while(size > 1000.0)
			        {
			            size /= 1000.0;
			            unitIndex++;
			        }
			        assert(unitIndex < std::size(unit));
			        return {size, unit[unitIndex]};
			    }
			} // namespace llama
			// ==
			// == ./include/llama/Core.hpp ==
			// ============================================================================

		// #include "RecordCoord.hpp"    // amalgamate: file already inlined

		// #include <type_traits>    // amalgamate: file already included

		#if __has_include(<concepts>)
		#    include <concepts>
		#endif
		namespace llama
		{
		#ifdef __cpp_lib_concepts
		    LLAMA_EXPORT
		    template<auto I>
		    concept isConstexpr = requires { std::integral_constant<decltype(I), I>{}; };

		    LLAMA_EXPORT
		    template<typename M>
		    concept Mapping = requires(M m) {
		        typename M::ArrayExtents;
		        typename M::RecordDim;
		        {
		            m.extents()
		        } -> std::same_as<typename M::ArrayExtents>;
		        {
		            +M::blobCount
		        } -> std::same_as<std::size_t>;
		        requires isConstexpr<M::blobCount>;
		        {
		            m.blobSize(typename M::ArrayExtents::value_type{})
		        } -> std::same_as<typename M::ArrayExtents::value_type>;
		    };

		    LLAMA_EXPORT
		    template<typename M, typename RC>
		    concept PhysicalField = requires(M m, typename M::ArrayExtents::Index ai) {
		        {
		            m.blobNrAndOffset(ai, RC{})
		        } -> std::same_as<NrAndOffset<typename M::ArrayExtents::value_type>>;
		    };

		    template<typename M>
		    struct MakeIsPhysical
		    {
		        template<typename RC>
		        using fn = mp_bool<PhysicalField<M, RC>>;
		    };

		    LLAMA_EXPORT
		    template<typename M>
		    inline constexpr bool allFieldsArePhysical
		        = mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsPhysical<M>::template fn>::value;

		    LLAMA_EXPORT
		    template<typename M>
		    concept PhysicalMapping = Mapping<M> && allFieldsArePhysical<M>;

		    LLAMA_EXPORT
		    template<typename R>
		    concept LValueReference = std::is_lvalue_reference_v<R>;

		    LLAMA_EXPORT
		    template<typename R>
		    concept AdlTwoStepSwappable = requires(R a, R b) { swap(a, b); } || requires(R a, R b) { std::swap(a, b); };

		    LLAMA_EXPORT
		    template<typename R>
		    concept ProxyReference = std::is_copy_constructible_v<R> && std::is_copy_assignable_v<R> && requires(R r) {
		        typename R::value_type;
		        {
		            static_cast<typename R::value_type>(r)
		        } -> std::same_as<typename R::value_type>;
		        {
		            r = std::declval<typename R::value_type>()
		        } -> std::same_as<R&>;
		    } && AdlTwoStepSwappable<R>;

		    LLAMA_EXPORT
		    template<typename R>
		    concept AnyReference = LValueReference<R> || ProxyReference<R>;

		    LLAMA_EXPORT
		    template<typename R, typename T>
		    concept AnyReferenceTo = (LValueReference<R> && std::is_same_v<std::remove_cvref_t<R>, T>)
		        || (ProxyReference<R> && std::is_same_v<typename R::value_type, T>);

		    LLAMA_EXPORT
		    template<typename M, typename RC>
		    concept ComputedField
		        = M::isComputed(RC{}) && requires(M m, typename M::ArrayExtents::Index ai, std::byte** blobs) {
		              {
		                  m.compute(ai, RC{}, blobs)
		              } -> AnyReferenceTo<GetType<typename M::RecordDim, RC>>;
		          };

		    template<typename M>
		    struct MakeIsComputed
		    {
		        template<typename RC>
		        using fn = mp_bool<ComputedField<M, RC>>;
		    };

		    LLAMA_EXPORT
		    template<typename M>
		    inline constexpr bool allFieldsAreComputed
		        = mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsComputed<M>::template fn>::value;

		    LLAMA_EXPORT
		    template<typename M>
		    concept FullyComputedMapping = Mapping<M> && allFieldsAreComputed<M>;

		    LLAMA_EXPORT
		    template<
		        typename M,
		        typename LeafCoords = LeafRecordCoords<typename M::RecordDim>,
		        std::size_t PhysicalCount = mp_count_if<LeafCoords, MakeIsPhysical<M>::template fn>::value,
		        std::size_t ComputedCount = mp_count_if<LeafCoords, MakeIsComputed<M>::template fn>::value>
		    inline constexpr bool allFieldsArePhysicalOrComputed
		        = (PhysicalCount + ComputedCount) >= mp_size<LeafCoords>::value && PhysicalCount > 0
		        && ComputedCount > 0; // == instead of >= would be better, but it's not easy to count correctly,
		                              // because we cannot check whether the call to blobNrOrOffset()
		                              // or compute() is actually valid

		    LLAMA_EXPORT
		    template<typename M>
		    concept PartiallyComputedMapping = Mapping<M> && allFieldsArePhysicalOrComputed<M>;

		    /// Additional semantic requirement: &b[i] + j == &b[i + j] for any integral i and j in range of the blob
		    LLAMA_EXPORT
		    template<typename B>
		    concept Blob = requires(B b, std::size_t i) {
		        // according to http://eel.is/c++draft/intro.object#3 only std::byte and unsigned char can
		        // provide storage for
		        // other types
		        requires std::is_lvalue_reference_v<decltype(b[i])>;
		        requires std::same_as<std::remove_cvref_t<decltype(b[i])>, std::byte>
		            || std::same_as<std::remove_cvref_t<decltype(b[i])>, unsigned char>;
		    };

		    LLAMA_EXPORT
		    template<typename BA>
		    concept BlobAllocator = requires(BA ba, std::size_t size) {
		        {
		            ba(std::integral_constant<std::size_t, 16>{}, size)
		        } -> Blob;
		    };

		    LLAMA_EXPORT
		    template<typename V>
		    concept AnyView = requires(V v, const V cv) {
		        typename V::Mapping;
		        typename V::BlobType;
		        typename V::ArrayExtents;
		        typename V::ArrayIndex;
		        typename V::RecordDim;
		        typename V::Accessor;

		        typename V::iterator;
		        typename V::const_iterator;

		        {
		            v.mapping()
		        } -> std::same_as<typename V::Mapping&>;

		        {
		            cv.mapping()
		        } -> std::same_as<const typename V::Mapping&>;

		        {
		            v.accessor()
		        } -> std::same_as<typename V::Accessor&>;

		        {
		            cv.accessor()
		        } -> std::same_as<const typename V::Accessor&>;

		        {
		            cv.extents()
		        } -> std::same_as<typename V::ArrayExtents>;

		        {
		            v.begin()
		        } -> std::same_as<typename V::iterator>;

		        {
		            cv.begin()
		        } -> std::same_as<typename V::const_iterator>;

		        {
		            v.end()
		        } -> std::same_as<typename V::iterator>;

		        {
		            cv.end()
		        } -> std::same_as<typename V::const_iterator>;

		        {
		            v.blobs()
		        } -> std::same_as<Array<typename V::BlobType, V::Mapping::blobCount>&>;
		        {
		            cv.blobs()
		        } -> std::same_as<const Array<typename V::BlobType, V::Mapping::blobCount>&>;
		    };
		#endif

		    namespace internal
		    {
		        template<typename R, typename = void>
		        struct IsProxyReferenceImpl : std::false_type
		        {
		        };

		        template<typename R>
		        struct IsProxyReferenceImpl<
		            R,
		            std::void_t<
		                typename R::value_type,
		                decltype(static_cast<typename R::value_type>(std::declval<R&>())),
		                decltype(std::declval<R&>() = std::declval<typename R::value_type>())>>
		            : std::bool_constant<std::is_copy_constructible_v<R> && std::is_copy_assignable_v<R>>
		        {
		        };
		    } // namespace internal

		    LLAMA_EXPORT
		    template<typename R>
		#ifdef __cpp_lib_concepts
		    inline constexpr bool isProxyReference = ProxyReference<R>;
		#else
		    inline constexpr bool isProxyReference = internal::IsProxyReferenceImpl<R>::value;
		#endif
		} // namespace llama
		// ==
		// == ./include/llama/Concepts.hpp ==
		// ============================================================================

		// ============================================================================
		// == ./include/llama/ProxyRefOpMixin.hpp ==
		// ==
		// Copyright 2022 Bernhard Manfred Gruber
		// SPDX-License-Identifier: MPL-2.0

		// #pragma once
		// #include "macros.hpp"    // amalgamate: file already inlined

		namespace llama
		{
		    /// CRTP mixin for proxy reference types to support all compound assignment and increment/decrement operators.
		    LLAMA_EXPORT
		    template<typename Derived, typename ValueType>
		    struct ProxyRefOpMixin
		    {
		    private:
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto derived() -> Derived&
		        {
		            return static_cast<Derived&>(*this);
		        }

		        // in principle, load() could be const, but we use it only from non-const functions
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto load() -> ValueType
		        {
		            return static_cast<ValueType>(derived());
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr void store(ValueType t)
		        {
		            derived() = std::move(t);
		        }

		    public:
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator+=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs += rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator-=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs -= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator*=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs *= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator/=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs /= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator%=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs %= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator<<=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs <<= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator>>=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs >>= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator&=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs &= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator|=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs |= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator^=(const ValueType& rhs) -> Derived&
		        {
		            ValueType lhs = load();
		            lhs ^= rhs;
		            store(lhs);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator++() -> Derived&
		        {
		            ValueType v = load();
		            ++v;
		            store(v);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator++(int) -> ValueType
		        {
		            ValueType v = load();
		            ValueType old = v++;
		            store(v);
		            return old;
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator--() -> Derived&
		        {
		            ValueType v = load();
		            --v;
		            store(v);
		            return derived();
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator--(int) -> ValueType
		        {
		            ValueType v = load();
		            ValueType old = v--;
		            store(v);
		            return old;
		        }

		        LLAMA_FN_HOST_ACC_INLINE friend constexpr void swap(Derived a, Derived b) noexcept
		        {
		            const auto va = static_cast<ValueType>(a);
		            const auto vb = static_cast<ValueType>(b);
		            a = vb;
		            b = va;
		        }
		    };
		} // namespace llama
		// ==
		// == ./include/llama/ProxyRefOpMixin.hpp ==
		// ============================================================================

	// #include "macros.hpp"    // amalgamate: file already inlined

	#include <atomic>
	#include <memory>
	#include <mutex>

	namespace llama::accessor
	{
	    /// Default accessor. Passes through the given reference.
	    LLAMA_EXPORT
	    struct Default
	    {
	        template<typename Reference>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const -> Reference
	        {
	            return std::forward<Reference>(r);
	        }
	    };

	    /// Allows only read access and returns values instead of references to memory.
	    LLAMA_EXPORT
	    struct ByValue
	    {
	        template<typename Reference>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const
	        {
	            using ValueType = std::decay_t<Reference>;
	            if constexpr(isProxyReference<ValueType>)
	                return static_cast<typename ValueType::value_type>(r);
	            else
	                return ValueType{r};
	        }
	    };

	    /// Allows only read access by qualifying the references to memory with const.
	    LLAMA_EXPORT
	    struct Const
	    {
	        // for l-value references
	        template<typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> const T&
	        {
	            return r;
	        }

	        template<typename Ref>
	        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
	        struct Reference : ProxyRefOpMixin<Reference<Ref>, typename Ref::value_type>
	        {
	        private:
	            Ref ref;

	        public:
	            using value_type = typename Ref::value_type;

	            LLAMA_FN_HOST_ACC_INLINE constexpr explicit Reference(Ref ref) : ref{ref}
	            {
	            }

	            Reference(const Reference&) = default;

	            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
	            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const Reference& other) -> Reference&
	            {
	                *this = static_cast<value_type>(other);
	                return *this;
	            }

	            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	            LLAMA_FN_HOST_ACC_INLINE operator value_type() const
	            {
	                return static_cast<value_type>(ref);
	            }

	            template<typename T>
	            LLAMA_FN_HOST_ACC_INLINE auto operator=(T) -> Reference&
	            {
	                static_assert(sizeof(T) == 0, "You cannot write through a Const accessor");
	                return *this;
	            }
	        };

	        // for proxy references
	        template<typename ProxyReference, std::enable_if_t<llama::isProxyReference<ProxyReference>, int> = 0>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ProxyReference r) const
	        {
	            return Reference<ProxyReference>{std::move(r)};
	        }
	    };

	    /// Qualifies references to memory with __restrict. Only works on l-value references.
	    LLAMA_EXPORT
	    struct Restrict
	    {
	        template<typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> T& __restrict
	        {
	            return r;
	        }
	    };

	#ifdef __cpp_lib_atomic_ref
	    /// Accessor wrapping a reference into a std::atomic_ref. Can only wrap l-value references.
	    LLAMA_EXPORT
	    struct Atomic
	    {
	        template<typename T>
	        LLAMA_FORCE_INLINE auto operator()(T& r) const -> std::atomic_ref<T>
	        {
	            return std::atomic_ref<T>{r};
	        }
	    };
	#endif

	    /// Locks a mutex during each access to the data structure.
	    LLAMA_EXPORT
	    template<typename Mutex = std::mutex>
	    struct Locked
	    {
	        // mutexes are usually not movable, so we put them on the heap, so the accessor is movable
	        std::unique_ptr<Mutex> m = std::make_unique<Mutex>();

	        template<typename Ref, typename Value>
	        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
	        struct Reference : ProxyRefOpMixin<Reference<Ref, Value>, Value>
	        {
	            Ref ref;
	            Mutex& m;

	            using value_type = Value;

	            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
	            LLAMA_FORCE_INLINE constexpr auto operator=(const Reference& other) -> Reference&
	            {
	                const std::lock_guard<Mutex> lock(m);
	                *this = static_cast<value_type>(other);
	                return *this;
	            }

	            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	            LLAMA_FORCE_INLINE operator value_type() const
	            {
	                const std::lock_guard<Mutex> lock(m);
	                return static_cast<value_type>(ref);
	            }

	            template<typename T>
	            LLAMA_FORCE_INLINE auto operator=(T t) -> Reference&
	            {
	                const std::lock_guard<Mutex> lock(m);
	                ref = t;
	                return *this;
	            }
	        };

	        template<typename PR>
	        LLAMA_FORCE_INLINE auto operator()(PR r) const -> Reference<PR, typename PR::value_type>
	        {
	            return {{}, r, *m};
	        }

	        template<typename T>
	        LLAMA_FORCE_INLINE auto operator()(T& r) const -> Reference<T&, std::remove_cv_t<T>>
	        {
	            return {{}, r, *m};
	        }
	    };

	    namespace internal
	    {
	        template<std::size_t I, typename Accessor>
	        struct StackedLeave : Accessor
	        {
	        };
	    } // namespace internal

	    /// Accessor combining multiple other accessors. The contained accessors are applied in left to right order to the
	    /// memory location when forming the reference returned from a view.
	    LLAMA_EXPORT
	    template<typename... Accessors>
	    struct Stacked : internal::StackedLeave<0, Default>
	    {
	    };

	    LLAMA_EXPORT
	    template<typename FirstAccessor, typename... MoreAccessors>
	    struct Stacked<FirstAccessor, MoreAccessors...>
	        : internal::StackedLeave<1 + sizeof...(MoreAccessors), FirstAccessor>
	        , Stacked<MoreAccessors...>
	    {
	        using First = internal::StackedLeave<1 + sizeof...(MoreAccessors), FirstAccessor>;
	        using Rest = Stacked<MoreAccessors...>;

	        LLAMA_FN_HOST_ACC_INLINE Stacked() = default;

	        LLAMA_FN_HOST_ACC_INLINE explicit Stacked(FirstAccessor first, MoreAccessors... rest)
	            : First{std::move(first)}
	            , Rest{std::move(rest)...}
	        {
	        }

	        template<typename Reference>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const -> decltype(auto)
	        {
	            return static_cast<const Rest&>(*this)(static_cast<const First&>(*this)(std::forward<Reference>(r)));
	        }
	    };
	} // namespace llama::accessor
	// ==
	// == ./include/llama/Accessors.hpp ==
	// ============================================================================

// #include "Array.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/ArrayIndexRange.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "ArrayExtents.hpp"    // amalgamate: file already inlined
	// #include "Core.hpp"    // amalgamate: file already inlined
	// #include "macros.hpp"    // amalgamate: file already inlined

	#include <algorithm>
	#include <iterator>
	// #include <limits>    // amalgamate: file already included
	#if CAN_USE_RANGES
	#    include <ranges>
	#endif

	namespace llama
	{
	    /// Iterator supporting \ref ArrayIndexRange.
	    LLAMA_EXPORT
	    template<typename ArrayExtents>
	    struct ArrayIndexIterator
	    {
	        static_assert(!std::is_const_v<ArrayExtents>);

	        using value_type = typename ArrayExtents::Index;
	        using difference_type = std::ptrdiff_t;
	        using reference = value_type;
	        using pointer = internal::IndirectValue<value_type>;
	        using iterator_category = std::random_access_iterator_tag;

	        static constexpr std::size_t rank = ArrayExtents::rank;

	        constexpr ArrayIndexIterator() noexcept = default;

	        LLAMA_FN_HOST_ACC_INLINE constexpr ArrayIndexIterator(ArrayExtents extents, value_type current) noexcept
	            : extents(extents)
	            , current(current)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator*() const noexcept -> value_type
	        {
	            return current;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator->() const noexcept -> pointer
	        {
	            return {**this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator++() noexcept -> ArrayIndexIterator&
	        {
	            current[rank - 1]++;
	            for(auto i = static_cast<int>(rank) - 2; i >= 0; i--)
	            {
	                if(current[i + 1] != extents[i + 1])
	                    return *this;
	                current[i + 1] = 0;
	                current[i]++;
	            }
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator++(int) noexcept -> ArrayIndexIterator
	        {
	            auto tmp = *this;
	            ++*this;
	            return tmp;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator--() noexcept -> ArrayIndexIterator&
	        {
	            current[rank - 1]--;
	            for(auto i = static_cast<int>(rank) - 2; i >= 0; i--)
	            {
	                // return if no underflow
	                if(current[i + 1] != static_cast<typename ArrayExtents::value_type>(-1))
	                    return *this;
	                current[i + 1] = extents[i] - 1;
	                current[i]--;
	            }
	            // decrementing beyond [0, 0, ..., 0] is UB
	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator--(int) noexcept -> ArrayIndexIterator
	        {
	            auto tmp = *this;
	            --*this;
	            return tmp;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator[](difference_type i) const noexcept -> reference
	        {
	            return *(*this + i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator+=(difference_type n) noexcept -> ArrayIndexIterator&
	        {
	            // add n to all lower dimensions with carry
	            for(auto i = static_cast<int>(rank) - 1; i > 0 && n != 0; i--)
	            {
	                n += static_cast<difference_type>(current[i]);
	                const auto s = static_cast<difference_type>(extents[i]);
	                auto mod = n % s;
	                n /= s;
	                if(mod < 0)
	                {
	                    mod += s;
	                    n--;
	                }
	                current[i] = mod;
	                assert(current[i] < extents[i]);
	            }

	            current[0] = static_cast<difference_type>(current[0]) + n;
	            // current is either within bounds or at the end ([last + 1, 0, 0, ..., 0])
	            assert(
	                (current[0] < extents[0]
	                 || (current[0] == extents[0]
	                     && std::all_of(std::begin(current) + 1, std::end(current), [](auto c) { return c == 0; })))
	                && "Iterator was moved past the end");

	            return *this;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator+(ArrayIndexIterator it, difference_type n) noexcept -> ArrayIndexIterator
	        {
	            it += n;
	            return it;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator+(difference_type n, ArrayIndexIterator it) noexcept -> ArrayIndexIterator
	        {
	            return it + n;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto operator-=(difference_type n) noexcept -> ArrayIndexIterator&
	        {
	            return operator+=(-n);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator-(ArrayIndexIterator it, difference_type n) noexcept -> ArrayIndexIterator
	        {
	            it -= n;
	            return it;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator-(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept
	            -> difference_type
	        {
	            assert(a.extents == b.extents);

	            difference_type n = a.current[rank - 1] - b.current[rank - 1];
	            difference_type size = a.extents[rank - 1];
	            for(auto i = static_cast<int>(rank) - 2; i >= 0; i--)
	            {
	                n += (a.current[i] - b.current[i]) * size;
	                size *= a.extents[i];
	            }

	            return n;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator==(
	            const ArrayIndexIterator<ArrayExtents>& a,
	            const ArrayIndexIterator<ArrayExtents>& b) noexcept -> bool
	        {
	            assert(a.extents == b.extents);
	            return a.current == b.current;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator!=(
	            const ArrayIndexIterator<ArrayExtents>& a,
	            const ArrayIndexIterator<ArrayExtents>& b) noexcept -> bool
	        {
	            return !(a == b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator<(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
	        {
	            assert(a.extents == b.extents);
	#ifdef __NVCC__
	            // from: https://en.cppreference.com/w/cpp/algorithm/lexicographical_compare
	            auto first1 = std::begin(a.current);
	            auto last1 = std::end(a.current);
	            auto first2 = std::begin(b.current);
	            auto last2 = std::end(b.current);
	            for(; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2)
	            {
	                if(*first1 < *first2)
	                    return true;
	                if(*first2 < *first1)
	                    return false;
	            }

	            return (first1 == last1) && (first2 != last2);
	#else
	            return std::lexicographical_compare(
	                std::begin(a.current),
	                std::end(a.current),
	                std::begin(b.current),
	                std::end(b.current));
	#endif
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator>(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
	        {
	            return b < a;
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator<=(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
	        {
	            return !(a > b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        friend constexpr auto operator>=(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
	        {
	            return !(a < b);
	        }

	    private:
	        ArrayExtents extents; // TODO(bgruber): we only need to store rank - 1 sizes
	        value_type current;
	    };

	    /// Range allowing to iterate over all indices in an \ref ArrayExtents.
	    LLAMA_EXPORT
	    template<typename ArrayExtents>
	    struct ArrayIndexRange
	        : private ArrayExtents
	#if CAN_USE_RANGES
	        , std::ranges::view_base
	#endif
	    {
	        static_assert(!std::is_const_v<ArrayExtents>);

	        constexpr ArrayIndexRange() noexcept = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr explicit ArrayIndexRange(ArrayExtents extents) noexcept : ArrayExtents(extents)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto begin() const noexcept -> ArrayIndexIterator<ArrayExtents>
	        {
	            return {*this, typename ArrayExtents::Index{}};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto end() const noexcept -> ArrayIndexIterator<ArrayExtents>
	        {
	            auto endPos = typename ArrayExtents::Index{};
	            endPos[0] = this->toArray()[0];
	            return {*this, endPos};
	        }
	    };
	} // namespace llama
	// ==
	// == ./include/llama/ArrayIndexRange.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/BlobAllocators.hpp ==
	// ==
	// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "Array.hpp"    // amalgamate: file already inlined
	// #include "Concepts.hpp"    // amalgamate: file already inlined
	// #include "macros.hpp"    // amalgamate: file already inlined

	#include <cstddef>
	// #include <memory>    // amalgamate: file already included
	#include <vector>
	#if __has_include(<cuda_runtime.h>)
	#    include <cuda_runtime.h>
	#endif
	#if __has_include(<sycl/sycl.hpp>)
	#    include <sycl/sycl.hpp>
	#endif

	namespace alpaka
	{
	    template<typename TElem, typename TIdx, typename TExtent, typename TDev>
	    auto allocBuf(const TDev& dev, const TExtent& extent); // NOLINT(readability-redundant-declaration)
	} // namespace alpaka

	namespace llama::bloballoc
	{
	    /// Allocates statically sized memory for a \ref View, which is copied each time a \ref View is copied.
	    /// \tparam BytesToReserve the amount of memory to reserve.
	    LLAMA_EXPORT
	    template<std::size_t BytesToReserve>
	    struct Array
	    {
	        template<std::size_t Alignment>
	        struct alignas(Alignment) AlignedArray : llama::Array<std::byte, BytesToReserve>
	        {
	        };

	        template<std::size_t Alignment>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(
	            std::integral_constant<std::size_t, Alignment>,
	            [[maybe_unused]] std::size_t count) const -> AlignedArray<Alignment>
	        {
	            assert(count == BytesToReserve);
	            return {};
	        }
	    };
	#ifdef __cpp_lib_concepts
	    static_assert(BlobAllocator<Array<64>>);
	#endif

	    /// Allocates heap memory managed by a `std::unique_ptr` for a \ref View. This memory can only be uniquely owned by
	    /// a single \ref View.
	    LLAMA_EXPORT
	    struct UniquePtr
	    {
	        template<std::size_t Alignment>
	        auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
	        {
	            auto* ptr
	                = static_cast<std::byte*>(::operator new[](count * sizeof(std::byte), std::align_val_t{Alignment}));
	            auto deleter = [](std::byte* ptr) { ::operator delete[](ptr, std::align_val_t{Alignment}); };
	            return std::unique_ptr<std::byte[], decltype(deleter)>{ptr, deleter};
	        }
	    };
	#ifdef __cpp_lib_concepts
	    static_assert(BlobAllocator<UniquePtr>);
	#endif

	    /// Allocates heap memory managed by a `std::shared_ptr` for a \ref View. This memory is shared between all copies
	    /// of a \ref View.
	    LLAMA_EXPORT
	    struct SharedPtr
	    {
	        template<std::size_t Alignment>
	        auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
	            -> std::shared_ptr<std::byte[]>
	        {
	            auto* ptr
	                = static_cast<std::byte*>(::operator new[](count * sizeof(std::byte), std::align_val_t{Alignment}));
	            auto deleter = [](std::byte* ptr) { ::operator delete[](ptr, std::align_val_t{Alignment}); };
	            return {ptr, deleter};
	        }
	    };
	#ifdef __cpp_lib_concepts
	    static_assert(BlobAllocator<SharedPtr>);
	#endif

	    /// An STL compatible allocator allowing to specify alignment.
	    LLAMA_EXPORT
	    template<typename T, std::size_t Alignment>
	    struct AlignedAllocator
	    {
	        using value_type = T;

	        inline AlignedAllocator() noexcept = default;

	        template<typename T2>
	        inline explicit AlignedAllocator(const AlignedAllocator<T2, Alignment>&) noexcept
	        {
	        }

	        inline auto allocate(std::size_t n) -> T*
	        {
	            return static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t{Alignment}));
	        }

	        inline void deallocate(T* p, std::size_t)
	        {
	            ::operator delete[](p, std::align_val_t{Alignment});
	        }

	        template<typename T2>
	        struct rebind // NOLINT(readability-identifier-naming)
	        {
	            using other = AlignedAllocator<T2, Alignment>;
	        };

	        auto operator!=(const AlignedAllocator<T, Alignment>& other) const -> bool
	        {
	            return !(*this == other);
	        }

	        auto operator==(const AlignedAllocator<T, Alignment>&) const -> bool
	        {
	            return true;
	        }
	    };

	    /// Allocates heap memory managed by a `std::vector` for a \ref View, which is copied each time a \ref View is
	    /// copied.
	    LLAMA_EXPORT
	    struct Vector
	    {
	        template<std::size_t Alignment>
	        inline auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
	        {
	            return std::vector<std::byte, AlignedAllocator<std::byte, Alignment>>(count);
	        }
	    };
	#ifdef __cpp_lib_concepts
	    static_assert(BlobAllocator<Vector>);
	#endif

	#if __has_include(<cuda_runtime.h>)
	    /// Allocates GPU device memory using cudaMalloc. The memory is managed by a std::unique_ptr with a deleter that
	    /// calles cudaFree. If you want to use a view created with this allocator in a CUDA kernel, call \ref shallowCopy
	    /// on the view before passing it to the kernel.
	    LLAMA_EXPORT
	    struct CudaMalloc
	    {
	        inline static const auto deleter = [](void* p)
	        {
	            if(const auto code = cudaFree(p); code != cudaSuccess)
	                throw std::runtime_error(std::string{"cudaFree failed with code "} + cudaGetErrorString(code));
	        };

	        template<std::size_t FieldAlignment>
	        inline auto operator()(std::integral_constant<std::size_t, FieldAlignment>, std::size_t count) const
	        {
	            std::byte* p = nullptr;
	            if(const auto code = cudaMalloc(&p, count); code != cudaSuccess)
	                throw std::runtime_error(std::string{"cudaMalloc failed with code "} + cudaGetErrorString(code));
	            if(reinterpret_cast<std::uintptr_t>(p) & (FieldAlignment - 1 != 0u))
	                throw std::runtime_error{"cudaMalloc does not align sufficiently"};
	            return std::unique_ptr<std::byte[], decltype(deleter)>(p, deleter);
	        }
	    };
	#endif

	    /// Allocates alpaka buffers as blobs.
	    LLAMA_EXPORT
	    template<typename Size, typename Dev>
	    struct AlpakaBuf
	    {
	        Dev& dev;

	        template<std::size_t Alignment>
	        inline auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
	        {
	            return alpaka::allocBuf<std::byte, Size>(dev, static_cast<Size>(count));
	        }
	    };

	#if __has_include(<sycl/sycl.hpp>)
	    /// Allocates shared USM memory using sycl::aligned_alloc_shared. The memory is managed by a std::unique_ptr with a
	    /// deleter that calles sycl::free. If you want to use a view created with this allocator in a kernel, call \ref
	    /// shallowCopy on the view before passing it to the kernel.
	    LLAMA_EXPORT
	    struct SyclMallocShared
	    {
	        sycl::queue queue;

	        static auto makeDeleter(sycl::queue q)
	        {
	            // create lambda in function independent of FieldAlignment template paramter to avoid different blob types
	            return [q](void* p) { sycl::free(p, q); };
	        }

	        template<std::size_t FieldAlignment>
	        inline auto operator()(std::integral_constant<std::size_t, FieldAlignment>, std::size_t count) const
	        {
	            std::byte* p = sycl::aligned_alloc_shared<std::byte>(FieldAlignment, count, queue);
	            if(reinterpret_cast<std::uintptr_t>(p) & (FieldAlignment - 1 != 0u))
	                throw std::runtime_error{"sycl::malloc_shared does not align sufficiently"};
	            return std::unique_ptr<std::byte[], decltype(makeDeleter(queue))>(p, makeDeleter(queue));
	        }
	    };
	#endif
	} // namespace llama::bloballoc
	// ==
	// == ./include/llama/BlobAllocators.hpp ==
	// ============================================================================

// #include "Concepts.hpp"    // amalgamate: file already inlined
// #include "Core.hpp"    // amalgamate: file already inlined
// #include "macros.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/mapping/One.hpp ==
	// ==
	// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../Core.hpp"    // amalgamate: file already inlined
		// ============================================================================
		// == ./include/llama/mapping/Common.hpp ==
		// ==
		// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
		// SPDX-License-Identifier: MPL-2.0

		// #pragma once
		// #include "../Core.hpp"    // amalgamate: file already inlined

		// #include <atomic>    // amalgamate: file already included
		#include <climits>
		#ifndef __cpp_lib_atomic_ref
		#    include <boost/atomic/atomic_ref.hpp>
		#endif

		namespace llama::mapping
		{
		    LLAMA_EXPORT
		    template<typename TArrayExtents, typename TRecordDim>
		    struct MappingBase : protected TArrayExtents
		    {
		        using ArrayExtents = TArrayExtents;
		        using RecordDim = TRecordDim;

		    protected:
		        using ArrayIndex = typename ArrayExtents::Index;
		        using size_type = typename ArrayExtents::value_type;

		    public:
		        constexpr MappingBase() = default;

		        LLAMA_FN_HOST_ACC_INLINE
		        constexpr explicit MappingBase(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
		        {
		        }

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
		        {
		            return static_cast<const ArrayExtents&>(*this);
		        }
		    };

		    /// Functor that maps an \ref ArrayIndex into linear numbers, where the fast moving index should be the rightmost
		    /// one, which models how C++ arrays work and is analogous to mdspan's layout_right. E.g. ArrayIndex<3> a; stores 3
		    /// indices where a[2] should be incremented in the innermost loop.
		    LLAMA_EXPORT
		    struct LinearizeArrayIndexRight
		    {
		        template<typename ArrayExtents>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayExtents& extents) -> typename ArrayExtents::value_type
		        {
		            return product(extents);
		        }

		        /// \param ai Index in the array dimensions.
		        /// \param extents Total size of the array dimensions.
		        /// \return Linearized index.
		        template<typename ArrayExtents>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
		            const typename ArrayExtents::Index& ai,
		            const ArrayExtents& extents) const -> typename ArrayExtents::value_type
		        {
		            if constexpr(ArrayExtents::rank == 0)
		                return 0;
		            else
		            {
		                auto address = ai[0];
		                for(int i = 1; i < static_cast<int>(ArrayExtents::rank); i++)
		                {
		                    address *= extents[i];
		                    address += ai[i];
		                }
		                return address;
		            }
		        }
		    };

		    LLAMA_EXPORT
		    using LinearizeArrayIndexCpp = LinearizeArrayIndexRight;

		    /// Functor that maps a \ref ArrayIndex into linear numbers the way Fortran arrays work. The fast moving index of
		    /// the ArrayIndex object should be the last one. E.g. ArrayIndex<3> a; stores 3 indices where a[0] should be
		    /// incremented in the innermost loop.
		    LLAMA_EXPORT
		    struct LinearizeArrayIndexLeft
		    {
		        template<typename ArrayExtents>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayExtents& extents) -> typename ArrayExtents::value_type
		        {
		            return product(extents);
		        }

		        /// \param ai Index in the array dimensions.
		        /// \param extents Total size of the array dimensions.
		        /// \return Linearized index.
		        template<typename ArrayExtents>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
		            const typename ArrayExtents::Index& ai,
		            const ArrayExtents& extents) const -> typename ArrayExtents::value_type
		        {
		            if constexpr(ArrayExtents::rank == 0)
		                return 0;
		            else
		            {
		                auto address = ai[ArrayExtents::rank - 1];
		                for(int i = static_cast<int>(ArrayExtents::rank) - 2; i >= 0; i--)
		                {
		                    address *= extents[i];
		                    address += ai[i];
		                }
		                return address;
		            }
		        }
		    };

		    LLAMA_EXPORT
		    using LinearizeArrayIndexFortran = LinearizeArrayIndexLeft;

		    /// Functor that maps an \ref ArrayIndex into linear numbers using the Z-order space filling curve (Morton codes).
		    LLAMA_EXPORT
		    struct LinearizeArrayIndexMorton
		    {
		        template<typename ArrayExtents>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayExtents& extents) const ->
		            typename ArrayExtents::value_type
		        {
		            if constexpr(ArrayExtents::rank == 0)
		                return 0;
		            else
		            {
		                auto longest = extents[0];
		                for(int i = 1; i < static_cast<int>(ArrayExtents::rank); i++)
		                    longest = std::max(longest, extents[i]);
		                const auto longestPO2 = bitCeil(longest);
		                return intPow(longestPO2, static_cast<typename ArrayExtents::value_type>(ArrayExtents::rank));
		            }
		        }

		        /// \param ai Coordinate in the array dimensions.
		        /// \param extents Total size of the array dimensions.
		        /// \return Linearized index.
		        template<typename ArrayExtents>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
		            const typename ArrayExtents::Index& ai,
		            [[maybe_unused]] const ArrayExtents& extents) const -> typename ArrayExtents::value_type
		        {
		            using size_type = typename ArrayExtents::value_type;
		            constexpr auto rank = static_cast<size_type>(ArrayExtents::rank);
		            size_type r = 0;
		            for(size_type bit = 0; bit < (static_cast<size_type>(sizeof(size_type)) * CHAR_BIT) / rank; bit++)
		                for(size_type i = 0; i < rank; i++)
		                    r |= (ai[i] & (size_type{1} << bit)) << ((bit + 1) * (rank - 1) - i);
		            return r;
		        }

		    private:
		        template<typename T>
		        LLAMA_FN_HOST_ACC_INLINE static constexpr auto bitCeil(T n) -> T
		        {
		            T r = 1u;
		            while(r < n)
		                r <<= 1u;
		            return r;
		        }

		        template<typename T>
		        LLAMA_FN_HOST_ACC_INLINE static constexpr auto intPow(T b, T e) -> T
		        {
		            e--;
		            auto r = b;
		            while(e != 0u)
		            {
		                r *= b;
		                e--;
		            }
		            return r;
		        }
		    };

		    /// Retains the order of the record dimension's fields.
		    LLAMA_EXPORT
		    template<typename TFlatRecordDim>
		    struct PermuteFieldsInOrder
		    {
		        using FlatRecordDim = TFlatRecordDim;

		        template<std::size_t FlatRecordCoord>
		        static constexpr std::size_t permute = FlatRecordCoord;
		    };

		    /// Sorts the record dimension's the fields according to a given predicate on the field types.
		    /// @tparam Less A binary predicate accepting two field types, which exposes a member value. Value must be true if
		    /// the first field type is less than the second one, otherwise false.
		    LLAMA_EXPORT
		    template<typename FlatOrigRecordDim, template<typename, typename> typename Less>
		    struct PermuteFieldsSorted
		    {
		    private:
		        using FlatSortedRecordDim = mp_sort<FlatOrigRecordDim, Less>;

		        template<typename A, typename B>
		        using LessWithIndices = Less<mp_at<FlatOrigRecordDim, A>, mp_at<FlatOrigRecordDim, B>>;

		        // A permutation from new FlatSortedRecordDim index to old FlatOrigRecordDim index
		        using PermutedIndices = mp_sort<mp_iota<mp_size<FlatOrigRecordDim>>, LessWithIndices>;

		        template<typename A, typename B>
		        using LessInvertPermutation
		            = std::bool_constant<(mp_at<PermutedIndices, A>::value < mp_at<PermutedIndices, B>::value)>;

		        // A permutation from old FlatOrigRecordDim index to new FlatSortedRecordDim index
		        using InversePermutedIndices = mp_sort<mp_iota<mp_size<FlatOrigRecordDim>>, LessInvertPermutation>;

		    public:
		        using FlatRecordDim = FlatSortedRecordDim;

		        template<std::size_t FlatRecordCoord>
		        static constexpr std::size_t permute = mp_at_c<InversePermutedIndices, FlatRecordCoord>::value;
		    };

		    namespace internal
		    {
		        template<typename A, typename B>
		        using LessAlignment = std::bool_constant<alignof(A) < alignof(B)>;

		        template<typename A, typename B>
		        using MoreAlignment = std::bool_constant<(alignof(A) > alignof(B))>;
		    } // namespace internal

		    /// Sorts the record dimension fields by increasing alignment of its fields.
		    LLAMA_EXPORT
		    template<typename FlatRecordDim>
		    using PermuteFieldsIncreasingAlignment = PermuteFieldsSorted<FlatRecordDim, internal::LessAlignment>;

		    /// Sorts the record dimension fields by decreasing alignment of its fields.
		    LLAMA_EXPORT
		    template<typename FlatRecordDim>
		    using PermuteFieldsDecreasingAlignment = PermuteFieldsSorted<FlatRecordDim, internal::MoreAlignment>;

		    /// Sorts the record dimension fields by the alignment of its fields to minimize padding.
		    LLAMA_EXPORT
		    template<typename FlatRecordDim>
		    using PermuteFieldsMinimizePadding = PermuteFieldsIncreasingAlignment<FlatRecordDim>;

		    namespace internal
		    {
		        template<auto I>
		        struct S;

		        template<typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE void atomicInc(CountType& i)
		        {
		#ifdef __CUDA_ARCH__
		            // if you get an error here that there is no overload of atomicAdd, your CMAKE_CUDA_ARCHITECTURE might be
		            // too low or you need to use a smaller CountType for the FieldAccessCount or Heatmap mapping.
		            if constexpr(mp_contains<mp_list<int, unsigned int, unsigned long long int>, CountType>::value)
		                atomicAdd(&i, CountType{1});
		            else if constexpr(sizeof(CountType) == sizeof(unsigned int))
		                atomicAdd(reinterpret_cast<unsigned int*>(&i), 1u);
		            else if constexpr(sizeof(CountType) == sizeof(unsigned long long int))
		                atomicAdd(reinterpret_cast<unsigned long long int*>(&i), 1ull);
		            else
		                static_assert(sizeof(CountType) == 0, "There is no CUDA atomicAdd for your CountType");
		#elif defined(__cpp_lib_atomic_ref)
		            ++std::atomic_ref<CountType>{i};
		#else
		            ++boost::atomic_ref<CountType>{i};
		#endif
		        }
		    } // namespace internal

		    LLAMA_EXPORT
		    enum class FieldAlignment
		    {
		        Pack,
		        Align
		    };
		} // namespace llama::mapping
		// ==
		// == ./include/llama/mapping/Common.hpp ==
		// ============================================================================


	namespace llama::mapping
	{
	    /// Maps all array dimension indices to the same location and layouts struct members consecutively. This mapping is
	    /// used for temporary, single element views.
	    /// \tparam TFieldAlignment If Align, padding bytes are inserted to guarantee that struct members are properly
	    /// aligned. If false, struct members are tightly packed.
	    /// \tparam PermuteFields Defines how the record dimension's fields should be permuted. See \ref
	    /// PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref PermuteFieldsDecreasingAlignment and
	    /// \ref PermuteFieldsMinimizePadding.
	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        FieldAlignment TFieldAlignment = FieldAlignment::Align,
	        template<typename> typename PermuteFields = PermuteFieldsMinimizePadding>
	    struct One : MappingBase<TArrayExtents, TRecordDim>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using size_type = typename Base::size_type;

	    public:
	        inline static constexpr FieldAlignment fieldAlignment = TFieldAlignment;
	        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
	        static constexpr std::size_t blobCount = 1;

	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ >= 12
	        using Base::Base;
	#else
	        constexpr One() = default;

	        LLAMA_FN_HOST_ACC_INLINE constexpr explicit One(TArrayExtents extents, TRecordDim = {}) : Base(extents)
	        {
	        }
	#endif

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
	        {
	            return flatSizeOf<
	                typename Permuter::FlatRecordDim,
	                fieldAlignment == FieldAlignment::Align,
	                false>; // no tail padding
	        }

	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            typename Base::ArrayIndex,
	            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
	        {
	            constexpr std::size_t flatFieldIndex =
	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6
	                *& // mess with nvcc compiler state to workaround bug
	#endif
	                 Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>;
	            constexpr auto offset = static_cast<size_type>(flatOffsetOf<
	                                                           typename Permuter::FlatRecordDim,
	                                                           flatFieldIndex,
	                                                           fieldAlignment == FieldAlignment::Align>);
	            return {size_type{0}, offset};
	        }
	    };

	    /// One mapping preserving the alignment of the field types by inserting padding.
	    /// \see One
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim>
	    using AlignedOne = One<ArrayExtents, RecordDim, FieldAlignment::Align, PermuteFieldsInOrder>;

	    /// One mapping preserving the alignment of the field types by inserting padding and permuting the field order to
	    /// minimize this padding.
	    /// \see One
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim>
	    using MinAlignedOne = One<ArrayExtents, RecordDim, FieldAlignment::Align, PermuteFieldsMinimizePadding>;

	    /// One mapping packing the field types tightly, violating the types' alignment requirements.
	    /// \see One
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim>
	    using PackedOne = One<ArrayExtents, RecordDim, FieldAlignment::Pack, PermuteFieldsInOrder>;

	    /// Binds parameters to a \ref One mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<
	        FieldAlignment FieldAlignment = FieldAlignment::Align,
	        template<typename> typename PermuteFields = PermuteFieldsMinimizePadding>
	    struct BindOne
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = One<ArrayExtents, RecordDim, FieldAlignment, PermuteFields>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isOne = false;

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        FieldAlignment FieldAlignment,
	        template<typename>
	        typename PermuteFields>
	    inline constexpr bool isOne<One<ArrayExtents, RecordDim, FieldAlignment, PermuteFields>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/One.hpp ==
	// ============================================================================


// #include <type_traits>    // amalgamate: file already included

namespace llama
{
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename TMapping, Blob BlobType, typename TAccessor>
#else
    template<typename TMapping, typename BlobType, typename TAccessor>
#endif
    struct View;

    namespace internal
    {
        template<typename Allocator, typename RecordDim>
        using AllocatorBlobType
            = decltype(std::declval<Allocator>()(std::integral_constant<std::size_t, alignOf<RecordDim>>{}, 0));

        template<typename Allocator, typename Mapping, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeBlobArray(
            const Allocator& alloc,
            const Mapping& mapping,
            std::integer_sequence<std::size_t, Is...>)
            -> Array<AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Mapping::blobCount>
        {
            [[maybe_unused]] constexpr auto alignment
                = alignOf<typename Mapping::RecordDim>; // g++-12 warns that alignment is unused
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return {alloc(std::integral_constant<std::size_t, alignment>{}, mapping.blobSize(Is))...};
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        } // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
    } // namespace internal

    /// Same as \ref allocView but does not run field constructors.
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#else
    template<typename Mapping, typename Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocViewUninitialized(
        Mapping mapping = {},
        const Allocator& alloc = {},
        Accessor accessor = {})
    {
        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
        return View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Accessor>{
            std::move(mapping),
            std::move(blobs),
            std::move(accessor)};
    }

    namespace internal
    {
        template<typename Mapping, typename RecordCoord, typename = void>
        struct IsComputed : std::false_type
        {
        };

        template<typename Mapping, typename RecordCoord>
        struct IsComputed<Mapping, RecordCoord, std::void_t<decltype(Mapping::isComputed(RecordCoord{}))>>
            : std::bool_constant<Mapping::isComputed(RecordCoord{})>
        {
        };
    } // namespace internal

    /// Returns true if the field accessed via the given mapping and record coordinate is a computed value.
    LLAMA_EXPORT
    template<typename Mapping, typename RecordCoord>
    inline constexpr bool isComputed = internal::IsComputed<Mapping, RecordCoord>::value;

    /// Returns true if any field accessed via the given mapping is a computed value.
    // TODO(bgruber): harmonize this with LLAMA's concepts from Concepts.hpp
    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool hasAnyComputedField = mp_any_of<
        LeafRecordCoords<typename Mapping::RecordDim>,
        mp_bind_front<internal::IsComputed, Mapping>::template fn>::value;

    LLAMA_EXPORT
    template<typename Mapping, typename BlobType, typename Accessor, std::size_t... RCs>
    LLAMA_FN_HOST_ACC_INLINE void constructField(
        View<Mapping, BlobType, Accessor>& view,
        typename Mapping::ArrayExtents::Index ai,
        RecordCoord<RCs...> rc)
    {
        using FieldType = GetType<typename Mapping::RecordDim, decltype(rc)>;

        // this handles physical and computed mappings
        if constexpr(sizeof...(RCs) == 0)
        {
            using RefType = decltype(view(ai));
            if constexpr(isProxyReference<RefType>)
            {
                view(ai) = FieldType{};
            }
            else if constexpr(
                std::is_lvalue_reference_v<RefType> && !std::is_const_v<std::remove_reference_t<RefType>>)
            {
                new(&view(ai)) FieldType{};
            }
        }
        else
        {
            using RefType = decltype(view(ai)(rc));
            if constexpr(isProxyReference<RefType>)
            {
                view(ai)(rc) = FieldType{};
            }
            else if constexpr(
                std::is_lvalue_reference_v<RefType> && !std::is_const_v<std::remove_reference_t<RefType>>)
            {
                new(&view(ai)(rc)) FieldType{};
            }
        }
    }

    /// Value-initializes all fields reachable through the given view. That is, constructors are run and fundamental
    /// types are zero-initialized. Computed fields are constructed if they return l-value references and assigned a
    /// default constructed value if they return a proxy reference.
    LLAMA_EXPORT
    template<typename Mapping, typename BlobType, typename Accessor>
    LLAMA_FN_HOST_ACC_INLINE void constructFields(View<Mapping, BlobType, Accessor>& view)
    {
        using View = View<Mapping, BlobType, Accessor>;
        using RecordDim = typename View::RecordDim;
        forEachArrayIndex(
            view.extents(),
            [&](typename View::ArrayIndex ai) LLAMA_LAMBDA_INLINE
            { forEachLeafCoord<RecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE { constructField(view, ai, rc); }); });
    }

    /// Creates a view based on the given mapping, e.g. \ref AoS or \ref :SoA. For allocating the view's underlying
    /// memory, the specified allocator callable is used (or the default one, which is \ref bloballoc::Vector). The
    /// allocator callable is called with the alignment and size of bytes to allocate for each blob of the mapping.
    /// Value-initialization is performed for all fields by calling \ref constructFields. This function is the
    /// preferred way to create a \ref View. See also \ref allocViewUninitialized.
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#else
    template<typename Mapping, typename Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {}, Accessor accessor = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Accessor>
    {
        auto view = allocViewUninitialized(std::move(mapping), alloc, std::move(accessor));
        constructFields(view);
        return view;
    }

    /// Same as \ref allocScalarView but does not run field constructors.
    LLAMA_EXPORT
    template<std::size_t Dim, typename RecordDim>
    LLAMA_FN_HOST_ACC_INLINE auto allocScalarViewUninitialized() -> decltype(auto)
    {
        constexpr auto mapping = mapping::MinAlignedOne<ArrayExtentsNCube<int, Dim, 1>, RecordDim>{};
        return allocViewUninitialized(mapping, bloballoc::Array<mapping.blobSize(0)>{});
    }

    /// Allocates a \ref View holding a single record backed by a byte array (\ref bloballoc::Array).
    /// \tparam Dim Dimension of the \ref ArrayExtents of the \ref View.
    LLAMA_EXPORT
    template<std::size_t Dim, typename RecordDim>
    LLAMA_FN_HOST_ACC_INLINE auto allocScalarView() -> decltype(auto)
    {
        auto view = allocScalarViewUninitialized<Dim, RecordDim>();
        constructFields(view);
        return view;
    }

    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord = RecordCoord<>, bool OwnView = false>
    struct RecordRef;

    /// A \ref RecordRef that owns and holds a single value.
    LLAMA_EXPORT
    template<typename RecordDim>
    using One = RecordRef<decltype(allocScalarView<0, RecordDim>()), RecordCoord<>, true>;

    /// Is true, if T is an instance of \ref One.
    LLAMA_EXPORT
    template<typename T>
    inline constexpr bool isOne = false;

    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord>
    inline constexpr bool isOne<RecordRef<View, BoundRecordCoord, true>> = true;

    // TODO(bgruber): Higher dimensional iterators might not have good codegen. Multiple nested loops seem to be
    // superior to a single iterator over multiple dimensions. At least compilers are able to produce better code.
    // std::mdspan also discovered similar difficulties and there was a discussion in WG21 in Oulu 2016 to
    // remove/postpone iterators from the design. In std::mdspan's design, the iterator iterated over the co-domain.
    LLAMA_EXPORT
    template<typename View>
    struct Iterator
    {
        using ArrayIndexIterator = llama::ArrayIndexIterator<typename View::ArrayExtents>;

        using iterator_category = std::random_access_iterator_tag;
        using value_type = One<typename View::RecordDim>;
        using difference_type = typename ArrayIndexIterator::difference_type;
        using pointer = internal::IndirectValue<RecordRef<View>>;
        using reference = RecordRef<View>;

        constexpr Iterator() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr Iterator(ArrayIndexIterator arrayIndex, View* view)
            : arrayIndex(arrayIndex)
            , view(view)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++() -> Iterator&
        {
            ++arrayIndex;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++(int) -> Iterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--() -> Iterator&
        {
            --arrayIndex;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--(int) -> Iterator
        {
            auto tmp{*this};
            --*this;
            return tmp;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator*() const -> reference
        {
            return (*view)(*arrayIndex);
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator->() const -> pointer
        {
            return {**this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator[](difference_type i) const -> reference
        {
            return *(*this + i);
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator+=(difference_type n) -> Iterator&
        {
            arrayIndex += n;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(Iterator it, difference_type n) -> Iterator
        {
            it += n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(difference_type n, Iterator it) -> Iterator
        {
            return it + n;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator-=(difference_type n) -> Iterator&
        {
            arrayIndex -= n;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(Iterator it, difference_type n) -> Iterator
        {
            it -= n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(const Iterator& a, const Iterator& b) -> difference_type
        {
            assert(a.view == b.view);
            return static_cast<std::ptrdiff_t>(a.arrayIndex - b.arrayIndex);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator==(const Iterator& a, const Iterator& b) -> bool
        {
            assert(a.view == b.view);
            return a.arrayIndex == b.arrayIndex;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator!=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a == b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<(const Iterator& a, const Iterator& b) -> bool
        {
            assert(a.view == b.view);
            return a.arrayIndex < b.arrayIndex;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>(const Iterator& a, const Iterator& b) -> bool
        {
            return b < a;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a > b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a < b);
        }

        ArrayIndexIterator arrayIndex;
        View* view;
    };

    /// Using a mapping, maps the given array index and record coordinate to a memory reference onto the given blobs.
    /// \return Either an l-value reference if the record coord maps to a physical field or a proxy reference if mapped
    /// to a computed field.
    LLAMA_EXPORT
    template<typename Mapping, typename RecordCoord, typename Blobs>
    LLAMA_FN_HOST_ACC_INLINE auto mapToMemory(
        Mapping& mapping,
        typename Mapping::ArrayExtents::Index ai,
        RecordCoord rc,
        Blobs& blobs) -> decltype(auto)
    {
        if constexpr(llama::isComputed<Mapping, RecordCoord>)
            return mapping.compute(ai, rc, blobs);
        else
        {
            const auto [nr, offset] = mapping.blobNrAndOffset(ai, rc);
            using Type = GetType<typename Mapping::RecordDim, RecordCoord>;
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return reinterpret_cast<CopyConst<std::remove_reference_t<decltype(blobs[nr][offset])>, Type>&>(
                blobs[nr][offset]);
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
    }

    /// Central LLAMA class holding memory for storage and giving access to values stored there defined by a
    /// mapping. A view should be created using \ref allocView. \tparam TMapping The mapping used by the view to
    /// map accesses into memory. \tparam TBlobType The storage type used by the view holding memory. \tparam
    /// TAccessor The accessor to use when an access is made through this view.
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename TMapping, Blob TBlobType, typename TAccessor = accessor::Default>
#else
    template<typename TMapping, typename TBlobType, typename TAccessor = accessor::Default>
#endif
    struct LLAMA_DECLSPEC_EMPTY_BASES View
        : private TMapping
        , private TAccessor
#if CAN_USE_RANGES
        , std::ranges::view_base
#endif
    {
        static_assert(!std::is_const_v<TMapping>);
        using Mapping = TMapping;
        using BlobType = TBlobType;
        using ArrayExtents = typename Mapping::ArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = typename Mapping::RecordDim;
        using Accessor = TAccessor;
        using iterator = Iterator<View>;
        using const_iterator = Iterator<const View>;

    private:
        using size_type = typename ArrayExtents::value_type;

    public:
        static_assert(
            std::is_same_v<Mapping, std::decay_t<Mapping>>,
            "Mapping must not be const qualified or a reference. Are you using decltype(...) as View template "
            "argument?");
        static_assert(
            std::is_same_v<ArrayExtents, std::decay_t<ArrayExtents>>,
            "Mapping::ArrayExtents must not be const qualified or a reference. Are you using decltype(...) as "
            "mapping "
            "template argument?");

        /// Performs default initialization of the blob array.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
        View() = default;

        /// Creates a LLAMA View manually. Prefer the allocations functions \ref allocView and \ref
        /// allocViewUninitialized if possible.
        /// \param mapping The mapping used by the view to map accesses into memory.
        /// \param blobs An array of blobs providing storage space for the mapped data.
        /// \param accessor The accessor to use when an access is made through this view.
        LLAMA_FN_HOST_ACC_INLINE
        explicit View(Mapping mapping, Array<BlobType, Mapping::blobCount> blobs = {}, Accessor accessor = {})
            : Mapping(std::move(mapping))
            , Accessor(std::move(accessor))
            , m_blobs(std::move(blobs))
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() -> Mapping&
        {
            return static_cast<Mapping&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() const -> const Mapping&
        {
            return static_cast<const Mapping&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE auto accessor() -> Accessor&
        {
            return static_cast<Accessor&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE auto accessor() const -> const Accessor&
        {
            return static_cast<const Accessor&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE auto extents() const -> ArrayExtents
        {
            return mapping().extents();
        }

#if !(defined(_MSC_VER) && defined(__NVCC__))
        template<typename V>
        auto operator()(llama::ArrayIndex<V, ArrayIndex::rank>) const
        {
            static_assert(!sizeof(V), "Passed ArrayIndex with SizeType different than Mapping::ArrayExtent");
        }
#endif

        /// Retrieves the \ref RecordRef at the given \ref ArrayIndex index.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) const -> decltype(auto)
        {
            if constexpr(isRecordDim<RecordDim>)
                return RecordRef<const View>{ai, *this};
            else
                return access(ai, RecordCoord<>{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
        {
            if constexpr(isRecordDim<RecordDim>)
                return RecordRef<View>{ai, *this};
            else
                return access(ai, RecordCoord<>{});
        }

        /// Retrieves the \ref RecordRef at the \ref ArrayIndex index constructed from the passed component
        /// indices.
        template<
            typename... Indices,
            std::enable_if_t<std::conjunction_v<std::is_convertible<Indices, size_type>...>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            return (*this)(ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...});
        }

        template<
            typename... Indices,
            std::enable_if_t<std::conjunction_v<std::is_convertible<Indices, size_type>...>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            return (*this)(ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...});
        }

        /// Retrieves the \ref RecordRef at the \ref ArrayIndex index constructed from the passed component
        /// indices.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) const -> decltype(auto)
        {
            return (*this)(ai);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) -> decltype(auto)
        {
            return (*this)(ai);
        }

#if !(defined(_MSC_VER) && defined(__NVCC__))
        template<typename V>
        auto operator[](llama::ArrayIndex<V, ArrayIndex::rank>) const
        {
            static_assert(!sizeof(V), "Passed ArrayIndex with SizeType different than Mapping::ArrayExtent");
        }
#endif

        /// Retrieves the \ref RecordRef at the 1D \ref ArrayIndex index constructed from the passed index.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type index) const -> decltype(auto)
        {
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type index) -> decltype(auto)
        {
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() -> iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() const -> const_iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() -> iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.end(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() const -> const_iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.end(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto blobs() -> Array<BlobType, Mapping::blobCount>&
        {
            return m_blobs;
        }

        LLAMA_FN_HOST_ACC_INLINE auto blobs() const -> const Array<BlobType, Mapping::blobCount>&
        {
            return m_blobs;
        }

    private:
        template<typename TView, typename TBoundRecordCoord, bool OwnView>
        friend struct RecordRef;

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto access(ArrayIndex ai, RecordCoord<Coords...> rc = {}) const -> decltype(auto)
        {
            return accessor()(mapToMemory(mapping(), ai, rc, m_blobs));
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto access(ArrayIndex ai, RecordCoord<Coords...> rc = {}) -> decltype(auto)
        {
            return accessor()(mapToMemory(mapping(), ai, rc, m_blobs));
        }

        Array<BlobType, Mapping::blobCount> m_blobs;
    };

    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename View>
    inline constexpr auto isView = AnyView<View>;
#else
    template<typename View>
    inline constexpr auto isView = false;

    // this definition does neither capture SubView nor user defined view's, but the issue resolves itself with a C++20
    // upgrade.
    template<typename Mapping, typename BlobType, typename Accessor>
    inline constexpr auto isView<View<Mapping, BlobType, Accessor>> = true;
#endif

    namespace internal
    {
        // remove in C++23, from: https://en.cppreference.com/w/cpp/utility/forward_like
        // NOLINTBEGIN
        template<class T, class U>
        [[nodiscard]] LLAMA_FN_HOST_ACC_INLINE constexpr auto&& forward_like(U&& x) noexcept
        {
            constexpr bool is_adding_const = std::is_const_v<std::remove_reference_t<T>>;
            if constexpr(std::is_lvalue_reference_v<T&&>)
            {
                if constexpr(is_adding_const)
                    return std::as_const(x);
                else
                    return static_cast<U&>(x);
            }
            else
            {
                if constexpr(is_adding_const)
                    return std::move(std::as_const(x));
                else
                    return std::move(x);
            }
        }
        // NOLINTEND

        template<typename Blobs, typename TransformBlobFunc, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeTransformedBlobArray(
            Blobs&& blobs,
            const TransformBlobFunc& transformBlob,
            std::integer_sequence<std::size_t, Is...>)
        {
            return llama::Array{transformBlob(forward_like<Blobs>(blobs[Is]))...};
        }
    } // namespace internal

    /// Applies the given transformation to the blobs of a view and creates a new view with the transformed blobs
    /// and the same mapping and accessor as the old view.
    LLAMA_EXPORT
    template<typename ViewFwd, typename TransformBlobFunc, typename = std::enable_if_t<isView<std::decay_t<ViewFwd>>>>
    LLAMA_FN_HOST_ACC_INLINE auto transformBlobs(ViewFwd&& view, const TransformBlobFunc& transformBlob)
    {
        using View = std::decay_t<ViewFwd>;
        constexpr auto blobCount = View::Mapping::blobCount;

        auto blobs = internal::makeTransformedBlobArray(
            internal::forward_like<ViewFwd>(view.blobs()),
            transformBlob,
            std::make_index_sequence<blobCount>{});
        return llama::View<typename View::Mapping, typename decltype(blobs)::value_type, typename View::Accessor>{
            view.mapping(),
            std::move(blobs),
            view.accessor()};
    }

    /// Creates a shallow copy of a view. This copy must not outlive the view, since it references its blob array.
    /// \tparam NewBlobType The blob type of the shallow copy. Must be a non owning pointer like type.
    /// \return A new view with the same mapping as view, where each blob refers to the blob in view.
    LLAMA_EXPORT
    template<
        typename View,
        typename NewBlobType = CopyConst<std::remove_reference_t<View>, std::byte>*,
        typename = std::enable_if_t<isView<std::decay_t<View>>>>
    LLAMA_FN_HOST_ACC_INLINE auto shallowCopy(View&& view)
    {
        if constexpr(std::is_same_v<typename std::decay_t<View>::BlobType, NewBlobType>)
            return view;
        else
            return transformBlobs(
                std::forward<View>(view),
                [](auto& blob)
                {
                    LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                    return static_cast<NewBlobType>(&blob[0]);
                    LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
                });
    }

    // Creates a new view from an existing view with the given accessor.
    // \param view A view which's mapping and blobs are forwarded into a new view with the different accessor.
    LLAMA_EXPORT
    template<typename NewAccessor, typename ViewFwd, typename = std::enable_if_t<isView<std::decay_t<ViewFwd>>>>
    LLAMA_FN_HOST_ACC_INLINE auto withAccessor(ViewFwd&& view, NewAccessor newAccessor = {})
    {
        using OldView = std::decay_t<ViewFwd>;
        return View<typename OldView::Mapping, typename OldView::BlobType, NewAccessor>{
            internal::forward_like<ViewFwd>(view.mapping()),
            internal::forward_like<ViewFwd>(view.blobs()),
            std::move(newAccessor)};
    }

    // Creates a new view from an existing view with the given mapping.
    // \param view A view which's accessor and blobs are forwarded into a new view with the different mapping.
    LLAMA_EXPORT
    template<typename NewMapping, typename ViewFwd, typename = std::enable_if_t<isView<std::decay_t<ViewFwd>>>>
    LLAMA_FN_HOST_ACC_INLINE auto withMapping(ViewFwd&& view, NewMapping newMapping = {})
    {
        using OldView = std::decay_t<ViewFwd>;
        static_assert(OldView::Mapping::blobCount == NewMapping::blobCount);
        for(std::size_t i = 0; i < NewMapping::blobCount; i++)
        {
            assert(view.mapping().blobSize(i) == newMapping.blobSize(i));
        }

        return View<NewMapping, typename OldView::BlobType, typename OldView::Accessor>{
            std::move(newMapping),
            internal::forward_like<ViewFwd>(view.blobs()),
            internal::forward_like<ViewFwd>(view.accessor())};
    }

    /// Like a \ref View, but array indices are shifted.
    /// @tparam TStoredParentView Type of the underlying view. May be cv qualified and/or a reference type.
    LLAMA_EXPORT
    template<typename TStoredParentView>
    struct SubView
    {
        using StoredParentView = TStoredParentView;
        using ParentView = std::remove_const_t<std::remove_reference_t<StoredParentView>>; ///< type of the parent view

        using Mapping = typename ParentView::Mapping;
        using ArrayExtents = typename ParentView::ArrayExtents;
        using ArrayIndex = typename ParentView::ArrayIndex;
        using BlobType = typename ParentView::BlobType;
        using RecordDim = typename ParentView::RecordDim;
        using Accessor = typename ParentView::Accessor;
        using iterator = typename ParentView::iterator;
        using const_iterator = typename ParentView::const_iterator;

    private:
        using size_type = typename ArrayExtents::value_type;

    public:
        /// Creates a SubView given an offset. The parent view is default constructed.
        LLAMA_FN_HOST_ACC_INLINE explicit SubView(ArrayIndex offset) : offset(offset)
        {
        }

        /// Creates a SubView given a parent \ref View and offset.
        template<typename StoredParentViewFwd>
        LLAMA_FN_HOST_ACC_INLINE SubView(StoredParentViewFwd&& parentView, ArrayIndex offset)
            : parentView(std::forward<StoredParentViewFwd>(parentView))
            , offset(offset)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() -> Mapping&
        {
            return parentView.mapping();
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() const -> const Mapping&
        {
            return parentView.mapping();
        }

        LLAMA_FN_HOST_ACC_INLINE auto accessor() -> Accessor&
        {
            return parentView.accessor();
        }

        LLAMA_FN_HOST_ACC_INLINE auto accessor() const -> const Accessor&
        {
            return parentView.accessor();
        }

        LLAMA_FN_HOST_ACC_INLINE auto extents() const -> ArrayExtents
        {
            return parentView.extents();
        }

        /// Same as \ref View::operator()(ArrayIndex), but shifted by the offset of this \ref SubView.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) const -> decltype(auto)
        {
            return parentView(ArrayIndex{ai + offset});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
        {
            return parentView(ArrayIndex{ai + offset});
        }

        /// Same as corresponding operator in \ref View, but shifted by the offset of this \ref SubView.
        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, size_type>...>,
                "Indices must be convertible to ArrayExtents::size_type");
            return parentView(
                ArrayIndex{ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...} + offset});
        }

        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, size_type>...>,
                "Indices must be convertible to ArrayExtents::size_type");
            return parentView(
                ArrayIndex{ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...} + offset});
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coords...> rc = {}) const -> decltype(auto)
        {
            return parentView(ArrayIndex{} + offset, rc);
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coords...> rc = {}) -> decltype(auto)
        {
            return parentView(ArrayIndex{} + offset, rc);
        }

        // TODO(bgruber): implement iterators. Those would be transform iterators on top of the parent view's
        // iterators, applying the offset on access.

        LLAMA_FN_HOST_ACC_INLINE auto blobs() -> Array<BlobType, Mapping::blobCount>&
        {
            return parentView.blobs();
        }

        LLAMA_FN_HOST_ACC_INLINE auto blobs() const -> const Array<BlobType, Mapping::blobCount>&
        {
            return parentView.blobs();
        }

        StoredParentView parentView;
        const ArrayIndex offset; ///< offset by which this view's \ref ArrayIndex indices are shifted when passed
                                 ///< to the parent view.
    };

    /// SubView vview(view); will store a reference to view.
    /// SubView vview(std::move(view)); will store the view.
    LLAMA_EXPORT
    template<typename TStoredParentView>
    SubView(TStoredParentView&&, typename std::remove_reference_t<TStoredParentView>::Mapping::ArrayExtents::Index)
        -> SubView<TStoredParentView>;
} // namespace llama
// ==
// == ./include/llama/View.hpp ==
// ============================================================================

// ============================================================================
// == ./include/llama/Tuple.hpp ==
// ==
// Copyright 2023 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// #pragma once
// #include "Meta.hpp"    // amalgamate: file already inlined
// #include "macros.hpp"    // amalgamate: file already inlined

namespace llama
{
    namespace internal
    {
        template<std::size_t I, typename T, bool = std::is_empty_v<T> && !std::is_final_v<T>>
        struct LLAMA_DECLSPEC_EMPTY_BASES TupleLeaf
        {
            T val;

            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() -> T&
            {
                return val;
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() const -> const T&
            {
                return val;
            }
        };

        template<std::size_t I, typename T>
        struct LLAMA_DECLSPEC_EMPTY_BASES TupleLeaf<I, T, true>
        {
            static_assert(!std::is_reference_v<T>, "llama::Tuple cannot store references to stateless types");

            LLAMA_FN_HOST_ACC_INLINE constexpr explicit TupleLeaf(T)
            {
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() const -> T
            {
                return {};
            }
        };
    } // namespace internal

    LLAMA_EXPORT
    template<typename... Elements>
    struct LLAMA_DECLSPEC_EMPTY_BASES Tuple
    {
    };

    /// Tuple class like `std::tuple` but suitable for use with offloading devices like GPUs.
    LLAMA_EXPORT
    template<typename TFirstElement, typename... RestElements>
    struct LLAMA_DECLSPEC_EMPTY_BASES Tuple<TFirstElement, RestElements...>
        : internal::TupleLeaf<1 + sizeof...(RestElements), TFirstElement>
        , Tuple<RestElements...>
    {
    private:
        using Leaf = internal::TupleLeaf<1 + sizeof...(RestElements), TFirstElement>;

    public:
        using FirstElement = TFirstElement;
        using RestTuple = Tuple<RestElements...>;

        constexpr Tuple() = default;

        /// Construct a tuple from values of the same types as the tuple stores.
        LLAMA_FN_HOST_ACC_INLINE constexpr explicit Tuple(FirstElement first, RestElements... rest)
            : Leaf{std::move(first)}
            , RestTuple(std::move(rest)...)
        {
        }

        /// Construct a tuple from forwarded values of potentially different types as the tuple stores.
        // SFINAE away this ctor if tuple elements cannot be constructed from ctor arguments
        template<
            typename T,
            typename... Ts,
            std::enable_if_t<
                sizeof...(RestElements) == sizeof...(Ts)
                    && std::is_constructible_v<FirstElement, T> && (std::is_constructible_v<RestElements, Ts> && ...),
                int>
            = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr explicit Tuple(T&& firstArg, Ts&&... restArgs)
            : Leaf{static_cast<FirstElement>(std::forward<T>(firstArg))}
            , RestTuple(std::forward<Ts>(restArgs)...)
        {
        }

        /// Returns the first element of the tuple
        LLAMA_FN_HOST_ACC_INLINE constexpr auto first() -> decltype(auto)
        {
            return Leaf::value();
        }

        /// Returns the first element of the tuple
        LLAMA_FN_HOST_ACC_INLINE constexpr auto first() const -> decltype(auto)
        {
            return Leaf::value();
        }

        /// Returns a tuple of all but the first element
        LLAMA_FN_HOST_ACC_INLINE constexpr auto rest() -> RestTuple&
        {
            return static_cast<RestTuple&>(*this);
        }

        /// Returns a tuple of all but the first element
        LLAMA_FN_HOST_ACC_INLINE constexpr auto rest() const -> const RestTuple&
        {
            return static_cast<const RestTuple&>(*this);
        }
    };

    LLAMA_EXPORT
    template<typename... Elements>
    LLAMA_HOST_ACC Tuple(Elements...) -> Tuple<std::remove_cv_t<std::remove_reference_t<Elements>>...>;

    LLAMA_EXPORT
    template<std::size_t I, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto get(Tuple<Elements...>& tuple) -> auto&
    {
        using Base [[maybe_unused]] // clang claims Base is unused ...
        = internal::TupleLeaf<sizeof...(Elements) - I, mp_at_c<llama::Tuple<Elements...>, I>>;
        return tuple.Base::value();
    }

    LLAMA_EXPORT
    template<std::size_t I, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto get(const Tuple<Elements...>& tuple) -> const auto&
    {
        using Base [[maybe_unused]] // clang claims Base is unused ...
        = internal::TupleLeaf<sizeof...(Elements) - I, mp_at_c<llama::Tuple<Elements...>, I>>;
        return tuple.Base::value();
    }
} // namespace llama

LLAMA_EXPORT
template<typename... Elements>
struct std::tuple_size<llama::Tuple<Elements...>> // NOLINT(cert-dcl58-cpp)
{
    static constexpr auto value = sizeof...(Elements);
};

LLAMA_EXPORT
template<std::size_t I, typename... Elements>
struct std::tuple_element<I, llama::Tuple<Elements...>> // NOLINT(cert-dcl58-cpp)
{
    using type = boost::mp11::mp_at_c<llama::Tuple<Elements...>, I>;
};

namespace llama
{
    namespace internal
    {
        template<typename... Elements, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto areEqual(
            const Tuple<Elements...>& a,
            const Tuple<Elements...>& b,
            std::index_sequence<Is...>) -> bool
        {
            return ((get<Is>(a) == get<Is>(b)) && ...);
        }
    } // namespace internal

    LLAMA_EXPORT
    template<typename... ElementsA, typename... ElementsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(const Tuple<ElementsA...>& a, const Tuple<ElementsB...>& b)
        -> bool
    {
        if constexpr(sizeof...(ElementsA) == sizeof...(ElementsB))
            if constexpr(mp_apply<mp_all, mp_transform<std::is_same, mp_list<ElementsA...>, mp_list<ElementsB...>>>::
                             value)
                return internal::areEqual(a, b, std::make_index_sequence<sizeof...(ElementsA)>{});
        return false;
    }

    LLAMA_EXPORT
    template<typename... ElementsA, typename... ElementsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(const Tuple<ElementsA...>& a, const Tuple<ElementsB...>& b)
        -> bool
    {
        return !(a == b);
    }

    namespace internal
    {
        template<typename Tuple1, typename Tuple2, size_t... Is1, size_t... Is2>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleCatImpl(
            const Tuple1& t1,
            const Tuple2& t2,
            std::index_sequence<Is1...>,
            std::index_sequence<Is2...>)
        {
            return Tuple{get<Is1>(t1)..., get<Is2>(t2)...};
        }
    } // namespace internal

    LLAMA_EXPORT
    template<typename Tuple1, typename Tuple2>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleCat(const Tuple1& t1, const Tuple2& t2)
    {
        return internal::tupleCatImpl(
            t1,
            t2,
            std::make_index_sequence<std::tuple_size_v<Tuple1>>{},
            std::make_index_sequence<std::tuple_size_v<Tuple2>>{});
    }

    namespace internal
    {
        template<
            std::size_t Pos,
            typename Tuple,
            typename Replacement,
            std::size_t... IsBefore,
            std::size_t... IsAfter>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleReplaceImpl(
            Tuple&& tuple,
            Replacement&& replacement,
            std::index_sequence<IsBefore...>,
            std::index_sequence<IsAfter...>)
        {
            return llama::Tuple{
                get<IsBefore>(std::forward<Tuple>(tuple))...,
                std::forward<Replacement>(replacement),
                get<Pos + 1 + IsAfter>(std::forward<Tuple>(tuple))...};
        }
    } // namespace internal

    /// Creates a copy of a tuple with the element at position Pos replaced by replacement.
    LLAMA_EXPORT
    template<std::size_t Pos, typename Tuple, typename Replacement>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleReplace(Tuple&& tuple, Replacement&& replacement)
    {
        return internal::tupleReplaceImpl<Pos>(
            std::forward<Tuple>(tuple),
            std::forward<Replacement>(replacement),
            std::make_index_sequence<Pos>{},
            std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>> - Pos - 1>{});
    }

    namespace internal
    {
        template<size_t... Is, typename... Elements, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleTransformHelper(
            std::index_sequence<Is...>,
            const Tuple<Elements...>& tuple,
            const Functor& functor)
        {
            return Tuple{functor(get<Is>(tuple))...};
        }
    } // namespace internal

    /// Applies a functor to every element of a tuple, creating a new tuple with the result of the element
    /// transformations. The functor needs to implement a template `operator()` to which all tuple elements are passed.
    LLAMA_EXPORT
    template<typename... Elements, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleTransform(const Tuple<Elements...>& tuple, const Functor& functor)
    {
        // note: cannot use mp11::tuple_transform since that returns a std::tuple
        return internal::tupleTransformHelper(std::make_index_sequence<sizeof...(Elements)>{}, tuple, functor);
    }

    /// Returns a copy of the tuple without the first element.
    LLAMA_EXPORT
    template<typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto popFront(const Tuple<Elements...>& tuple)
    {
        return tuple.rest();
    }
} // namespace llama
// ==
// == ./include/llama/Tuple.hpp ==
// ============================================================================

// ============================================================================
// == ./include/llama/RecordRef.hpp ==
// ==
// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// #pragma once
// #include "Concepts.hpp"    // amalgamate: file already inlined
// #include "ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/StructName.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "Core.hpp"    // amalgamate: file already inlined

	// #include <stdexcept>    // amalgamate: file already included
	#include <string_view>

	namespace llama
	{
	    namespace internal
	    {
	        // TODO(bgruber): just use std::copy which became constexpr in C++20
	        template<typename In, typename Out>
	        constexpr auto constexprCopy(In f, In l, Out d) -> Out
	        {
	            while(f != l)
	                *d++ = *f++;
	            return d;
	        }

	        // TODO(bgruber): just use std::search which became constexpr in C++20
	        // from: https://en.cppreference.com/w/cpp/algorithm/search
	        template<class ForwardIt1, class ForwardIt2>
	        constexpr auto constexprSearch(ForwardIt1 first, ForwardIt1 last, ForwardIt2 sFirst, ForwardIt2 sLast)
	            -> ForwardIt1
	        {
	            while(true)
	            {
	                ForwardIt1 it = first;
	                for(ForwardIt2 sIt = sFirst;; ++it, ++sIt)
	                {
	                    if(sIt == sLast)
	                        return first;
	                    if(it == last)
	                        return last;
	                    if(!(*it == *sIt))
	                        break;
	                }
	                ++first;
	            }
	        }

	        // TODO(bgruber): just use std::remove_copy which became constexpr in C++20
	        // from: https://en.cppreference.com/w/cpp/algorithm/remove_copy
	        template<class InputIt, class OutputIt, class T>
	        constexpr auto constexprRemoveCopy(InputIt first, InputIt last, OutputIt d_first, const T& value) -> OutputIt
	        {
	            for(; first != last; ++first)
	            {
	                if(!(*first == value))
	                {
	                    *d_first++ = *first;
	                }
	            }
	            return d_first;
	        }

	        // TODO(bgruber): just use std::count which became constexpr in C++20
	        // from: https://en.cppreference.com/w/cpp/algorithm/count
	        template<class InputIt, class T>
	        auto constexprCount(InputIt first, InputIt last, const T& value) ->
	            typename std::iterator_traits<InputIt>::difference_type
	        {
	            typename std::iterator_traits<InputIt>::difference_type ret = 0;
	            for(; first != last; ++first)
	            {
	                if(*first == value)
	                {
	                    ret++;
	                }
	            }
	            return ret;
	        }

	        template<std::size_t NewSize, typename T, std::size_t N>
	        constexpr auto resizeArray(Array<T, N> a)
	        {
	            Array<char, NewSize> r{};
	            constexprCopy(a.begin(), a.begin() + NewSize, r.begin());
	            return r;
	        }

	        template<typename T>
	        constexpr auto typeNameAsArray()
	        {
	            // adapted from Matthew Rodusek:
	            // https://bitwizeshift.github.io/posts/2021/03/09/getting-an-unmangled-type-name-at-compile-time/
	            //
	            // Boost Software License - Version 1.0 - August 17th, 2003
	            //
	            // Permission is hereby granted, free of charge, to any person or organization
	            // obtaining a copy of the software and accompanying documentation covered by
	            // this license (the "Software") to use, reproduce, display, distribute,
	            // execute, and transmit the Software, and to prepare derivative works of the
	            // Software, and to permit third-parties to whom the Software is furnished to
	            // do so, all subject to the following:
	            //
	            // The copyright notices in the Software and this entire statement, including
	            // the above license grant, this restriction and the following disclaimer,
	            // must be included in all copies of the Software, in whole or in part, and
	            // all derivative works of the Software, unless such copies or derivative
	            // works are solely in the form of machine-executable object code generated by
	            // a source language processor.
	            //
	            // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	            // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	            // FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
	            // SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
	            // FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
	            // ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	            // DEALINGS IN THE SOFTWARE.

	#if defined(__clang__)
	            constexpr auto prefix = std::string_view{"[T = "};
	            constexpr auto suffix = std::string_view{"]"};
	            constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
	#elif defined(__GNUC__)
	            constexpr auto prefix = std::string_view{"with T = "};
	            constexpr auto suffix = std::string_view{"]"};
	            constexpr auto function = std::string_view{__PRETTY_FUNCTION__};
	#elif defined(_MSC_VER)
	            constexpr auto prefix = std::string_view{"typeNameAsArray<"};
	            constexpr auto suffix = std::string_view{">(void)"};
	            constexpr auto function = std::string_view{__FUNCSIG__};
	#else
	#    warning Unsupported compiler
	            constexpr auto prefix = std::string_view{};
	            constexpr auto suffix = std::string_view{};
	            constexpr auto function = std::string_view{};
	#endif

	            constexpr auto start = function.find(prefix) + prefix.size();
	            constexpr auto end = function.rfind(suffix);
	            static_assert(start <= end);

	            constexpr auto name = function.substr(start, (end - start));

	            constexpr auto arrAndSize = [&]() constexpr
	            {
	                Array<char, name.size()> nameArray{};
	                constexprCopy(name.begin(), name.end(), nameArray.begin());

	#ifdef _MSC_VER
	                // MSVC 19.32 runs into a syntax error if we just capture nameArray. Passing it as argument is a
	                // workaround. Applies to the following lambdas.

	                // strip "struct " and "class ".
	                auto removeAllOccurences = [](auto& nameArray, std::size_t size, std::string_view str) constexpr
	                {
	                    auto e = nameArray.begin() + size;
	                    while(true)
	                    {
	                        auto it = constexprSearch(nameArray.begin(), e, str.begin(), str.end());
	                        if(it == e)
	                            break;
	                        constexprCopy(it + str.size(), e, it);
	                        e -= str.size();
	                    }
	                    return e - nameArray.begin();
	                };

	                auto size = removeAllOccurences(nameArray, nameArray.size(), std::string_view{"struct "});
	                size = removeAllOccurences(nameArray, size, std::string_view{"class "});
	#else
	                auto size = nameArray.size();
	#endif

	                if(size > 3)
	                {
	                    // remove spaces between closing template angle brackets and after commas
	                    auto e = nameArray.begin() + size;
	                    for(auto b = nameArray.begin(); b < e - 2; b++)
	                    {
	                        if((b[0] == '>' && b[1] == ' ' && b[2] == '>') || (b[0] == ',' && b[1] == ' '))
	                        {
	                            constexprCopy(b + 2, e, b + 1);
	                            e--;
	                        }
	                    }
	                    size = e - nameArray.begin();
	                }

	                return std::pair{nameArray, size};
	            }();

	            return resizeArray<arrAndSize.second>(arrAndSize.first);
	        }

	        template<typename T>
	        inline constexpr auto typeNameStorage = typeNameAsArray<T>();
	    } // namespace internal

	    LLAMA_EXPORT
	    template<typename T>
	    inline constexpr auto qualifiedTypeName = []
	    {
	        constexpr auto& value = internal::typeNameStorage<T>;
	        return std::string_view{value.data(), value.size()};
	    }();

	    namespace internal
	    {
	        constexpr auto isIdentChar(char c) -> bool
	        {
	            if(c >= 'A' && c <= 'Z')
	                return true;
	            if(c >= 'a' && c <= 'z')
	                return true;
	            if(c >= '0' && c <= '9')
	                return true;
	            if(c == '_')
	                return true;
	            return false;
	        }

	        template<typename T>
	        inline constexpr auto structNameStorage = []() constexpr
	        {
	            // strip namespace qualifiers before type names
	            constexpr auto arrAndSize = []() constexpr
	            {
	                auto s = internal::typeNameStorage<T>;
	                auto b = s.begin();
	                auto e = s.end();

	#if defined(__clang__)
	                constexpr auto anonNs = std::string_view{"(anonymous namespace)::"};
	#elif defined(__NVCOMPILER)
	                constexpr auto anonNs = std::string_view{"<unnamed>::"};
	#elif defined(__GNUG__)
	                constexpr auto anonNs = std::string_view{"{anonymous}::"};
	#elif defined(_MSC_VER)
	                constexpr auto anonNs = std::string_view{"`anonymous-namespace'::"};
	#else
	                constexpr auto anonNs = std::string_view{"@"}; // just anything we won't find
	#endif
	                std::size_t pos = 0;
	                while((pos = std::string_view(b, e - b).find(anonNs)) != std::string::npos)
	                {
	                    constexprCopy(b + pos + anonNs.size(), e, b + pos);
	                    e -= anonNs.size();
	                }

	                while(true)
	                {
	                    // find iterator to after "::"
	                    auto l = b;
	                    while(l + 1 < e && !(l[0] == ':' && l[1] == ':'))
	                        l++;
	                    if(l + 1 == e)
	                        break;
	                    l += 2;

	                    // find iterator to first identifier char before "::"
	                    auto f = l - 3; // start at first char before "::"
	                    while(s.begin() < f && isIdentChar(f[-1]))
	                        f--;

	                    // cut out [f:l[
	                    constexprCopy(l, e, f);
	                    e -= (l - f);
	                    b = f;
	                }

	                return std::pair{s, e - s.begin()};
	            }();

	            return resizeArray<arrAndSize.second>(arrAndSize.first);
	        }();
	    } // namespace internal

	    LLAMA_EXPORT
	    template<typename T>
	    constexpr auto structName(T = {}) -> std::string_view
	    {
	        constexpr auto& value = internal::structNameStorage<T>;
	        return std::string_view{&value[0], value.size()};
	    }

	    namespace internal
	    {
	        constexpr auto intToStrSize(std::size_t s)
	        {
	            std::size_t len = 1;
	            while(s >= 10)
	            {
	                len++;
	                s /= 10;
	            }
	            return len;
	        }

	        template<typename RecordDim, std::size_t... Coords>
	        LLAMA_ACC inline constexpr auto recordCoordTagsStorage = []() constexpr
	        {
	            using Tags = GetTags<RecordDim, RecordCoord<Coords...>>;

	            // precompute char array size
	            constexpr auto size = [&]() constexpr
	            {
	                std::size_t s = 0;
	                mp_for_each<Tags>(
	                    [&](auto tag)
	                    {
	                        using Tag = decltype(tag);
	                        if constexpr(isRecordCoord<Tag>)
	                        {
	                            // handle array indices
	                            static_assert(Tag::size == 1);
	                            s += 2; // for the '[' and ']'
	                            s += intToStrSize(Tag::front);
	                        }
	                        else
	                        {
	                            if(s != 0)
	                                s++; // for the '.'s
	                            s += structName(tag).size();
	                        }
	                    });
	                return s;
	            }();
	            llama::Array<char, size> a{};
	            auto it = a.begin();

	            mp_for_each<Tags>(
	                [&](auto tag) constexpr
	                {
	                    using Tag = decltype(tag);
	                    if constexpr(isRecordCoord<Tag>)
	                    {
	                        auto n = Tag::front;
	                        *it = '[';
	                        it++;
	                        it += intToStrSize(n);
	                        auto it2 = it; // take copy because we write number backward
	                        do // NOLINT(cppcoreguidelines-avoid-do-while)
	                        {
	                            it2--;
	                            *it2 = '0' + n % 10;
	                            n /= 10;
	                        } while(n != 0);
	                        *it = ']';
	                        it++;
	                    }
	                    else
	                    {
	                        if(it != a.begin())
	                        {
	                            *it = '.';
	                            it++;
	                        }
	                        constexpr auto sn = structName(tag);
	                        constexprCopy(sn.begin(), sn.end(), it);
	                        it += sn.size();
	                    }
	                });

	            if(!a.empty() && a.back() == 0)
	                throw std::logic_error{"Implementation error: Array should have been completely overwritten."};

	            return a;
	        }();
	    } // namespace internal

	    /// Returns a pretty representation of the record coordinate inside the given record dimension. Tags are
	    /// interspersed by '.' and arrays are represented using subscript notation ("[123]").
	    LLAMA_EXPORT
	    template<typename RecordDim, std::size_t... Coords>
	    constexpr auto prettyRecordCoord(RecordCoord<Coords...> = {}) -> std::string_view
	    {
	        constexpr auto& value = internal::recordCoordTagsStorage<RecordDim, Coords...>;
	        return std::string_view{value.data(), value.size()};
	    }

	    LLAMA_EXPORT
	    template<typename RecordDim>
	    constexpr auto prettyRecordCoord(RecordCoord<>) -> std::string_view
	    {
	        return {};
	    }
	} // namespace llama
	// ==
	// == ./include/llama/StructName.hpp ==
	// ============================================================================

// #include "View.hpp"    // amalgamate: file already inlined
// #include "macros.hpp"    // amalgamate: file already inlined

#include <boost/functional/hash.hpp>
#include <iosfwd>
// #include <type_traits>    // amalgamate: file already included

namespace llama
{
    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord, bool OwnView>
    struct RecordRef;

    LLAMA_EXPORT
    template<typename View>
    inline constexpr auto isRecordRef = false;

    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord, bool OwnView>
    inline constexpr auto isRecordRef<RecordRef<View, BoundRecordCoord, OwnView>> = true;

    /// Returns a \ref One with the same record dimension as the given record ref, with values copyied from rr.
    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord, bool OwnView>
    LLAMA_FN_HOST_ACC_INLINE auto copyRecord(const RecordRef<View, BoundRecordCoord, OwnView>& rr)
    {
        using RecordDim = typename RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim;
        One<RecordDim> temp;
        temp = rr;
        return temp;
    }

    namespace internal
    {
        template<
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefArithOperator(
            LeftRecord& left,
            const RecordRef<RightView, RightBoundRecordDim, RightOwnView>& right) -> LeftRecord&
        {
            using RightRecord = RecordRef<RightView, RightBoundRecordDim, RightOwnView>;
            // if the record dimension left and right is the same, a single loop is enough and no tag check is needed.
            // this safes a lot of compilation time.
            using LARD = typename LeftRecord::AccessibleRecordDim;
            using RARD = typename RightRecord::AccessibleRecordDim;
            if constexpr(std::is_same_v<LARD, RARD>)
            {
                forEachLeafCoord<LARD>([&](auto rc) LLAMA_LAMBDA_INLINE { Functor{}(left(rc), right(rc)); });
            }
            else
            {
                forEachLeafCoord<LARD>(
                    [&](auto leftRC) LLAMA_LAMBDA_INLINE
                    {
                        using LeftInnerCoord = decltype(leftRC);
                        forEachLeafCoord<RARD>(
                            [&](auto rightRC) LLAMA_LAMBDA_INLINE
                            {
                                using RightInnerCoord = decltype(rightRC);
                                if constexpr(hasSameTags<LARD, LeftInnerCoord, RARD, RightInnerCoord>)
                                {
                                    Functor{}(left(leftRC), right(rightRC));
                                }
                            });
                    });
            }
            return left;
        }

        template<typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefArithOperator(LeftRecord& left, const T& right) -> LeftRecord&
        {
            forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>([&](auto leftRC) LLAMA_LAMBDA_INLINE
                                                                       { Functor{}(left(leftRC), right); });
            return left;
        }

        template<
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefRelOperator(
            const LeftRecord& left,
            const RecordRef<RightView, RightBoundRecordDim, RightOwnView>& right) -> bool
        {
            using RightRecord = RecordRef<RightView, RightBoundRecordDim, RightOwnView>;
            bool result = true;
            // if the record dimension left and right is the same, a single loop is enough and no tag check is needed.
            // this safes a lot of compilation time.
            using LARD = typename LeftRecord::AccessibleRecordDim;
            using RARD = typename RightRecord::AccessibleRecordDim;
            if constexpr(std::is_same_v<LARD, RARD>)
            {
                forEachLeafCoord<LARD>([&](auto rc) LLAMA_LAMBDA_INLINE { result &= Functor{}(left(rc), right(rc)); });
            }
            else
            {
                forEachLeafCoord<LARD>(
                    [&](auto leftRC) LLAMA_LAMBDA_INLINE
                    {
                        using LeftInnerCoord = decltype(leftRC);
                        forEachLeafCoord<RARD>(
                            [&](auto rightRC) LLAMA_LAMBDA_INLINE
                            {
                                using RightInnerCoord = decltype(rightRC);
                                if constexpr(hasSameTags<LARD, LeftInnerCoord, RARD, RightInnerCoord>)
                                {
                                    result &= Functor{}(left(leftRC), right(rightRC));
                                }
                            });
                    });
            }
            return result;
        }

        template<typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefRelOperator(const LeftRecord& left, const T& right) -> bool
        {
            bool result = true;
            forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>([&](auto leftRC) LLAMA_LAMBDA_INLINE
                                                                       { result &= Functor{}(left(leftRC), right); });
            return result;
        }

        struct Assign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) = b;
            }
        };

        struct PlusAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) += b;
            }
        };

        struct MinusAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) -= b;
            }
        };

        struct MultiplyAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) *= b;
            }
        };

        struct DivideAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) /= b;
            }
        };

        struct ModuloAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) %= b;
            }
        };

        template<
            typename ProxyReference,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<ProxyReference>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(ProxyReference&& leaf, T) -> ProxyReference
        {
            return leaf;
        }

        template<
            typename TWithOptionalConst,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<TWithOptionalConst>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(TWithOptionalConst& leaf, T)
            -> std::reference_wrapper<TWithOptionalConst>
        {
            return leaf;
        }

        template<typename RecordRef, typename T, std::size_t N, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImplForArray(RecordRef&& vd, T (&&)[N], std::index_sequence<Is...>)
        {
            return std::make_tuple(asTupleImpl(vd(RecordCoord<Is>{}), T{})...);
        }

        template<typename RecordRef, typename T, std::size_t N>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(RecordRef&& vd, T (&&a)[N])
        {
            return asTupleImplForArray(std::forward<RecordRef>(vd), std::move(a), std::make_index_sequence<N>{});
        }

        template<typename RecordRef, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(RecordRef&& vd, Record<Fields...>)
        {
            return std::make_tuple(asTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template<
            typename ProxyReference,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<ProxyReference>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(ProxyReference&& leaf, T) -> std::tuple<ProxyReference>
        {
            static_assert(!std::is_reference_v<ProxyReference>);
            return {std::move(leaf)}; // NOLINT(bugprone-move-forwarding-reference)
        }

        template<
            typename TWithOptionalConst,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<TWithOptionalConst>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(TWithOptionalConst& leaf, T) -> std::tuple<TWithOptionalConst&>
        {
            return {leaf};
        }

        template<typename RecordRef, typename T, std::size_t N, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImplForArray(RecordRef&& vd, T (&&)[N], std::index_sequence<Is...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(RecordCoord<Is>{}), T{})...);
        }

        template<typename RecordRef, typename T, std::size_t N>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(RecordRef&& vd, T (&&a)[N])
        {
            return asFlatTupleImplForArray(std::forward<RecordRef>(vd), std::move(a), std::make_index_sequence<N>{});
        }

        template<typename RecordRef, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(RecordRef&& vd, Record<Fields...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template<typename T, typename = void>
        inline constexpr auto isTupleLike = false;

        // get<I>(t) and std::tuple_size<T> must be available
        using std::get; // make sure a get<0>() can be found, so the compiler can compile the trait
        template<typename T>
        inline constexpr auto isTupleLike<T, std::void_t<decltype(get<0>(std::declval<T>())), std::tuple_size<T>>>
            = true;

        template<typename... Ts>
        inline constexpr auto dependentFalse = false;

        template<typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>);

        template<typename T1, typename T2>
        LLAMA_FN_HOST_ACC_INLINE void assignTupleElement(T1&& dst, T2&& src)
        {
            if constexpr(isTupleLike<std::decay_t<T1>> && isTupleLike<std::decay_t<T2>>)
            {
                static_assert(std::tuple_size_v<std::decay_t<T1>> == std::tuple_size_v<std::decay_t<T2>>);
                assignTuples(dst, src, std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>{});
            }
            else if constexpr(!isTupleLike<std::decay_t<T1>> && !isTupleLike<std::decay_t<T2>>)
                std::forward<T1>(dst) = std::forward<T2>(src);
            else
                static_assert(
                    dependentFalse<T1, T2>,
                    "Elements to assign are not tuple/tuple or non-tuple/non-tuple.");
        }

        template<typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>)
        {
            static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>);
            using std::get;
            (assignTupleElement(get<Is>(std::forward<Tuple1>(dst)), get<Is>(std::forward<Tuple2>(src))), ...);
        }

        template<typename T, typename Tuple, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeFromTuple(Tuple&& src, std::index_sequence<Is...>)
        {
            using std::get;
            return T{get<Is>(src)...}; // no forward of src, since we call get multiple times on it
        }

        template<typename T, typename SFINAE, typename... Args>
        inline constexpr auto isDirectListInitializableImpl = false;

        template<typename T, typename... Args>
        inline constexpr auto
            isDirectListInitializableImpl<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...>
            = true;

        template<typename T, typename... Args>
        inline constexpr auto isDirectListInitializable = isDirectListInitializableImpl<T, void, Args...>;

        template<typename T, typename Tuple>
        inline constexpr auto isDirectListInitializableFromTuple = false;

        template<typename T, template<typename...> typename Tuple, typename... Args>
        inline constexpr auto isDirectListInitializableFromTuple<T, Tuple<Args...>>
            = isDirectListInitializable<T, Args...>;

        template<typename T, typename Simd, typename SrcRC, typename DstRC>
        LLAMA_FN_HOST_ACC_INLINE void loadSimdFromField(const T& srcRef, Simd& dstSimd, SrcRC srcRC, DstRC dstRC);

        template<typename Simd, typename TFwd, typename SrcRC, typename DstRC>
        LLAMA_FN_HOST_ACC_INLINE void storeSimdToField(const Simd& srcSimd, TFwd&& dstRef, SrcRC srcRC, DstRC dstRC);
    } // namespace internal

    /// Record reference type returned by \ref View after resolving an array dimensions coordinate or partially
    /// resolving a \ref RecordCoord. A record reference does not hold data itself, it just binds enough information
    /// (array dimensions coord and partial record coord) to retrieve it later from a \ref View. Records references
    /// should not be created by the user. They are returned from various access functions in \ref View and RecordRef
    /// itself.
    LLAMA_EXPORT
    template<typename TView, typename TBoundRecordCoord, bool OwnView>
    struct RecordRef : private TView::Mapping::ArrayExtents::Index
    {
        using View = TView; ///< View this record reference points into.
        using BoundRecordCoord
            = TBoundRecordCoord; ///< Record coords into View::RecordDim which are already bound by this RecordRef.

    private:
        using ArrayIndex = typename View::Mapping::ArrayExtents::Index;
        using RecordDim = typename View::Mapping::RecordDim;

        std::conditional_t<OwnView, View, View&> view;

    public:
        /// Subtree of the record dimension of View starting at BoundRecordCoord. If BoundRecordCoord is
        /// `RecordCoord<>` (default) AccessibleRecordDim is the same as `Mapping::RecordDim`.
        using AccessibleRecordDim = GetType<RecordDim, BoundRecordCoord>;

        /// Creates an empty RecordRef. Only available for if the view is owned. Used by llama::One.
        LLAMA_FN_HOST_ACC_INLINE RecordRef()
            /* requires(OwnView) */
            : ArrayIndex{}
            , view{allocScalarView<0, RecordDim>()}
        {
            static_assert(OwnView, "The default constructor of RecordRef is only available if it owns the view.");
        }

        LLAMA_FN_HOST_ACC_INLINE
        RecordRef(ArrayIndex ai, std::conditional_t<OwnView, View&&, View&> view)
            : ArrayIndex{ai}
            , view{static_cast<decltype(view)>(view)}
        {
        }

        RecordRef(const RecordRef&) = default;

        // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const RecordRef& other) -> RecordRef&
        {
            // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
            return this->operator=<RecordRef>(other);
        }

        RecordRef(RecordRef&&) noexcept = default;
        auto operator=(RecordRef&&) noexcept -> RecordRef& = default;

        ~RecordRef() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayIndex() const -> ArrayIndex
        {
            return static_cast<const ArrayIndex&>(*this);
        }

        /// Create a RecordRef from a different RecordRef. Only available for if the view is owned. Used by
        /// llama::One.
        template<typename OtherView, typename OtherBoundRecordCoord, bool OtherOwnView>
        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        LLAMA_FN_HOST_ACC_INLINE RecordRef(const RecordRef<OtherView, OtherBoundRecordCoord, OtherOwnView>& recordRef)
            /* requires(OwnView) */
            : RecordRef()
        {
            static_assert(
                OwnView,
                "The copy constructor of RecordRef from a different RecordRef is only available if it owns "
                "the "
                "view.");
            *this = recordRef;
        }

        // TODO(bgruber): unify with previous in C++20 and use explicit(cond)
        /// Create a RecordRef from a scalar. Only available for if the view is owned. Used by llama::One.
        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE explicit RecordRef(const T& scalar)
            /* requires(OwnView) */
            : RecordRef()
        {
            static_assert(
                OwnView,
                "The constructor of RecordRef from a scalar is only available if it owns the view.");
            *this = scalar;
        }

        /// Access a record in the record dimension underneath the current record reference using a \ref RecordCoord.
        /// If the access resolves to a leaf, an l-value reference to a variable inside the \ref View storage is
        /// returned, otherwise another RecordRef.
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            using AccessedType = GetType<RecordDim, AbsolutCoord>;
            if constexpr(isRecordDim<AccessedType>)
                return RecordRef<const View, AbsolutCoord>{arrayIndex(), this->view};
            else
                return this->view.access(arrayIndex(), AbsolutCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            using AccessedType = GetType<RecordDim, AbsolutCoord>;
            if constexpr(isRecordDim<AccessedType>)
                return RecordRef<View, AbsolutCoord>{arrayIndex(), this->view};
            else
                return this->view.access(arrayIndex(), AbsolutCoord{});
        }

        /// Access a record in the record dimension underneath the current record reference using a series of tags. If
        /// the access resolves to a leaf, an l-value reference to a variable inside the \ref View storage is returned,
        /// otherwise another RecordRef.
        template<typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, Tags...>;
            return operator()(RecordCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template<typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, Tags...>;
            return operator()(RecordCoord{});
        }

#ifdef LLAMA_HAS_STRING_FIELDS
        /// Experimental
        template<internal::FixedString Name>
        LLAMA_FN_HOST_ACC_INLINE auto at() const -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, internal::StringTag<Name>>;
            return operator()(RecordCoord{});
        }

        // FIXME(bgruber): remove redundancy
        /// Experimental
        template<internal::FixedString Name>
        LLAMA_FN_HOST_ACC_INLINE auto at() -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, internal::StringTag<Name>>;
            return operator()(RecordCoord{});
        }
#endif

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const T& other) -> RecordRef&
        {
            // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
            return internal::recordRefArithOperator<internal::Assign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator+=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::PlusAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator-=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::MinusAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator*=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::MultiplyAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator/=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::DivideAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator%=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::ModuloAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) += t;
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const T& t, const RecordRef& vd)
        {
            return vd + t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator-(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) -= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) *= t;
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const T& t, const RecordRef& vd)
        {
            return vd * t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator/(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) /= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator%(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) %= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::equal_to<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const T& t, const RecordRef& vd) -> bool
        {
            return vd == t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const RecordRef& vd, const T& t) -> bool
        {
            return !(vd == t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const RecordRef& vd) -> bool
        {
            return !(t == vd);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::less<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const T& t, const RecordRef& vd) -> bool
        {
            return vd > t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::less_equal<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const T& t, const RecordRef& vd) -> bool
        {
            return vd >= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::greater<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const T& t, const RecordRef& vd) -> bool
        {
            return vd < t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::greater_equal<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const T& t, const RecordRef& vd) -> bool
        {
            return vd <= t;
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple()
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple() const
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple()
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple() const
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        template<std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template<std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template<typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this RecordRef recursively represents "
                "like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        template<typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() const -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this RecordRef recursively represents "
                "like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        struct Loader
        {
            RecordRef& vd;

            template<typename T>
            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE operator T()
            {
                return vd.loadAs<T>();
            }
        };

        struct LoaderConst
        {
            const RecordRef& vd;

            template<typename T>
            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE operator T() const
            {
                return vd.loadAs<T>();
            }
        };

        LLAMA_FN_HOST_ACC_INLINE auto load() -> Loader
        {
            return {*this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto load() const -> LoaderConst
        {
            return {*this};
        }

        template<typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE void store(const TupleLike& t)
        {
            internal::assignTuples(asTuple(), t, std::make_index_sequence<std::tuple_size_v<TupleLike>>{});
        }

        // swap for equal RecordRef
        LLAMA_FN_HOST_ACC_INLINE friend void swap(
            std::conditional_t<OwnView, RecordRef&, RecordRef> a,
            std::conditional_t<OwnView, RecordRef&, RecordRef> b) noexcept
        {
            forEachLeafCoord<AccessibleRecordDim>(
                [&](auto rc) LLAMA_LAMBDA_INLINE
                {
                    using std::swap;
                    LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                    // FIXME(bgruber): swap is constexpr in C++20, so nvcc rightfully complains that we call a __host__
                    // function here. But we must call ADL swap, so we can pick up any swap() for any user defined type
                    // in the record dimension. Let's see if this ever hits us. Moving to C++20 will solve it.
                    swap(a(rc), b(rc));
                    LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
                });
        }

        // FIXME(bgruber): the SIMD load/store functions need to navigate back from a record ref to the contained view
        // to find subsequent elements. This is not a great design for now and the SIMD load/store functions should
        // probably take iterators to records.
        template<typename T, typename Simd, typename SrcRC, typename DstRC>
        friend LLAMA_FN_HOST_ACC_INLINE void internal::loadSimdFromField(
            const T& srcRef,
            Simd& dstSimd,
            SrcRC srcRC,
            DstRC dstRC);
        template<typename Simd, typename TFwd, typename SrcRC, typename DstRC>
        friend LLAMA_FN_HOST_ACC_INLINE void internal::storeSimdToField(
            const Simd& srcSimd,
            TFwd&& dstRef,
            SrcRC srcRC,
            DstRC dstRC);
    };

    // swap for heterogeneous RecordRef
    LLAMA_EXPORT
    template<
        typename ViewA,
        typename BoundRecordDimA,
        bool OwnViewA,
        typename ViewB,
        typename BoundRecordDimB,
        bool OwnViewB>
    LLAMA_FN_HOST_ACC_INLINE auto swap(
        RecordRef<ViewA, BoundRecordDimA, OwnViewA>& a,
        RecordRef<ViewB, BoundRecordDimB, OwnViewB>& b) noexcept
        -> std::enable_if_t<std::is_same_v<
            typename RecordRef<ViewA, BoundRecordDimA, OwnViewA>::AccessibleRecordDim,
            typename RecordRef<ViewB, BoundRecordDimB, OwnViewB>::AccessibleRecordDim>>
    {
        using LeftRecord = RecordRef<ViewA, BoundRecordDimA, OwnViewA>;
        forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
            [&](auto rc) LLAMA_LAMBDA_INLINE
            {
                using std::swap;
                swap(a(rc), b(rc));
            });
    }

    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord, bool OwnView>
    auto operator<<(std::ostream& os, const RecordRef<View, BoundRecordCoord, OwnView>& vr) -> std::ostream&
    {
        using RecordDim = typename RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim;
        os << "{";
        if constexpr(std::is_array_v<RecordDim>)
        {
            mp_for_each<mp_iota_c<std::extent_v<RecordDim>>>(
                [&](auto ic)
                {
                    constexpr std::size_t i = decltype(ic)::value;
                    if(i > 0)
                        os << ", ";
                    os << '[' << i << ']' << ": " << vr(RecordCoord<i>{});
                });
        }
        else
        {
            mp_for_each<mp_iota<mp_size<RecordDim>>>(
                [&](auto ic)
                {
                    constexpr std::size_t i = decltype(ic)::value;
                    if(i > 0)
                        os << ", ";
                    using Field = mp_at_c<RecordDim, i>;
                    os << structName<GetFieldTag<Field>>() << ": " << vr(RecordCoord<i>{});
                });
        }
        os << "}";
        return os;
    }

    LLAMA_EXPORT
    template<typename RecordRefFwd, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaf(RecordRefFwd&& vr, Functor&& functor)
    {
        using RecordRef = std::remove_reference_t<RecordRefFwd>;
        forEachLeafCoord<typename RecordRef::AccessibleRecordDim>(
            [functor = std::forward<Functor>(functor), &vr = vr](auto rc)
                LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr mutable) { std::forward<Functor>(functor)(vr(rc)); });
    }

    namespace internal
    {
        // gets the value type for a given T, where T models either a value, an l-value reference, a proxy reference or
        // a RecordRef.
        template<typename T, typename = void>
        struct ValueOf
        {
            using type = T;
        };

        template<typename T>
        struct ValueOf<T, std::enable_if_t<isRecordRef<T>>>
        {
            using type = One<typename T::AccessibleRecordDim>;
        };

#ifdef __cpp_lib_concepts
        template<ProxyReference T>
#else
        template<typename T>
#endif
        struct ValueOf<T, std::enable_if_t<isProxyReference<T>>>
        {
            using type = typename T::value_type;
        };

        template<typename T>
        struct ValueOf<T&>
        {
            using type = std::remove_const_t<T>;
        };
    } // namespace internal

    /// Pulls a copy of the given value or reference. Proxy references are resolved to their value types.
    LLAMA_EXPORT
    template<typename T>
    LLAMA_FN_HOST_ACC_INLINE auto decayCopy(T&& valueOrRef) -> typename internal::ValueOf<T>::type
    {
        return std::forward<T>(valueOrRef);
    }

    /// Scope guard type. ScopedUpdate takes a copy of a value through a reference and stores it internally during
    /// construction. The stored value is written back when ScopedUpdate is destroyed. ScopedUpdate tries to act like
    /// the stored value as much as possible, exposing member functions of the stored value and acting like a proxy
    /// reference if the stored value is a primitive type.
    LLAMA_EXPORT
    template<typename Reference, typename = void>
    struct ScopedUpdate : internal::ValueOf<Reference>::type
    {
        using value_type = typename internal::ValueOf<Reference>::type;

        /// Loads a copy of the value referenced by r. Stores r and the loaded value.
        LLAMA_FN_HOST_ACC_INLINE explicit ScopedUpdate(Reference r) : value_type(r), ref(r)
        {
        }

        ScopedUpdate(const ScopedUpdate&) = delete;
        auto operator=(const ScopedUpdate&) -> ScopedUpdate& = delete;

        ScopedUpdate(ScopedUpdate&&) noexcept = default;
        auto operator=(ScopedUpdate&&) noexcept -> ScopedUpdate& = default;

        using value_type::operator=;

        /// Stores the internally stored value back to the referenced value.
        LLAMA_FN_HOST_ACC_INLINE ~ScopedUpdate()
        {
            ref = static_cast<value_type&>(*this);
        }

        /// Get access to the stored value.
        LLAMA_FN_HOST_ACC_INLINE auto get() -> value_type&
        {
            return *this;
        }

        /// Get access to the stored value.
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> const value_type&
        {
            return *this;
        }

    private:
        Reference ref;
    };

    LLAMA_EXPORT
    template<typename Reference>
    struct ScopedUpdate<
        Reference,
        std::enable_if_t<std::is_fundamental_v<typename internal::ValueOf<Reference>::type>>>
        : ProxyRefOpMixin<ScopedUpdate<Reference>, typename internal::ValueOf<Reference>::type>
    {
        using value_type = typename internal::ValueOf<Reference>::type;

        LLAMA_FN_HOST_ACC_INLINE explicit ScopedUpdate(Reference r) : value(r), ref(r)
        {
        }

        ScopedUpdate(const ScopedUpdate&) = delete;
        auto operator=(const ScopedUpdate&) -> ScopedUpdate& = delete;

        ScopedUpdate(ScopedUpdate&&) noexcept = default;
        auto operator=(ScopedUpdate&&) noexcept -> ScopedUpdate& = default;

        LLAMA_FN_HOST_ACC_INLINE auto get() -> value_type&
        {
            return value;
        }

        LLAMA_FN_HOST_ACC_INLINE auto get() const -> const value_type&
        {
            return value;
        }

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        LLAMA_FN_HOST_ACC_INLINE operator const value_type&() const
        {
            return value;
        }

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        LLAMA_FN_HOST_ACC_INLINE operator value_type&()
        {
            return value;
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type v) -> ScopedUpdate&
        {
            value = v;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE ~ScopedUpdate()
        {
            ref = value;
        }

    private:
        value_type value;
        Reference ref;
    };

    namespace internal
    {
        template<typename T, typename = void>
        struct ReferenceTo
        {
            using type = T&;
        };

        template<typename T>
        struct ReferenceTo<T, std::enable_if_t<isRecordRef<T> && !isOne<T>>>
        {
            using type = T;
        };

#ifdef __cpp_lib_concepts
        template<ProxyReference T>
#else
        template<typename T>
#endif
        struct ReferenceTo<T, std::enable_if_t<isProxyReference<T>>>
        {
            using type = T;
        };
    } // namespace internal

    LLAMA_EXPORT
    template<typename T>
    ScopedUpdate(T) -> ScopedUpdate<typename internal::ReferenceTo<std::remove_reference_t<T>>::type>;
} // namespace llama

LLAMA_EXPORT
template<typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_size<llama::RecordRef<View, BoundRecordCoord, OwnView>> // NOLINT(cert-dcl58-cpp)
    : boost::mp11::mp_size<typename llama::RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim>
{
};

LLAMA_EXPORT
template<std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, llama::RecordRef<View, BoundRecordCoord, OwnView>> // NOLINT(cert-dcl58-cpp)
{
    using type = decltype(std::declval<llama::RecordRef<View, BoundRecordCoord, OwnView>>().template get<I>());
};

LLAMA_EXPORT
template<std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, const llama::RecordRef<View, BoundRecordCoord, OwnView>> // NOLINT(cert-dcl58-cpp)
{
    using type = decltype(std::declval<const llama::RecordRef<View, BoundRecordCoord, OwnView>>().template get<I>());
};

LLAMA_EXPORT
template<typename View, typename BoundRecordCoord, bool OwnView>
struct std::hash<llama::RecordRef<View, BoundRecordCoord, OwnView>> // NOLINT(cert-dcl58-cpp)
{
    LLAMA_FN_HOST_ACC_INLINE auto operator()(const llama::RecordRef<View, BoundRecordCoord, OwnView>& rr) const
        -> std::size_t
    {
        std::size_t acc = 0;
        llama::forEachLeaf(
            rr,
            [&](auto&& ref) LLAMA_LAMBDA_INLINE { boost::hash_combine(acc, llama::decayCopy(ref)); });
        return acc;
    }
};

#if CAN_USE_RANGES
LLAMA_EXPORT
template<
    typename ViewA,
    typename BoundA,
    bool OwnA,
    typename ViewB,
    typename BoundB,
    bool OwnB,
    template<class>
    class TQual,
    template<class>
    class UQual>
struct std::
    // NOLINTNEXTLINE(cert-dcl58-cpp)
    basic_common_reference<llama::RecordRef<ViewA, BoundA, OwnA>, llama::RecordRef<ViewB, BoundB, OwnB>, TQual, UQual>
{
    using type = std::enable_if_t<
        std::is_same_v<
            typename llama::RecordRef<ViewA, BoundA, OwnA>::AccessibleRecordDim,
            typename llama::RecordRef<ViewB, BoundB, OwnB>::AccessibleRecordDim>,
        llama::One<typename ViewA::RecordDim>>;
};
#endif
// ==
// == ./include/llama/RecordRef.hpp ==
// ============================================================================

// ============================================================================
// == ./include/llama/Simd.hpp ==
// ==
// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// #pragma once
// #include "Core.hpp"    // amalgamate: file already inlined
// #include "RecordRef.hpp"    // amalgamate: file already inlined
// #include "macros.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/mapping/AoS.hpp ==
	// ==
	// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already inlined

	namespace llama::mapping
	{
	    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
	    /// \tparam TFieldAlignment If Align, padding bytes are inserted to guarantee that struct members are properly
	    /// aligned. If Pack, struct members are tightly packed.
	    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    /// \tparam PermuteFields Defines how the record dimension's fields should be permuted. See \ref
	    /// PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref PermuteFieldsDecreasingAlignment and
	    /// \ref PermuteFieldsMinimizePadding.
	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        FieldAlignment TFieldAlignment = FieldAlignment::Align,
	        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        template<typename> typename PermuteFields = PermuteFieldsInOrder>
	    struct AoS : MappingBase<TArrayExtents, TRecordDim>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using size_type = typename Base::size_type;

	    public:
	        inline static constexpr FieldAlignment fieldAlignment = TFieldAlignment;
	        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
	        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
	        inline static constexpr std::size_t blobCount = 1;

	        using Base::Base;

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
	        {
	            return LinearizeArrayIndexFunctor{}.size(Base::extents())
	                * flatSizeOf<typename Permuter::FlatRecordDim, fieldAlignment == FieldAlignment::Align>;
	        }

	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            typename Base::ArrayIndex ai,
	            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
	        {
	            constexpr std::size_t flatFieldIndex =
	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6
	                *& // mess with nvcc compiler state to workaround bug
	#endif
	                 Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>;
	            const auto offset
	                = LinearizeArrayIndexFunctor{}(ai, Base::extents())
	                    * static_cast<size_type>(
	                        flatSizeOf<typename Permuter::FlatRecordDim, fieldAlignment == FieldAlignment::Align>)
	                + static_cast<size_type>(flatOffsetOf<
	                                         typename Permuter::FlatRecordDim,
	                                         flatFieldIndex,
	                                         fieldAlignment == FieldAlignment::Align>);
	            return {size_type{0}, offset};
	        }
	    };

	    // we can drop this when inherited ctors also inherit deduction guides
	    LLAMA_EXPORT
	    template<typename TArrayExtents, typename TRecordDim>
	    AoS(TArrayExtents, TRecordDim) -> AoS<TArrayExtents, TRecordDim>;

	    /// Array of struct mapping preserving the alignment of the field types by inserting padding.
	    /// \see AoS
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    using AlignedAoS = AoS<ArrayExtents, RecordDim, FieldAlignment::Align, LinearizeArrayIndexFunctor>;

	    /// Array of struct mapping preserving the alignment of the field types by inserting padding and permuting the
	    /// field order to minimize this padding. \see AoS
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    using MinAlignedAoS = AoS<
	        ArrayExtents,
	        RecordDim,
	        FieldAlignment::Align,
	        LinearizeArrayIndexFunctor,
	        PermuteFieldsMinimizePadding>;

	    /// Array of struct mapping packing the field types tightly, violating the type's alignment requirements.
	    /// \see AoS
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    using PackedAoS = AoS<ArrayExtents, RecordDim, FieldAlignment::Pack, LinearizeArrayIndexFunctor>;

	    /// Binds parameters to an \ref AoS mapping except for array and record dimension, producing a quoted meta
	    /// function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<
	        FieldAlignment Alignment = FieldAlignment::Align,
	        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    struct BindAoS
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = AoS<ArrayExtents, RecordDim, Alignment, LinearizeArrayIndexFunctor>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isAoS = false;

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        FieldAlignment FieldAlignment,
	        typename LinearizeArrayIndexFunctor,
	        template<typename>
	        typename PermuteFields>
	    inline constexpr bool
	        isAoS<AoS<ArrayExtents, RecordDim, FieldAlignment, LinearizeArrayIndexFunctor, PermuteFields>>
	        = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/AoS.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/AoSoA.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already inlined

	// #include <limits>    // amalgamate: file already included

	namespace llama::mapping
	{
	    /// The maximum number of vector lanes that can be used to fetch each leaf type in the record dimension into a
	    /// vector register of the given size in bits.
	    LLAMA_EXPORT
	    template<typename RecordDim, std::size_t VectorRegisterBits>
	    inline constexpr std::size_t maxLanes = []() constexpr
	    {
	        auto max = std::numeric_limits<std::size_t>::max();
	        forEachLeafCoord<RecordDim>(
	            [&](auto rc)
	            {
	                using AttributeType = GetType<RecordDim, decltype(rc)>;
	                max = std::min(max, VectorRegisterBits / (sizeof(AttributeType) * CHAR_BIT));
	            });
	        return max;
	    }();

	    /// Array of struct of arrays mapping. Used to create a \ref View via \ref allocView.
	    /// \tparam Lanes The size of the inner arrays of this array of struct of arrays.
	    /// \tparam PermuteFields Defines how the record dimension's fields should be permuted. See \ref
	    /// PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref PermuteFieldsDecreasingAlignment and
	    /// \ref PermuteFieldsMinimizePadding.
	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        typename TArrayExtents::value_type Lanes,
	        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        template<typename> typename PermuteFields = PermuteFieldsInOrder>
	    struct AoSoA : MappingBase<TArrayExtents, TRecordDim>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using size_type = typename Base::size_type;

	    public:
	        inline static constexpr typename TArrayExtents::value_type lanes = Lanes;
	        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
	        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
	        inline static constexpr std::size_t blobCount = 1;

	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ >= 12
	        using Base::Base;
	#else
	        constexpr AoSoA() = default;

	        LLAMA_FN_HOST_ACC_INLINE constexpr explicit AoSoA(TArrayExtents extents, TRecordDim = {}) : Base(extents)
	        {
	        }
	#endif

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
	        {
	            const auto rs = static_cast<size_type>(sizeOf<TRecordDim>);
	            return roundUpToMultiple(LinearizeArrayIndexFunctor{}.size(Base::extents()) * rs, Lanes * rs);
	        }

	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            typename Base::ArrayIndex ai,
	            RecordCoord<RecordCoords...> rc = {}) const -> NrAndOffset<size_type>
	        {
	            return blobNrAndOffset(LinearizeArrayIndexFunctor{}(ai, Base::extents()), rc);
	        }

	        // Exposed for aosoaCommonBlockCopy. Should be private ...
	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            size_type flatArrayIndex,
	            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
	        {
	            constexpr std::size_t flatFieldIndex =
	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6
	                *& // mess with nvcc compiler state to workaround bug
	#endif
	                 Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>;
	            const auto blockIndex = flatArrayIndex / Lanes;
	            const auto laneIndex = flatArrayIndex % Lanes;
	            const auto offset = static_cast<size_type>(sizeOf<TRecordDim> * Lanes) * blockIndex
	                + static_cast<size_type>(flatOffsetOf<typename Permuter::FlatRecordDim, flatFieldIndex, false>) * Lanes
	                + static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>)) * laneIndex;
	            return {0, offset};
	        }
	    };

	    /// Binds parameters to an \ref AoSoA mapping except for array and record dimension, producing a quoted meta
	    /// function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<
	        std::size_t Lanes,
	        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        template<typename> typename PermuteFields = PermuteFieldsInOrder>
	    struct BindAoSoA
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = AoSoA<ArrayExtents, RecordDim, Lanes, LinearizeArrayIndexFunctor, PermuteFields>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isAoSoA = false;

	    LLAMA_EXPORT
	    template<typename AD, typename RD, typename AD::value_type L, typename Lin, template<typename> typename Perm>
	    inline constexpr bool isAoSoA<AoSoA<AD, RD, L, Lin, Perm>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/AoSoA.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/SoA.hpp ==
	// ==
	// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already inlined

	// #include <limits>    // amalgamate: file already included

	namespace llama::mapping
	{
	    LLAMA_EXPORT
	    enum class Blobs
	    {
	        Single,
	        OnePerField
	    };

	    LLAMA_EXPORT
	    enum class SubArrayAlignment
	    {
	        Pack,
	        Align
	    };

	    /// Struct of array mapping. Used to create a \ref View via \ref allocView. We recommend to use multiple blobs when
	    /// the array extents are dynamic and an aligned single blob version when they are static.
	    /// \tparam TBlobs If OnePerField, every element of the record dimension is mapped to its own blob.
	    /// \tparam TSubArrayAlignment Only relevant when TBlobs == Single, ignored otherwise. If Align, aligns the sub
	    /// arrays created within the single blob by inserting padding. If the array extents are dynamic, this may add some
	    /// overhead to the mapping logic.
	    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    /// \tparam PermuteFieldsSingleBlob Defines how the record dimension's fields should be permuted if Blobs is
	    /// Single. See \ref PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref
	    /// PermuteFieldsDecreasingAlignment and \ref PermuteFieldsMinimizePadding.
	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        Blobs TBlobs = Blobs::OnePerField,
	        SubArrayAlignment TSubArrayAlignment
	        = TBlobs == Blobs::Single ? SubArrayAlignment::Align : SubArrayAlignment::Pack,
	        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        template<typename> typename PermuteFieldsSingleBlob = PermuteFieldsInOrder>
	    struct SoA : MappingBase<TArrayExtents, TRecordDim>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using size_type = typename TArrayExtents::value_type;

	    public:
	        inline static constexpr Blobs blobs = TBlobs;
	        inline static constexpr SubArrayAlignment subArrayAlignment = TSubArrayAlignment;
	        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
	        using Permuter = PermuteFieldsSingleBlob<FlatRecordDim<TRecordDim>>;
	        inline static constexpr std::size_t blobCount
	            = blobs == Blobs::OnePerField ? mp_size<FlatRecordDim<TRecordDim>>::value : 1;

	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ >= 12
	        using Base::Base;
	#else
	        constexpr SoA() = default;

	        LLAMA_FN_HOST_ACC_INLINE constexpr explicit SoA(TArrayExtents extents, TRecordDim = {}) : Base(extents)
	        {
	        }
	#endif

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize([[maybe_unused]] size_type blobIndex) const -> size_type
	        {
	            const auto flatSize = LinearizeArrayIndexFunctor{}.size(Base::extents());
	            if constexpr(blobs == Blobs::OnePerField)
	            {
	                constexpr auto typeSizes = []() constexpr
	                {
	                    Array<size_type, blobCount> r{};
	                    forEachLeafCoord<TRecordDim>([&r, i = 0](auto rc) mutable constexpr
	                                                 { r[i++] = sizeof(GetType<TRecordDim, decltype(rc)>); });
	                    return r;
	                }();
	                return flatSize * typeSizes[blobIndex];
	            }
	            else if constexpr(subArrayAlignment == SubArrayAlignment::Align)
	            {
	                size_type size = 0;
	                using FRD = typename Permuter::FlatRecordDim;
	                mp_for_each<mp_transform<mp_identity, FRD>>(
	                    [&](auto ti) LLAMA_LAMBDA_INLINE
	                    {
	                        using FieldType = typename decltype(ti)::type;
	                        size = roundUpToMultiple(size, static_cast<size_type>(alignof(FieldType)));
	                        size += static_cast<size_type>(sizeof(FieldType)) * flatSize;
	                    });
	                return size;
	            }
	            else
	            {
	                return flatSize * static_cast<size_type>(sizeOf<TRecordDim>);
	            }
	        }

	    private:
	        static LLAMA_CONSTEVAL auto computeSubArrayOffsets()
	        {
	            using FRD = typename Permuter::FlatRecordDim;
	            constexpr auto staticFlatSize = LinearizeArrayIndexFunctor{}.size(TArrayExtents{});
	            constexpr auto subArrays = mp_size<FRD>::value;
	            Array<size_type, subArrays> r{};
	            // r[0] == 0, only compute the following offsets
	            mp_for_each<mp_iota_c<subArrays - 1>>(
	                [&](auto ic)
	                {
	                    constexpr auto i = decltype(ic)::value;
	                    r[i + 1] = r[i];
	                    using ThisFieldType = mp_at_c<FRD, i>;
	                    r[i + 1] += static_cast<size_type>(sizeof(ThisFieldType)) * staticFlatSize;
	                    using NextFieldType = mp_at_c<FRD, i + 1>;
	                    r[i + 1] = roundUpToMultiple(r[i + 1], static_cast<size_type>(alignof(NextFieldType)));
	                });
	            return r;
	        }

	    public:
	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            typename Base::ArrayIndex ai,
	            RecordCoord<RecordCoords...> rc = {}) const -> NrAndOffset<size_type>
	        {
	            return blobNrAndOffset(LinearizeArrayIndexFunctor{}(ai, Base::extents()), rc);
	        }

	        // Exposed for aosoaCommonBlockCopy. Should be private ...
	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            size_type flatArrayIndex,
	            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
	        {
	            const size_type elementOffset
	                = flatArrayIndex * static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>));
	            if constexpr(blobs == Blobs::OnePerField)
	            {
	                constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
	                return {blob, elementOffset};
	            }
	            else
	            {
	                constexpr std::size_t flatFieldIndex =
	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6
	                    *& // mess with nvcc compiler state to workaround bug
	#endif
	                     Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>;
	                const size_type flatSize = LinearizeArrayIndexFunctor{}.size(Base::extents());
	                using FRD = typename Permuter::FlatRecordDim;
	                if constexpr(subArrayAlignment == SubArrayAlignment::Align)
	                {
	                    if constexpr(TArrayExtents::rankStatic == TArrayExtents::rank)
	                    {
	                        // full array extents are known statically, we can precompute the sub array offsets
	                        constexpr auto subArrayOffsets = computeSubArrayOffsets();
	                        return {0, subArrayOffsets[flatFieldIndex] + elementOffset};
	                    }
	                    else
	                    {
	                        // TODO(bgruber): we can take a shortcut here if we know that flatSize is a multiple of all
	                        // type's alignment. We can also precompute a table of sub array starts (and maybe store it),
	                        // or rely on the compiler it out of loops.
	                        size_type offset = 0;
	                        mp_for_each<mp_iota_c<flatFieldIndex>>(
	                            [&](auto ic) LLAMA_LAMBDA_INLINE
	                            {
	                                constexpr auto i = decltype(ic)::value;
	                                using ThisFieldType = mp_at_c<FRD, i>;
	                                offset += static_cast<size_type>(sizeof(ThisFieldType)) * flatSize;
	                                using NextFieldType = mp_at_c<FRD, i + 1>;
	                                offset = roundUpToMultiple(offset, static_cast<size_type>(alignof(NextFieldType)));
	                            });
	                        offset += elementOffset;
	                        return {0, offset};
	                    }
	                }
	                else
	                {
	                    const auto offset
	                        = elementOffset + static_cast<size_type>(flatOffsetOf<FRD, flatFieldIndex, false>) * flatSize;
	                    return {0, offset};
	                }
	            }
	        }
	    };

	    // we can drop this when inherited ctors also inherit deduction guides
	    LLAMA_EXPORT
	    template<typename TArrayExtents, typename TRecordDim>
	    SoA(TArrayExtents, TRecordDim) -> SoA<TArrayExtents, TRecordDim>;

	    /// Struct of array mapping storing the entire layout in a single blob. The starts of the sub arrays are aligned by
	    /// inserting padding. \see SoA
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    using AlignedSingleBlobSoA
	        = SoA<ArrayExtents, RecordDim, Blobs::Single, SubArrayAlignment::Align, LinearizeArrayIndexFunctor>;

	    /// Struct of array mapping storing the entire layout in a single blob. The sub arrays are tightly packed,
	    /// violating the type's alignment requirements. \see SoA
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    using PackedSingleBlobSoA
	        = SoA<ArrayExtents, RecordDim, Blobs::Single, SubArrayAlignment::Pack, LinearizeArrayIndexFunctor>;

	    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
	    /// \see SoA
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    using MultiBlobSoA
	        = SoA<ArrayExtents, RecordDim, Blobs::OnePerField, SubArrayAlignment::Pack, LinearizeArrayIndexFunctor>;

	    /// Binds parameters to an \ref SoA mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<
	        Blobs Blobs = Blobs::OnePerField,
	        SubArrayAlignment SubArrayAlignment = SubArrayAlignment::Pack,
	        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
	    struct BindSoA
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayIndexFunctor>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isSoA = false;

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        Blobs Blobs,
	        SubArrayAlignment SubArrayAlignment,
	        typename LinearizeArrayIndexFunctor>
	    inline constexpr bool isSoA<SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayIndexFunctor>>
	        = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/SoA.hpp ==
	// ============================================================================


// #include <type_traits>    // amalgamate: file already included

namespace llama
{
    /// Traits of a specific Simd implementation. Please specialize this template for the SIMD types you are going to
    /// use in your program.
    /// Each specialization SimdTraits<Simd> must provide:
    /// * an alias `value_type` to indicate the element type of the Simd.
    /// * a `static constexpr size_t lanes` variable holding the number of SIMD lanes of the Simd.
    /// * a `static auto loadUnalinged(const value_type* mem) -> Simd` function, loading a Simd from the given memory
    /// address.
    /// * a `static void storeUnaligned(Simd simd, value_type* mem)` function, storing the given Simd to a given
    /// memory address.
    /// * a `static auto gather(const value_type* mem, std::array<int, lanes> indices) -> Simd` function, gathering
    ///  values into a Simd from the memory addresses identified by mem + indices * sizeof(value_type).
    /// * a `static void scatter(Simd simd, value_type* mem, std::array<int, lanes> indices)` function, scattering the
    /// values from a Simd to the memory addresses identified by mem + indices * sizeof(value_type).
    LLAMA_EXPORT
    template<typename Simd, typename SFINAE = void>
    struct SimdTraits
    {
        static_assert(sizeof(Simd) == 0, "Please specialize SimdTraits for the type Simd");
    };

    LLAMA_EXPORT
    template<typename T>
    struct SimdTraits<T, std::enable_if_t<std::is_arithmetic_v<T>>>
    {
        using value_type = T;

        inline static constexpr std::size_t lanes = 1;

        static LLAMA_FN_HOST_ACC_INLINE auto loadUnaligned(const T* mem) -> T
        {
            return *mem;
        }

        static LLAMA_FN_HOST_ACC_INLINE void storeUnaligned(T t, T* mem)
        {
            *mem = t;
        }

        static LLAMA_FN_HOST_ACC_INLINE auto gather(const value_type* mem, std::array<int, lanes> indices) -> T
        {
            return mem[indices[0]];
        }

        static LLAMA_FN_HOST_ACC_INLINE void scatter(T t, value_type* mem, std::array<int, lanes> indices)
        {
            mem[indices[0]] = t;
        }
    };

    /// The number of SIMD simdLanes the given SIMD vector or \ref Simd<T> has. If Simd is not a structural \ref Simd
    /// or \ref SimdN, this is a shortcut for SimdTraits<Simd>::lanes.
    LLAMA_EXPORT
    template<typename Simd, typename SFINAE = void>
    inline constexpr auto simdLanes = SimdTraits<Simd>::lanes;

    /// Chooses the number of SIMD lanes for the given record dimension by mapping each field type to a SIMD type and
    /// then reducing their sizes.
    /// @tparam MakeSimd Type function creating a SIMD type given a field type from the record dimension.
    /// @param reduce Binary reduction function to reduce the SIMD lanes.
    LLAMA_EXPORT
    template<typename RecordDim, template<typename> typename MakeSimd, typename BinaryReductionFunction>
    LLAMA_CONSTEVAL auto chooseSimdLanes(BinaryReductionFunction reduce) -> std::size_t
    {
        using FRD = FlatRecordDim<RecordDim>;
        std::size_t lanes = simdLanes<MakeSimd<mp_first<FRD>>>;
        mp_for_each<mp_transform<std::add_pointer_t, mp_drop_c<FRD, 1>>>(
            [&](auto* t)
            {
                using T = std::remove_reference_t<decltype(*t)>;
                lanes = reduce(lanes, simdLanes<MakeSimd<T>>);
            });
        assert(lanes > 0);
        return lanes;
    }

    /// Determines the number of simd lanes suitable to process all types occurring in the given record dimension. The
    /// algorithm ensures that even SIMD vectors for the smallest field type are filled completely and may thus require
    /// multiple SIMD vectors for some field types.
    /// @tparam RecordDim The record dimension to simdize
    /// @tparam MakeSimd Type function creating a SIMD type given a field type from the record dimension.
    LLAMA_EXPORT
    template<typename RecordDim, template<typename> typename MakeSimd>
    inline constexpr std::size_t simdLanesWithFullVectorsFor
        = chooseSimdLanes<RecordDim, MakeSimd>([](auto a, auto b) { return std::max(a, b); });

    /// Determines the number of simd lanes suitable to process all types occurring in the given record dimension. The
    /// algorithm ensures that the smallest number of SIMD registers is needed and may thus only partially fill
    /// registers for some data types.
    /// @tparam RecordDim The record dimension to simdize
    /// @tparam MakeSimd Type function creating a SIMD type given a field type from the record dimension.
    LLAMA_EXPORT
    template<typename RecordDim, template<typename> typename MakeSimd>
    inline constexpr std::size_t simdLanesWithLeastRegistersFor
        = chooseSimdLanes<RecordDim, MakeSimd>([](auto a, auto b) { return std::min(a, b); });

    namespace internal
    {
        template<std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
        struct BindMakeSizedSimd
        {
            template<typename U>
            using fn = MakeSizedSimd<U, N>;
        };

        template<
            typename RecordDim,
            std::size_t N,
            template<typename, /* std::integral */ auto>
            typename MakeSizedSimd>
        struct SimdizeNImpl
        {
            using type = TransformLeaves<RecordDim, internal::BindMakeSizedSimd<N, MakeSizedSimd>::template fn>;
        };

        template<typename RecordDim, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
        struct SimdizeNImpl<RecordDim, 1, MakeSizedSimd>
        {
            using type = RecordDim;
        };
    } // namespace internal

    /// Transforms the given record dimension into a SIMD version of it. Each leaf field type will be replaced by a
    /// sized SIMD vector with length N, as determined by MakeSizedSimd. If N is 1, SimdizeN<T, 1, ...> is an alias for
    /// T.
    LLAMA_EXPORT
    template<typename RecordDim, std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
    using SimdizeN = typename internal::SimdizeNImpl<RecordDim, N, MakeSizedSimd>::type;

    /// Transforms the given record dimension into a SIMD version of it. Each leaf field type will be replaced by a
    /// SIMD vector, as determined by MakeSimd.
    LLAMA_EXPORT
    template<typename RecordDim, template<typename> typename MakeSimd>
    using Simdize = TransformLeaves<RecordDim, MakeSimd>;

    /// Creates a SIMD version of the given type. Of T is a record dimension, creates a \ref One where each field is a
    /// SIMD type of the original field type. The SIMD vectors have length N. If N is 1, an ordinary \ref One of the
    /// record dimension T is created. If T is not a record dimension, a SIMD vector with value T and length N is
    /// created. If N is 1 (and T is not a record dimension), then T is produced.
    LLAMA_EXPORT
    template<typename T, std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
    using SimdN = typename std::conditional_t<
        isRecordDim<T>,
        std::conditional_t<N == 1, mp_identity<One<T>>, mp_identity<One<SimdizeN<T, N, MakeSizedSimd>>>>,
        std::conditional_t<N == 1, mp_identity<T>, mp_identity<SimdizeN<T, N, MakeSizedSimd>>>>::type;

    /// Creates a SIMD version of the given type. Of T is a record dimension, creates a \ref One where each field is a
    /// SIMD type of the original field type.
    LLAMA_EXPORT
    template<typename T, template<typename> typename MakeSimd>
    using Simd = typename std::
        conditional_t<isRecordDim<T>, mp_identity<One<Simdize<T, MakeSimd>>>, mp_identity<Simdize<T, MakeSimd>>>::type;

    namespace internal
    {
        template<std::size_t S>
        struct SizeEqualTo
        {
            template<typename Simd>
            using fn = std::bool_constant<simdLanes<Simd> == S>;
        };
    } // namespace internal

    /// Specialization for Simd<RecordDim>. Only works if all SIMD types in the fields of the record dimension have the
    /// same size.
    LLAMA_EXPORT
    template<typename Simd>
    inline constexpr std::size_t simdLanes<Simd, std::enable_if_t<isRecordRef<Simd>>> = []
    {
        using FRD = FlatRecordDim<typename Simd::AccessibleRecordDim>;
        using FirstFieldType = mp_first<FRD>;
        static_assert(mp_all_of_q<FRD, internal::SizeEqualTo<simdLanes<FirstFieldType>>>::value);
        return simdLanes<FirstFieldType>;
    }();

    namespace internal
    {
        template<typename AoSMapping, typename ElementType, std::size_t Lanes>
        inline constexpr auto aosStridedIndices = []()
        {
            auto stride = flatSizeOf<
                              typename AoSMapping::Permuter::FlatRecordDim,
                              AoSMapping::fieldAlignment == llama::mapping::FieldAlignment::Align>
                / sizeof(ElementType);
            std::array<int, Lanes> indices{};
            for(int i = 0; i < static_cast<int>(Lanes); i++)
                indices[i] = i * stride;
            return indices;
        }();

        template<typename T, typename Simd, typename SrcRC, typename DstRC>
        LLAMA_FN_HOST_ACC_INLINE void loadSimdFromField(const T& srcRef, Simd& dstSimd, SrcRC srcRC, DstRC dstRC)
        {
            using FieldType = GetType<typename T::AccessibleRecordDim, SrcRC>;
            using ElementSimd = std::decay_t<decltype(dstSimd(dstRC))>;
            using Traits = SimdTraits<ElementSimd>;

            auto loadElementWise = [&]
            {
                auto b = ArrayIndexIterator{srcRef.view.extents(), srcRef.arrayIndex()};
                for(std::size_t i = 0; i < Traits::lanes; i++)
                    reinterpret_cast<FieldType*>(&dstSimd(dstRC))[i]
                        = srcRef.view(*b++)(cat(typename T::BoundRecordCoord{}, srcRC));
            };

            // TODO(bgruber): can we generalize the logic whether we can load a dstSimd from that mapping?
            using Mapping = typename T::View::Mapping;
            if constexpr(mapping::isSoA<Mapping>)
            {
                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                dstSimd(dstRC) = Traits::loadUnaligned(&srcRef(srcRC));
                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
            }
            else if constexpr(mapping::isAoSoA<typename T::View::Mapping>)
            {
                // TODO(bgruber): this check is too strict
                if(T::View::Mapping::ArrayExtents::rank == 1 && srcRef.arrayIndex()[0] % Traits::lanes == 0
                   && T::View::Mapping::lanes >= Traits::lanes)
                {
                    LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                    dstSimd(dstRC) = Traits::loadUnaligned(&srcRef(srcRC));
                    LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
                }
                else
                    loadElementWise();
            }
            else if constexpr(mapping::isAoS<Mapping>)
            {
                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                dstSimd(dstRC) = Traits::gather(&srcRef(srcRC), aosStridedIndices<Mapping, FieldType, Traits::lanes>);
                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
            }
            else
                loadElementWise();
        }

        template<typename Simd, typename TFwd, typename SrcRC, typename DstRC>
        LLAMA_FN_HOST_ACC_INLINE void storeSimdToField(const Simd& srcSimd, TFwd&& dstRef, SrcRC srcRC, DstRC dstRC)
        {
            using T = std::remove_reference_t<TFwd>;
            using FieldType = GetType<typename T::AccessibleRecordDim, DstRC>;
            using ElementSimd = std::decay_t<decltype(srcSimd(srcRC))>;
            using Traits = SimdTraits<ElementSimd>;

            auto storeElementWise = [&]
            {
                // TODO(bgruber): how does this generalize conceptually to 2D and higher dimensions? in which
                // direction should we collect SIMD values?
                auto b = ArrayIndexIterator{dstRef.view.extents(), dstRef.arrayIndex()};
                for(std::size_t i = 0; i < Traits::lanes; i++)
                    dstRef.view (*b++)(cat(typename T::BoundRecordCoord{}, dstRC))
                        = reinterpret_cast<const FieldType*>(&srcSimd(srcRC))[i];
            };

            // TODO(bgruber): can we generalize the logic whether we can store a srcSimd to that mapping?
            using Mapping = typename std::remove_reference_t<T>::View::Mapping;
            if constexpr(mapping::isSoA<Mapping>)
            {
                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                Traits::storeUnaligned(srcSimd(srcRC), &dstRef(dstRC));
                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
            }
            else if constexpr(mapping::isAoSoA<typename T::View::Mapping>)
            {
                // TODO(bgruber): this check is too strict
                if(T::View::Mapping::ArrayExtents::rank == 1 && dstRef.arrayIndex()[0] % Traits::lanes == 0
                   && T::View::Mapping::lanes >= Traits::lanes)
                {
                    LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                    Traits::storeUnaligned(srcSimd(srcRC), &dstRef(dstRC));
                    LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
                }
                else
                    storeElementWise();
            }
            else if constexpr(mapping::isAoS<Mapping>)
            {
                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                Traits::scatter(srcSimd(srcRC), &dstRef(dstRC), aosStridedIndices<Mapping, FieldType, Traits::lanes>);
                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
            }
            else
                storeElementWise();
        }
    } // namespace internal

    /// Loads SIMD vectors of data starting from the given record reference to dstSimd. Only field tags occurring in
    /// RecordRef are loaded. If Simd contains multiple fields of SIMD types, a SIMD vector will be fetched for each of
    /// the fields. The number of elements fetched per SIMD vector depends on the SIMD width of the vector. Simd is
    /// allowed to have different vector lengths per element.
    LLAMA_EXPORT
    template<typename T, typename Simd>
    LLAMA_FN_HOST_ACC_INLINE void loadSimd(const T& srcRef, Simd& dstSimd)
    {
        // structured dstSimd type and record reference
        if constexpr(isRecordRef<Simd> && isRecordRef<T>)
        {
            if constexpr(simdLanes<Simd> == simdLanes<T>) // fast path mainly for scalar SimdN<T, 1, ...>
                dstSimd = srcRef;
            else
            {
                using SrcARD = typename T::AccessibleRecordDim;
                using DstArd = typename Simd::AccessibleRecordDim;
                if constexpr(std::is_same_v<SrcARD, DstArd>)
                {
                    forEachLeafCoord<SrcARD>([&](auto rc) LLAMA_LAMBDA_INLINE
                                             { internal::loadSimdFromField(srcRef, dstSimd, rc, rc); });
                }
                else
                {
                    forEachLeafCoord<SrcARD>(
                        [&](auto srcRC) LLAMA_LAMBDA_INLINE
                        {
                            using SrcInnerCoord = decltype(srcRC);
                            forEachLeafCoord<DstArd>(
                                [&](auto dstRC) LLAMA_LAMBDA_INLINE
                                {
                                    using DstInnerCoord = decltype(dstRC);
                                    if constexpr(hasSameTags<SrcARD, SrcInnerCoord, DstArd, DstInnerCoord>)
                                    {
                                        internal::loadSimdFromField(srcRef, dstSimd, srcRC, dstRC);
                                    }
                                });
                        });
                }
            }
        }
        // unstructured dstSimd and reference type
        else if constexpr(!isRecordRef<Simd> && !isRecordRef<T>)
        {
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            dstSimd = SimdTraits<Simd>::loadUnaligned(&srcRef);
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
        else
        {
            // TODO(bgruber): when could we get here? Is this always an error?
            static_assert(sizeof(Simd) == 0, "Invalid combination of Simd type and reference type");
        }
    }

    /// Stores SIMD vectors of element data from the given srcSimd into memory starting at the provided record
    /// reference. Only field tags occurring in RecordRef are stored. If Simd contains multiple fields of SIMD types, a
    /// SIMD vector will be stored for each of the fields. The number of elements stored per SIMD vector depends on the
    /// SIMD width of the vector. Simd is allowed to have different vector lengths per element.
    LLAMA_EXPORT
    template<typename Simd, typename TFwd>
    LLAMA_FN_HOST_ACC_INLINE void storeSimd(const Simd& srcSimd, TFwd&& dstRef)
    {
        using T = std::decay_t<TFwd>;
        // structured Simd type and record reference
        if constexpr(isRecordRef<Simd> && isRecordRef<T>)
        {
            if constexpr(simdLanes<Simd> == simdLanes<T>) // fast path mainly for scalar SimdN<T, 1, ...>
                dstRef = srcSimd;
            else
            {
                using SrcARD = typename Simd::AccessibleRecordDim;
                using DstArd = typename T::AccessibleRecordDim;
                if constexpr(std::is_same_v<SrcARD, DstArd>)
                {
                    forEachLeafCoord<SrcARD>([&](auto rc) LLAMA_LAMBDA_INLINE
                                             { internal::storeSimdToField(srcSimd, dstRef, rc, rc); });
                }
                else
                {
                    forEachLeafCoord<SrcARD>(
                        [&](auto srcRC) LLAMA_LAMBDA_INLINE
                        {
                            using SrcInnerCoord = decltype(srcRC);
                            forEachLeafCoord<DstArd>(
                                [&](auto dstRC) LLAMA_LAMBDA_INLINE
                                {
                                    using DstInnerCoord = decltype(dstRC);
                                    if constexpr(hasSameTags<SrcARD, SrcInnerCoord, DstArd, DstInnerCoord>)
                                    {
                                        internal::storeSimdToField(srcSimd, dstRef, srcRC, dstRC);
                                    }
                                });
                        });
                }
            }
        }
        // unstructured srcSimd and reference type
        else if constexpr(!isRecordRef<Simd> && !isRecordRef<T>)
        {
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            SimdTraits<Simd>::storeUnaligned(srcSimd, &dstRef);
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
        else
        {
            // TODO(bgruber): when could we get here? Is this always an error?
            static_assert(sizeof(Simd) == 0, "Invalid combination of Simd type and reference type");
        }
    }

    LLAMA_EXPORT
    template<
        std::size_t N,
        template<typename, /* std::integral */ auto>
        typename MakeSizedSimd,
        typename View,
        typename UnarySimdFunction>
    void simdForEachN(View& view, UnarySimdFunction f)
    {
        using IndexType = typename View::Mapping::ArrayExtents::value_type;
        const auto total = product(view.mapping().extents());
        auto it = view.begin();
        IndexType i = 0;
        // simd loop
        while(i + IndexType{N} <= total)
        {
            SimdN<typename View::RecordDim, N, MakeSizedSimd> simd;
            loadSimd(*it, simd);
            if constexpr(std::is_void_v<decltype(f(simd))>)
                f(simd);
            else
                storeSimd(f(simd), *it);
            i += IndexType{N};
            it += IndexType{N};
        }
        // tail
        while(i < total)
        {
            auto scalar = One<typename View::RecordDim>{*it};
            if constexpr(std::is_void_v<decltype(f(scalar))>)
                f(scalar);
            else
                *it = f(scalar);
            ++i;
            ++it;
        }
    }

    LLAMA_EXPORT
    template<
        template<typename>
        typename MakeSimd,
        template<typename, /* std::integral */ auto>
        typename MakeSizedSimd,
        typename View,
        typename UnarySimdFunction>
    void simdForEach(View& view, UnarySimdFunction f)
    {
        constexpr auto n = llama::simdLanesWithFullVectorsFor<typename View::RecordDim, MakeSimd>;
        simdForEachN<n, MakeSizedSimd>(view, f);
    }
} // namespace llama
// ==
// == ./include/llama/Simd.hpp ==
// ============================================================================

// ============================================================================
// == ./include/llama/llama.hpp ==
// ==
// Copyright 2018 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// #pragma once
/// \mainpage LLAMA API documentation
///
/// LLAMA is a C++17 template header-only library for the abstraction of memory access patterns. It distinguishes
/// between the view of the algorithm on the memory and the real layout in the background. This enables performance
/// portability for multicore, manycore and gpu applications with the very same code.
///
/// In contrast to many other solutions LLAMA can define nested data structures of arbitrary depths and is not limited
/// only to struct of array and array of struct data layouts. It is also capable to explicitly define padding,
/// blocking, striding and any other run time or compile time access pattern simultaneously.
///
/// To archieve this goal LLAMA is split into mostly independent, orthogonal parts completely written in modern C++17
/// to run on as many architectures and with as many compilers as possible while still supporting extensions needed
/// e.g. to run on GPU or other many core hardware.
///
/// This page documents the API of LLAMA. The user documentation and an overview about the concepts and ideas can be
/// found here: https://llama-doc.rtfd.io
///
/// LLAMA is licensed under the MPL-2.0.

// NOLINTNEXTLINE(modernize-macro-to-enum)
#define LLAMA_VERSION_MAJOR 0
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define LLAMA_VERSION_MINOR 5
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define LLAMA_VERSION_PATCH 0

// suppress warnings on missing return statements. we get a lot of these because nvcc/nvc++ have some troubles with if
// constexpr.
#ifdef __NVCC__
#    ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#        pragma nv_diag_suppress 940
#    else
#        pragma diag_suppress 940
#    endif
#endif
#ifdef __NVCOMPILER
#    pragma push
#    pragma diag_suppress 941
#endif

// #include "ArrayExtents.hpp"    // amalgamate: file already inlined
// #include "ArrayIndexRange.hpp"    // amalgamate: file already inlined
// #include "BlobAllocators.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/Copy.hpp ==
	// ==
	// Copyright 2021 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "View.hpp"    // amalgamate: file already inlined
	// #include "mapping/AoSoA.hpp"    // amalgamate: file already inlined
	// #include "mapping/SoA.hpp"    // amalgamate: file already inlined

	#include <cstring>
	#include <numeric>

	namespace llama
	{
	    namespace internal
	    {
	        template<typename RecordDim>
	        void assertTrivialCopyable()
	        {
	            forEachLeafCoord<RecordDim>(
	                [](auto rc)
	                {
	                    static_assert(
	                        std::is_trivially_copyable_v<GetType<RecordDim, decltype(rc)>>,
	                        "All types in the record dimension must be trivially copyable");
	                });
	        }

	        // need a custom memcpy symbol in LLAMA, because with clang+CUDA, there are multiple std::memcpy symbols, so
	        // the address is ambiguous.
	        inline constexpr auto memcpy
	            = [](void* dst, const void* src, std::size_t size) { std::memcpy(dst, src, size); };

	        template<typename MemcpyFunc = decltype(memcpy)>
	        void parallelMemcpy(
	            std::byte* dst,
	            const std::byte* src,
	            std::size_t size,
	            std::size_t threadId = 0,
	            std::size_t threadCount = 1,
	            MemcpyFunc singleThreadMemcpy = memcpy)
	        {
	            const auto sizePerThread = size / threadCount;
	            const auto sizeLastThread = sizePerThread + size % threadCount;
	            const auto sizeThisThread = threadId == threadCount - 1 ? sizeLastThread : sizePerThread;
	            singleThreadMemcpy(dst + threadId * sizePerThread, src + threadId * sizePerThread, sizeThisThread);
	        }
	    } // namespace internal

	    /// Copy the blobs' content from the source view to the destination view in parallel with the given thread
	    /// configuration.  Both views need to have the same mappings with the same array extents.
	    /// @param threadId Zero-based id of calling thread for multi-threaded invocations.
	    /// @param threadCount Thread count in case of multi-threaded invocation.
	    /// \param singleThreadMemcpy The implementation of memcpy. By default: std::memcpy.
	    LLAMA_EXPORT
	    template<typename Mapping, typename SrcBlob, typename DstBlob, typename MemcpyFunc = decltype(internal::memcpy)>
	    void memcpyBlobs(
	        const View<Mapping, SrcBlob>& srcView,
	        View<Mapping, DstBlob>& dstView,
	        std::size_t threadId = 0,
	        std::size_t threadCount = 1,
	        MemcpyFunc singleThreadMemcpy = internal::memcpy)
	    {
	        internal::assertTrivialCopyable<typename Mapping::RecordDim>();

	        // TODO(bgruber): we do not verify if the mappings have other runtime state than the array dimensions
	        if(srcView.extents() != dstView.extents())
	            throw std::runtime_error{"Array dimensions sizes are different"};

	        // TODO(bgruber): this is maybe not the best parallel copying strategy
	        for(std::size_t i = 0; i < Mapping::blobCount; i++)
	            internal::parallelMemcpy(
	                &dstView.blobs()[i][0],
	                &srcView.blobs()[i][0],
	                dstView.mapping().blobSize(i),
	                threadId,
	                threadCount,
	                singleThreadMemcpy);
	    }

	    namespace internal
	    {
	        inline constexpr auto copyBlobWithMemcpy = [](const auto& src, auto& dst, std::size_t size)
	        {
	            static_assert(std::is_trivially_copyable_v<std::remove_reference_t<decltype(*&src[0])>>);
	            static_assert(std::is_trivially_copyable_v<std::remove_reference_t<decltype(*&dst[0])>>);
	            std::memcpy(&dst[0], &src[0], size);
	        };
	    } // namespace internal

	    /// Copy the blobs' content from the source view to the destination view. Both views need to have the same mapping,
	    /// and thus the same blob count and blob sizes. The copy is performed blob by blob.
	    /// \param copyBlob The function to use for copying blobs. Default is \ref internal::copyBlobWithMemcpy, which uses
	    /// std::memcpy.
	    LLAMA_EXPORT
	    template<
	        typename Mapping,
	        typename SrcBlob,
	        typename DstBlob,
	        typename BlobCopyFunc = decltype(internal::copyBlobWithMemcpy)>
	    void copyBlobs(
	        const View<Mapping, SrcBlob>& srcView,
	        View<Mapping, DstBlob>& dstView,
	        BlobCopyFunc copyBlob = internal::copyBlobWithMemcpy)
	    {
	        // TODO(bgruber): we do not verify if the mappings have other runtime state than the array dimensions
	        if(srcView.extents() != dstView.extents())
	            throw std::runtime_error{"Array dimensions sizes are different"};
	        for(std::size_t i = 0; i < Mapping::blobCount; i++)
	            copyBlob(srcView.blobs()[i], dstView.blobs()[i], dstView.mapping().blobSize(i));
	    }

	    /// Field-wise copy from source to destination view. Both views need to have the same array and record dimensions.
	    /// @param threadId Optional. Thread id in case of multi-threaded copy.
	    /// @param threadCount Optional. Thread count in case of multi-threaded copy.
	    LLAMA_EXPORT
	    template<typename SrcMapping, typename SrcBlob, typename DstMapping, typename DstBlob>
	    void fieldWiseCopy(
	        const View<SrcMapping, SrcBlob>& srcView,
	        View<DstMapping, DstBlob>& dstView,
	        std::size_t threadId = 0,
	        std::size_t threadCount = 1)
	    {
	        // TODO(bgruber): think if we can remove this restriction
	        static_assert(
	            std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>,
	            "The source and destination record dimensions must be the same");

	        if(srcView.extents() != dstView.extents())
	            throw std::runtime_error{"Array dimensions sizes are different"};

	        auto copyOne = [&](auto ai) LLAMA_LAMBDA_INLINE
	        {
	            forEachLeafCoord<typename DstMapping::RecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE
	                                                             { dstView(ai)(rc) = srcView(ai)(rc); });
	        };

	        constexpr auto dims = SrcMapping::ArrayExtents::rank;
	        const auto extents = srcView.extents().toArray();
	        const auto workPerThread = (extents[0] + threadCount - 1) / threadCount;
	        const auto start = threadId * workPerThread;
	        const auto end = std::min((threadId + 1) * workPerThread, static_cast<std::size_t>(extents[0]));
	        for(auto i = start; i < end; i++)
	        {
	            using SrcSizeType = typename SrcMapping::ArrayExtents::value_type;
	            if constexpr(dims > 1)
	                forEachArrayIndex(extents, copyOne, static_cast<SrcSizeType>(i));
	            else
	                copyOne(ArrayIndex<SrcSizeType, dims>{static_cast<std::size_t>(i)});
	        }
	    }

	    namespace internal
	    {
	        template<typename Mapping>
	        inline constexpr std::size_t aosoaLanes = 1;

	        template<
	            typename ArrayExtents,
	            typename RecordDim,
	            mapping::Blobs Blobs,
	            mapping::SubArrayAlignment SubArrayAlignment,
	            typename LinearizeArrayIndexFunctor,
	            template<typename>
	            typename PermuteSBFields>
	        inline constexpr std::size_t aosoaLanes<
	            mapping::
	                SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayIndexFunctor, PermuteSBFields>>
	            = std::numeric_limits<std::size_t>::max();

	        template<
	            typename ArrayExtents,
	            typename RecordDim,
	            typename ArrayExtents::value_type Lanes,
	            typename LinearizeArrayIndexFunctor,
	            template<typename>
	            typename PermuteFields>
	        inline constexpr std::size_t
	            aosoaLanes<mapping::AoSoA<ArrayExtents, RecordDim, Lanes, LinearizeArrayIndexFunctor, PermuteFields>>
	            = Lanes;
	    } // namespace internal

	    /// AoSoA copy strategy which transfers data in common blocks. SoA mappings are also allowed for at most 1
	    /// argument.
	    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
	    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
	    LLAMA_EXPORT
	    template<typename SrcMapping, typename SrcBlob, typename DstMapping, typename DstBlob>
	    void aosoaCommonBlockCopy(
	        const View<SrcMapping, SrcBlob>& srcView,
	        View<DstMapping, DstBlob>& dstView,
	        bool readOpt,
	        std::size_t threadId = 0,
	        std::size_t threadCount = 1)
	    {
	        static_assert(
	            mapping::isAoSoA<SrcMapping> || mapping::isSoA<SrcMapping>,
	            "Only AoSoA and SoA mappings allowed as source");
	        static_assert(
	            mapping::isAoSoA<DstMapping> || mapping::isSoA<DstMapping>,
	            "Only AoSoA and SoA mappings allowed as destination");

	        // TODO(bgruber): think if we can remove this restriction
	        static_assert(
	            std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>,
	            "The source and destination record dimensions must be the same");
	        static_assert(
	            std::is_same_v<
	                typename SrcMapping::LinearizeArrayIndexFunctor,
	                typename DstMapping::LinearizeArrayIndexFunctor>,
	            "Source and destination mapping need to use the same array dimensions linearizer");
	        using RecordDim = typename SrcMapping::RecordDim;
	        internal::assertTrivialCopyable<RecordDim>();

	        static constexpr auto lanesSrc = internal::aosoaLanes<SrcMapping>;
	        static constexpr auto lanesDst = internal::aosoaLanes<DstMapping>;

	        if(srcView.extents() != dstView.extents())
	            throw std::runtime_error{"Array dimensions sizes are different"};

	        static constexpr auto srcIsAoSoA = lanesSrc != std::numeric_limits<std::size_t>::max();
	        static constexpr auto dstIsAoSoA = lanesDst != std::numeric_limits<std::size_t>::max();

	        static_assert(srcIsAoSoA || dstIsAoSoA, "At least one of the mappings must be an AoSoA mapping");
	        static_assert(!srcIsAoSoA || SrcMapping::blobCount == 1, "Implementation assumes AoSoA with single blob");
	        static_assert(!dstIsAoSoA || DstMapping::blobCount == 1, "Implementation assumes AoSoA with single blob");

	        const auto flatSize = product(dstView.extents());

	        // TODO(bgruber): implement the following by adding additional copy loops for the remaining elements
	        if(!srcIsAoSoA && flatSize % lanesDst != 0)
	            throw std::runtime_error{"Source SoA mapping's total array elements must be evenly divisible by the "
	                                     "destination AoSoA Lane count."};
	        if(!dstIsAoSoA && flatSize % lanesSrc != 0)
	            throw std::runtime_error{"Destination SoA mapping's total array elements must be evenly divisible by the "
	                                     "source AoSoA Lane count."};

	        auto mapSrc = [&](std::size_t flatArrayIndex, auto rc) LLAMA_LAMBDA_INLINE
	        {
	            const auto [blob, off] = srcView.mapping().blobNrAndOffset(flatArrayIndex, rc);
	            return &srcView.blobs()[blob][off];
	        };
	        auto mapDst = [&](std::size_t flatArrayIndex, auto rc) LLAMA_LAMBDA_INLINE
	        {
	            const auto [blob, off] = dstView.mapping().blobNrAndOffset(flatArrayIndex, rc);
	            return &dstView.blobs()[blob][off];
	        };

	        static constexpr auto l = []
	        {
	            if constexpr(srcIsAoSoA && dstIsAoSoA)
	                return std::gcd(lanesSrc, lanesDst);
	            return std::min(lanesSrc, lanesDst);
	        }();
	        if(readOpt)
	        {
	            // optimized for linear reading
	            constexpr auto srcL = srcIsAoSoA ? lanesSrc : l;
	            const auto elementsPerThread = flatSize / srcL / threadCount * srcL;
	            {
	                const auto start = threadId * elementsPerThread;
	                const auto stop = threadId == threadCount - 1 ? flatSize : (threadId + 1) * elementsPerThread;

	                auto copyLBlock = [&](const std::byte*& threadSrc, std::size_t dstIndex, auto rc) LLAMA_LAMBDA_INLINE
	                {
	                    constexpr auto bytes = l * sizeof(GetType<RecordDim, decltype(rc)>);
	                    std::memcpy(mapDst(dstIndex, rc), threadSrc, bytes);
	                    threadSrc += bytes;
	                };
	                if constexpr(srcIsAoSoA)
	                {
	                    auto* threadSrc = mapSrc(start, RecordCoord<>{});
	                    for(std::size_t i = start; i < stop; i += lanesSrc)
	                        forEachLeafCoord<RecordDim>(
	                            [&](auto rc) LLAMA_LAMBDA_INLINE
	                            {
	                                for(std::size_t j = 0; j < lanesSrc; j += l)
	                                    copyLBlock(threadSrc, i + j, rc);
	                            });
	                }
	                else
	                {
	                    forEachLeafCoord<RecordDim>(
	                        [&](auto rc) LLAMA_LAMBDA_INLINE
	                        {
	                            auto* threadSrc = mapSrc(start, rc);
	                            for(std::size_t i = start; i < stop; i += l)
	                                copyLBlock(threadSrc, i, rc);
	                        });
	                }
	            }
	        }
	        else
	        {
	            // optimized for linear writing
	            constexpr auto dstL = dstIsAoSoA ? lanesDst : l;
	            const auto elementsPerThread = flatSize / dstL / threadCount * dstL;
	            {
	                const auto start = threadId * elementsPerThread;
	                const auto stop = threadId == threadCount - 1 ? flatSize : (threadId + 1) * elementsPerThread;

	                auto copyLBlock = [&](std::byte*& threadDst, std::size_t srcIndex, auto rc) LLAMA_LAMBDA_INLINE
	                {
	                    constexpr auto bytes = l * sizeof(GetType<RecordDim, decltype(rc)>);
	                    std::memcpy(threadDst, mapSrc(srcIndex, rc), bytes);
	                    threadDst += bytes;
	                };
	                if constexpr(dstIsAoSoA)
	                {
	                    auto* threadDst = mapDst(start, RecordCoord<>{});
	                    for(std::size_t i = start; i < stop; i += lanesDst)
	                        forEachLeafCoord<RecordDim>(
	                            [&](auto rc) LLAMA_LAMBDA_INLINE
	                            {
	                                for(std::size_t j = 0; j < lanesDst; j += l)
	                                    copyLBlock(threadDst, i + j, rc);
	                            });
	                }
	                else
	                {
	                    forEachLeafCoord<RecordDim>(
	                        [&](auto rc) LLAMA_LAMBDA_INLINE
	                        {
	                            auto* threadDst = mapDst(start, rc);
	                            for(std::size_t i = start; i < stop; i += l)
	                                copyLBlock(threadDst, i, rc);
	                        });
	                }
	            }
	        }
	    }

	    /// @brief Generic implementation of \ref copy defaulting to \ref fieldWiseCopy. LLAMA provides several
	    /// specializations of this construct for specific mappings. Users are encouraged to also specialize this template
	    /// with better copy algorithms for further combinations of mappings, if they can and want to provide a better
	    /// implementation.
	    LLAMA_EXPORT
	    template<typename SrcMapping, typename DstMapping, typename SFINAE = void>
	    struct Copy
	    {
	        template<typename SrcView, typename DstView>
	        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
	        {
	            fieldWiseCopy(srcView, dstView, threadId, threadCount);
	        }
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    struct Copy<Mapping, Mapping>
	    {
	        template<typename SrcView, typename DstView>
	        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
	        {
	            // FIXME(bgruber): need to fallback to fieldWiseCopy when elements are not trivially copyable
	            memcpyBlobs(srcView, dstView, threadId, threadCount);
	        }
	    };

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        typename LinearizeArrayIndex,
	        typename ArrayExtents::value_type LanesSrc,
	        typename ArrayExtents::value_type LanesDst,
	        template<typename>
	        typename PermuteFields>
	    struct Copy<
	        mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>,
	        mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>,
	        std::enable_if_t<LanesSrc != LanesDst>>
	    {
	        template<typename SrcBlob, typename DstBlob>
	        void operator()(
	            const View<mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>, SrcBlob>&
	                srcView,
	            View<mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>, DstBlob>&
	                dstView,
	            std::size_t threadId,
	            std::size_t threadCount)
	        {
	            constexpr auto readOpt = LanesSrc < LanesDst; // read contiguously on the AoSoA with the smaller lane count
	            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
	        }
	    };

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        typename LinearizeArrayIndex,
	        template<typename>
	        typename PermuteFields,
	        typename ArrayExtents::value_type LanesSrc,
	        mapping::Blobs DstBlobs,
	        mapping::SubArrayAlignment DstSubArrayAlignment>
	    struct Copy<
	        mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>,
	        mapping::SoA<ArrayExtents, RecordDim, DstBlobs, DstSubArrayAlignment, LinearizeArrayIndex, PermuteFields>>
	    {
	        template<typename SrcBlob, typename DstBlob>
	        void operator()(
	            const View<mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>, SrcBlob>&
	                srcView,
	            View<
	                mapping::
	                    SoA<ArrayExtents, RecordDim, DstBlobs, DstSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
	                DstBlob>& dstView,
	            std::size_t threadId,
	            std::size_t threadCount)
	        {
	            constexpr auto readOpt = true; // read contiguously on the AoSoA
	            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
	        }
	    };

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        typename LinearizeArrayIndex,
	        template<typename>
	        typename PermuteFields,
	        typename ArrayExtents::value_type LanesDst,
	        mapping::Blobs SrcBlobs,
	        mapping::SubArrayAlignment SrcSubArrayAlignment>
	    struct Copy<
	        mapping::SoA<ArrayExtents, RecordDim, SrcBlobs, SrcSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
	        mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>>
	    {
	        template<typename SrcBlob, typename DstBlob>
	        void operator()(
	            const View<
	                mapping::
	                    SoA<ArrayExtents, RecordDim, SrcBlobs, SrcSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
	                SrcBlob>& srcView,
	            View<mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>, DstBlob>&
	                dstView,
	            std::size_t threadId,
	            std::size_t threadCount)
	        {
	            constexpr auto readOpt = false; // read contiguously on the AoSoA
	            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
	        }
	    };

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        mapping::Blobs SrcBlobs,
	        mapping::Blobs DstBlobs,
	        mapping::SubArrayAlignment SrcSubArrayAlignment,
	        mapping::SubArrayAlignment DstSubArrayAlignment,
	        typename LinearizeArrayIndex,
	        template<typename>
	        typename PermuteFields>
	    struct Copy<
	        mapping::SoA<ArrayExtents, RecordDim, SrcBlobs, SrcSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
	        mapping::SoA<ArrayExtents, RecordDim, DstBlobs, DstSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
	        std::enable_if_t<SrcBlobs != DstBlobs || SrcSubArrayAlignment != DstSubArrayAlignment>>
	    {
	        template<typename SrcBlob, typename DstBlob>
	        void operator()(
	            const View<
	                mapping::
	                    SoA<ArrayExtents, RecordDim, SrcBlobs, SrcSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
	                SrcBlob>& srcView,
	            View<
	                mapping::
	                    SoA<ArrayExtents, RecordDim, DstBlobs, DstSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
	                DstBlob>& dstView,
	            std::size_t threadId,
	            std::size_t threadCount)
	        {
	            if(srcView.extents() != dstView.extents())
	                throw std::runtime_error{"Array dimensions sizes are different"};

	            const auto subArrayLength = product(srcView.extents());
	            forEachLeafCoord<RecordDim>(
	                [&](auto rc) LLAMA_LAMBDA_INLINE
	                {
	                    auto subArrayStart = [&](auto& view, auto rc) LLAMA_LAMBDA_INLINE
	                    {
	                        const auto [blob, off] = view.mapping().blobNrAndOffset(0, rc);
	                        return &view.blobs()[blob][off];
	                    };
	                    internal::parallelMemcpy(
	                        subArrayStart(dstView, rc),
	                        subArrayStart(srcView, rc),
	                        subArrayLength * sizeof(GetType<RecordDim, decltype(rc)>),
	                        threadId,
	                        threadCount);
	                });
	        }
	    };

	    /// Copy data from source to destination view. Both views need to have the same array and record
	    /// dimensions, but may have different mappings. The blobs need to be read- and writeable. Delegates to \ref Copy
	    /// to choose an implementation.
	    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
	    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
	    LLAMA_EXPORT
	    template<typename SrcMapping, typename SrcBlob, typename DstMapping, typename DstBlob>
	    void copy(
	        const View<SrcMapping, SrcBlob>& srcView,
	        View<DstMapping, DstBlob>& dstView,
	        std::size_t threadId = 0,
	        std::size_t threadCount = 1)
	    {
	        Copy<SrcMapping, DstMapping>{}(srcView, dstView, threadId, threadCount);
	    }
	} // namespace llama
	// ==
	// == ./include/llama/Copy.hpp ==
	// ============================================================================

// #include "Core.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/DumpMapping.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	#if __has_include(<fmt/format.h>)
	// #    include "ArrayIndexRange.hpp"    // amalgamate: file already inlined
	// #    include "Core.hpp"    // amalgamate: file already inlined
	// #    include "StructName.hpp"    // amalgamate: file already inlined
	// #    include "View.hpp"    // amalgamate: file already inlined

	#    include <fmt/format.h>
	#    include <functional>
	#    include <optional>
	// #    include <string>    // amalgamate: file already included
	// #    include <vector>    // amalgamate: file already included

	namespace llama
	{
	    namespace internal
	    {
	        inline auto color(std::string_view recordCoordTags) -> std::size_t
	        {
	            auto c = std::hash<std::string_view>{}(recordCoordTags) &std::size_t{0xFFFFFF};
	            c |= std::size_t{0x404040}; // ensure color per channel is at least 0x40.
	            return c;
	        }

	        // from: https://stackoverflow.com/questions/5665231/most-efficient-way-to-escape-xml-html-in-c-string
	        inline auto xmlEscape(const std::string& str) -> std::string
	        {
	            std::string result;
	            result.reserve(str.size());
	            for(const char c : str)
	            {
	                switch(c)
	                {
	                case '&':
	                    result.append("&amp;");
	                    break;
	                case '\"':
	                    result.append("&quot;");
	                    break;
	                case '\'':
	                    result.append("&apos;");
	                    break;
	                case '<':
	                    result.append("&lt;");
	                    break;
	                case '>':
	                    result.append("&gt;");
	                    break;
	                default:
	                    result += c;
	                    break;
	                }
	            }
	            return result;
	        }

	        template<typename T, std::size_t Dim>
	        auto formatArrayIndex(const ArrayIndex<T, Dim>& ai)
	        {
	            if constexpr(Dim == 1)
	                return std::to_string(ai[0]);
	            else
	            {
	                std::string s = "{";
	                for(auto v : ai)
	                {
	                    if(s.size() >= 2)
	                        s += ",";
	                    s += std::to_string(v);
	                }
	                s += "}";
	                return s;
	            }
	        }

	        template<typename ArrayIndex>
	        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	        struct FieldBox
	        {
	            ArrayIndex arrayIndex;
	            std::string_view recordCoordTags;
	            NrAndOffset<std::size_t> nrAndOffset;
	            std::size_t size;
	        };

	        template<typename View>
	        void fillBlobsWithPattern(View& view, uint8_t pattern)
	        {
	            const auto& mapping = view.mapping();
	            for(std::size_t i = 0; i < View::Mapping::blobCount; i++)
	                std::memset(&view.blobs()[i][0], pattern, mapping.blobSize(i));
	        }

	        template<typename View, typename RecordCoord>
	        void boxesFromComputedField(
	            View& view,
	            typename View::Mapping::ArrayExtents::Index ai,
	            RecordCoord rc,
	            std::vector<FieldBox<typename View::Mapping::ArrayExtents::Index>>& infos)
	        {
	            using Mapping = typename View::Mapping;
	            using RecordDim = typename Mapping::RecordDim;

	            auto emitInfo = [&](auto nrAndOffset, std::size_t size) {
	                infos.push_back({ai, prettyRecordCoord<RecordDim>(rc), nrAndOffset, size});
	            };

	            using Type = GetType<RecordDim, decltype(rc)>;
	            // computed values can come from anywhere, so we can only apply heuristics
	            auto& blobs = view.blobs();
	            auto&& ref = view.mapping().compute(ai, rc, blobs);

	            // if we get a reference, try to find the mapped address in one of the blobs
	            if constexpr(std::is_lvalue_reference_v<decltype(ref)>)
	            {
	                auto address = reinterpret_cast<std::intptr_t>(&ref);
	                for(std::size_t i = 0; i < blobs.size(); i++)
	                {
	                    // TODO(bgruber): this is UB, because we are comparing pointers from unrelated
	                    // allocations
	                    const auto front = reinterpret_cast<std::intptr_t>(&blobs[i][0]);
	                    const auto back = reinterpret_cast<std::intptr_t>(&blobs[i][view.mapping().blobSize(i) - 1]);
	                    if(front <= address && address <= back)
	                    {
	                        emitInfo(NrAndOffset{i, static_cast<std::size_t>(address - front)}, sizeof(Type));
	                        return; // a mapping can only map to one location in the blobs
	                    }
	                }
	            }

	            if constexpr(std::is_default_constructible_v<Type>)
	            {
	                const auto infosBefore = infos.size();

	                // try to observe written bytes
	                const auto pattern = std::uint8_t{0xFF};
	                fillBlobsWithPattern(view, pattern);
	                ref = Type{}; // a broad range of types is default constructible and should write
	                              // something zero-ish
	                auto wasTouched = [&](auto b) { return static_cast<std::uint8_t>(b) != pattern; };
	                for(std::size_t i = 0; i < Mapping::blobCount; i++)
	                {
	                    const auto blobSize = view.mapping().blobSize(i);
	                    const auto* begin = &blobs[i][0];
	                    const auto* end = begin + blobSize;

	                    auto* searchBegin = begin;
	                    while(true)
	                    {
	                        const auto* touchedBegin = std::find_if(searchBegin, end, wasTouched);
	                        if(touchedBegin == end)
	                            break;
	                        const auto& touchedEnd = std::find_if_not(touchedBegin + 1, end, wasTouched);
	                        emitInfo(
	                            NrAndOffset{i, static_cast<std::size_t>(touchedBegin - begin)},
	                            touchedEnd - touchedBegin);
	                        if(touchedEnd == end)
	                            break;
	                        searchBegin = touchedEnd + 1;
	                    }
	                }

	                if(infosBefore != infos.size())
	                    return;
	            }

	            // if we come here, we could not find out where the value is coming from
	            emitInfo(NrAndOffset{Mapping::blobCount, std::size_t{0}}, sizeof(Type));
	        }

	        template<typename Mapping>
	        auto boxesFromMapping(const Mapping& mapping) -> std::vector<FieldBox<typename Mapping::ArrayExtents::Index>>
	        {
	            std::vector<FieldBox<typename Mapping::ArrayExtents::Index>> infos;

	            std::optional<decltype(allocView(mapping))> view;
	            if constexpr(hasAnyComputedField<Mapping>)
	                view = allocView(mapping);

	            using RecordDim = typename Mapping::RecordDim;
	            for(auto ai : ArrayIndexRange{mapping.extents()})
	                forEachLeafCoord<RecordDim>(
	                    [&](auto rc)
	                    {
	                        using Type = GetType<RecordDim, decltype(rc)>;
	                        if constexpr(llama::isComputed<Mapping, decltype(rc)>)
	                            boxesFromComputedField(view.value(), ai, rc, infos);
	                        else
	                        {
	                            const auto [nr, off] = mapping.blobNrAndOffset(ai, rc);
	                            infos.push_back(
	                                {ai,
	                                 prettyRecordCoord<RecordDim>(rc),
	                                 {static_cast<std::size_t>(nr), static_cast<std::size_t>(off)},
	                                 sizeof(Type)});
	                        }
	                    });

	            return infos;
	        }

	        template<typename ArrayIndex>
	        auto breakBoxes(std::vector<FieldBox<ArrayIndex>> boxes, std::size_t wrapByteCount)
	            -> std::vector<FieldBox<ArrayIndex>>
	        {
	            for(std::size_t i = 0; i < boxes.size(); i++)
	            {
	                auto& fb = boxes[i];
	                if(fb.nrAndOffset.offset / wrapByteCount != (fb.nrAndOffset.offset + fb.size - 1) / wrapByteCount)
	                {
	                    const auto remainingSpace = wrapByteCount - fb.nrAndOffset.offset % wrapByteCount;
	                    auto newFb = fb;
	                    newFb.nrAndOffset.offset = fb.nrAndOffset.offset + remainingSpace;
	                    newFb.size = fb.size - remainingSpace;
	                    fb.size = remainingSpace;
	                    boxes.push_back(newFb);
	                }
	            }
	            return boxes;
	        }

	        inline auto cssClass(std::string tags)
	        {
	            std::replace(begin(tags), end(tags), '.', '_');
	            std::replace(begin(tags), end(tags), '<', '_');
	            std::replace(begin(tags), end(tags), '>', '_');
	            return tags;
	        };
	    } // namespace internal

	    /// Returns an SVG image visualizing the memory layout created by the given mapping. The created memory blocks are
	    /// wrapped after wrapByteCount bytes.
	    LLAMA_EXPORT
	    template<typename Mapping>
	    auto toSvg(const Mapping& mapping, std::size_t wrapByteCount = 64, bool breakBoxes = true) -> std::string
	    {
	        constexpr auto byteSizeInPixel = 30;
	        constexpr auto blobBlockWidth = 60;

	        auto infos = internal::boxesFromMapping(mapping);
	        if(breakBoxes)
	            infos = internal::breakBoxes(std::move(infos), wrapByteCount);
	        std::stable_sort(
	            begin(infos),
	            end(infos),
	            [](const auto& a, const auto& b) {
	                return std::tie(a.nrAndOffset.nr, a.nrAndOffset.offset)
	                    < std::tie(b.nrAndOffset.nr, b.nrAndOffset.offset);
	            });

	        std::string svg;

	        std::array<int, Mapping::blobCount + hasAnyComputedField<Mapping> + 1> blobYOffset{};
	        auto writeBlobHeader = [&](std::size_t i, std::size_t size, std::string_view name)
	        {
	            const auto blobRows = (size + wrapByteCount - 1) / wrapByteCount;
	            blobYOffset[i + 1] = blobYOffset[i] + (blobRows + 1) * byteSizeInPixel; // one row gap between blobs
	            const auto height = blobRows * byteSizeInPixel;
	            svg += fmt::format(
	                R"a(<rect x="0" y="{}" width="{}" height="{}" fill="#AAA" stroke="#000"/>
	<text x="{}" y="{}" fill="#000" text-anchor="middle">{}</text>
	)a",
	                blobYOffset[i],
	                blobBlockWidth,
	                height,
	                blobBlockWidth / 2,
	                blobYOffset[i] + height / 2,
	                name);
	        };
	        for(std::size_t i = 0; i < Mapping::blobCount; i++)
	            writeBlobHeader(i, mapping.blobSize(i), "Blob: " + std::to_string(i));

	        svg = fmt::format(
	                  R"(<?xml version="1.0" encoding="UTF-8" standalone="no"?>
	<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
	    <style>
	        .label {{ font: {}px sans-serif; }}
	    </style>
	)",
	                  blobBlockWidth + wrapByteCount * byteSizeInPixel,
	                  blobYOffset.back() == 0 ? 987654321 : blobYOffset.back() - byteSizeInPixel,
	                  byteSizeInPixel / 2)
	            + svg;

	        std::size_t computedSizeSoFar = 0;
	        std::size_t lastBlobNr = std::numeric_limits<std::size_t>::max();
	        std::size_t usedBytesInBlobSoFar = 0;
	        for(const auto& info : infos)
	        {
	            if(lastBlobNr != info.nrAndOffset.nr)
	            {
	                usedBytesInBlobSoFar = 0;
	                lastBlobNr = info.nrAndOffset.nr;
	            }

	            const auto blobY = blobYOffset[info.nrAndOffset.nr];
	            const auto offset = [&]
	            {
	                if(info.nrAndOffset.nr < Mapping::blobCount)
	                    return info.nrAndOffset.offset;

	                const auto offset = computedSizeSoFar;
	                computedSizeSoFar += info.size;
	                return offset;
	            }();
	            auto x = (offset % wrapByteCount) * byteSizeInPixel + blobBlockWidth;
	            auto y = (offset / wrapByteCount) * byteSizeInPixel + blobY;
	            const auto fill = internal::color(info.recordCoordTags);
	            const auto width = byteSizeInPixel * info.size;

	            const auto nextOffset = [&]
	            {
	                if(&info == &infos.back())
	                    return std::numeric_limits<std::size_t>::max();
	                const auto& nextInfo = (&info)[1];
	                if(info.nrAndOffset.nr < Mapping::blobCount && info.nrAndOffset.nr == nextInfo.nrAndOffset.nr)
	                    return nextInfo.nrAndOffset.offset;

	                return std::numeric_limits<std::size_t>::max();
	            }();
	            const auto isOverlapped = offset < usedBytesInBlobSoFar || nextOffset < offset + info.size;
	            usedBytesInBlobSoFar = offset + info.size;

	            constexpr auto cropBoxes = true;
	            if(cropBoxes)
	            {
	                svg += fmt::format(
	                    R"(<svg x="{}" y="{}" width="{}" height="{}">
	)",
	                    x,
	                    y,
	                    width,
	                    byteSizeInPixel);
	                x = 0;
	                y = 0;
	            }
	            svg += fmt::format(
	                R"(<rect x="{}" y="{}" width="{}" height="{}" fill="#{:X}" stroke="#000" fill-opacity="{}"/>
	)",
	                x,
	                y,
	                width,
	                byteSizeInPixel,
	                fill,
	                isOverlapped ? 0.3 : 1.0);
	            for(std::size_t i = 1; i < info.size; i++)
	            {
	                svg += fmt::format(
	                    R"(<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#777"/>
	)",
	                    x + i * byteSizeInPixel,
	                    y + byteSizeInPixel * 2 / 3,
	                    x + i * byteSizeInPixel,
	                    y + byteSizeInPixel);
	            }
	            svg += fmt::format(
	                R"(<text x="{}" y="{}" fill="#000" text-anchor="middle" class="label">{} {}</text>
	)",
	                x + width / 2,
	                y + byteSizeInPixel * 3 / 4,
	                internal::formatArrayIndex(info.arrayIndex),
	                internal::xmlEscape(std::string{info.recordCoordTags}));
	            if(cropBoxes)
	                svg += R"(</svg>
	)";
	        }

	        if(hasAnyComputedField<Mapping>)
	        {
	            if(computedSizeSoFar > 0)
	                writeBlobHeader(Mapping::blobCount, computedSizeSoFar, "Comp.");
	            else
	            {
	                const auto blobRows = (wrapByteCount - 1) / wrapByteCount;
	                blobYOffset[Mapping::blobCount + 1]
	                    = blobYOffset[Mapping::blobCount] + blobRows * byteSizeInPixel; // fix-up, omit gap
	            }

	            // fix total SVG size
	            const auto i = svg.find("987654321");
	            assert(i != std::string::npos);
	            svg.replace(i, 9, std::to_string(blobYOffset.back() - byteSizeInPixel));
	        }

	        svg += "</svg>";
	        return svg;
	    }

	    /// Returns an HTML document visualizing the memory layout created by the given mapping. The visualization is
	    /// resizeable.
	    LLAMA_EXPORT
	    template<typename Mapping>
	    auto toHtml(const Mapping& mapping) -> std::string
	    {
	        constexpr auto byteSizeInPixel = 30;
	        constexpr auto rulerLengthInBytes = 512;
	        constexpr auto rulerByteInterval = 8;

	        auto infos = internal::boxesFromMapping(mapping);
	        std::stable_sort(
	            begin(infos),
	            end(infos),
	            [](const auto& a, const auto& b) {
	                return std::tie(a.nrAndOffset.nr, a.nrAndOffset.offset)
	                    < std::tie(b.nrAndOffset.nr, b.nrAndOffset.offset);
	            });
	        infos.erase(
	            std::unique(
	                begin(infos),
	                end(infos),
	                [](const auto& a, const auto& b) { return a.nrAndOffset == b.nrAndOffset; }),
	            end(infos));

	        std::string html;
	        html += fmt::format(
	            R"(<!DOCTYPE html>
	<html>
	<head>
	<style>
	.box {{
	    outline: 1px solid;
	    display: inline-block;
	    white-space: nowrap;
	    height: {}px;
	    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #777 29px, #777 30px);
	    text-align: center;
	    overflow: hidden;
	    vertical-align: middle;
	}}
	#ruler {{
	    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #000 29px, #000 30px);
	    border-bottom: 1px solid;
	    height: 20px;
	    margin-bottom: 20px;
	}}
	#ruler div {{
	    position: absolute;
	    display: inline-block;
	}}
	)",
	            byteSizeInPixel);
	        using RecordDim = typename Mapping::RecordDim;
	        forEachLeafCoord<RecordDim>(
	            [&](auto rc)
	            {
	                constexpr int size = sizeof(GetType<RecordDim, decltype(rc)>);

	                html += fmt::format(
	                    R"(.{} {{
	    width: {}px;
	    background-color: #{:X};
	}}
	)",
	                    internal::cssClass(std::string{prettyRecordCoord<RecordDim>(rc)}),
	                    byteSizeInPixel * size,
	                    internal::color(prettyRecordCoord<RecordDim>(rc)));
	            });

	        html += fmt::format(R"(</style>
	</head>
	<body>
	    <header id="ruler">
	)");
	        for(auto i = 0; i < rulerLengthInBytes; i += rulerByteInterval)
	            html += fmt::format(
	                R"(</style>
	        <div style="margin-left: {}px;">{}</div>)",
	                i * byteSizeInPixel,
	                i);
	        html += fmt::format(R"(
	    </header>
	)");

	        auto currentBlobNr = std::numeric_limits<std::size_t>::max();
	        for(const auto& info : infos)
	        {
	            if(currentBlobNr != info.nrAndOffset.nr)
	            {
	                currentBlobNr = info.nrAndOffset.nr;
	                html += fmt::format("<h1>Blob: {}</h1>", currentBlobNr);
	            }
	            html += fmt::format(
	                R"(<div class="box {0}" title="{1} {2}">{1} {2}</div>)",
	                internal::cssClass(std::string{info.recordCoordTags}),
	                internal::formatArrayIndex(info.arrayIndex),
	                internal::xmlEscape(std::string{info.recordCoordTags}));
	        }
	        html += R"(</body>
	</html>)";
	        return html;
	    }
	} // namespace llama

	#endif
	// ==
	// == ./include/llama/DumpMapping.hpp ==
	// ============================================================================

// #include "Meta.hpp"    // amalgamate: file already inlined
// #include "ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
// #include "RecordRef.hpp"    // amalgamate: file already inlined
// #include "Simd.hpp"    // amalgamate: file already inlined
// #include "StructName.hpp"    // amalgamate: file already inlined
// #include "Tuple.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/Vector.hpp ==
	// ==
	// Copyright 2021 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "RecordRef.hpp"    // amalgamate: file already inlined
	// #include "View.hpp"    // amalgamate: file already inlined

	// #include <algorithm>    // amalgamate: file already included
	// #include <stdexcept>    // amalgamate: file already included
	// #include <string>    // amalgamate: file already included

	namespace llama
	{
	    // TODO(bgruber): expose blob allocator
	    /// An equivalent of std::vector<T> backed by a \ref View. Elements are never value initialized though. No strong
	    /// exception guarantee.
	    /// WARNING: This class is experimental.
	    /// @tparam Mapping The mapping to be used for the underlying view. Needs to have 1 array dimension.
	    LLAMA_EXPORT
	    template<typename Mapping>
	    struct Vector
	    {
	        static_assert(Mapping::ArrayExtents::rank == 1, "llama::Vector only supports 1D mappings");

	        using ViewType = decltype(allocViewUninitialized<Mapping>());
	        using RecordDim = typename Mapping::RecordDim;

	        using iterator = decltype(std::declval<ViewType>().begin());
	        using value_type = typename iterator::value_type;
	        using size_type = typename Mapping::ArrayExtents::value_type;

	        Vector() = default;

	        template<typename RecordRef = One<RecordDim>>
	        LLAMA_FN_HOST_ACC_INLINE explicit Vector(size_type count, const RecordRef& value = {})
	        {
	            reserve(count);
	            for(size_type i = 0; i < count; i++)
	                push_back(value);
	        }

	        template<typename Iterator>
	        LLAMA_FN_HOST_ACC_INLINE Vector(Iterator first, Iterator last)
	        {
	            if constexpr(std::is_same_v<
	                             typename std::iterator_traits<Iterator>::iterator_category,
	                             std::random_access_iterator_tag>)
	                reserve(std::distance(first, last));
	            for(; first != last; ++first)
	                push_back(*first);
	        }

	        Vector(const Vector& other) = default;

	        LLAMA_FN_HOST_ACC_INLINE Vector(Vector&& other) noexcept
	        {
	            swap(other);
	        }

	        auto operator=(const Vector& other) -> Vector& = default;

	        LLAMA_FN_HOST_ACC_INLINE auto operator=(Vector&& other) noexcept -> Vector&
	        {
	            swap(other);
	            return *this;
	        }

	        ~Vector() = default;

	        // TODO(bgruber): assign

	    private:
	        LLAMA_FN_HOST_ACC_INLINE void outOfRange([[maybe_unused]] size_type i) const
	        {
	#if __CUDA_ARCH__
	            assert(false && "Index out of range");
	#else
	            throw std::out_of_range{"Index " + std::to_string(i) + "out of range [0:" + std::to_string(m_size) + "["};
	#endif
	        }

	    public:
	        LLAMA_FN_HOST_ACC_INLINE auto at(size_type i) -> decltype(auto)
	        {
	            if(i >= m_size)
	                outOfRange(i);
	            return m_view(i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto at(size_type i) const -> decltype(auto)
	        {
	            if(i >= m_size)
	                outOfRange(i);
	            return m_view(i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type i) -> decltype(auto)
	        {
	            return m_view(i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type i) const -> decltype(auto)
	        {
	            return m_view(i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto front() -> decltype(auto)
	        {
	            return m_view(0);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto front() const -> decltype(auto)
	        {
	            return m_view(0);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto back() -> decltype(auto)
	        {
	            return m_view(m_size - 1);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto back() const -> decltype(auto)
	        {
	            return m_view(m_size - 1);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto begin() -> decltype(auto)
	        {
	            return m_view.begin();
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto begin() const -> decltype(auto)
	        {
	            return m_view.begin();
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto cbegin() -> decltype(auto)
	        {
	            return std::as_const(m_view).begin();
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto cbegin() const -> decltype(auto)
	        {
	            return m_view.begin();
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto end() -> decltype(auto)
	        {
	            return m_view.begin() + m_size;
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto end() const -> decltype(auto)
	        {
	            return m_view.begin() + m_size;
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto cend() -> decltype(auto)
	        {
	            return std::as_const(m_view).begin() + m_size;
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto cend() const -> decltype(auto)
	        {
	            return m_view.begin() + m_size;
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto empty() const -> bool
	        {
	            return m_size == 0;
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto size() const -> size_type
	        {
	            return m_size;
	        }

	        LLAMA_FN_HOST_ACC_INLINE void reserve(size_type cap)
	        {
	            if(cap > capacity())
	                changeCapacity(cap);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto capacity() const -> size_type
	        {
	            return m_view.extents()[0];
	        }

	        // NOLINTNEXTLINE(readability-identifier-naming)
	        LLAMA_FN_HOST_ACC_INLINE void shrink_to_fit()
	        {
	            changeCapacity(m_size);
	        }

	        LLAMA_FN_HOST_ACC_INLINE void clear()
	        {
	            m_size = 0;
	        }

	        template<typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto insert(iterator pos, T&& t) -> iterator
	        {
	            const auto i = pos - begin();
	            reserve(m_size + 1); // might invalidate pos
	            pos = begin() + i;
	            std::copy_backward(pos, end(), end() + 1);
	            m_view[i] = std::forward<T>(t);
	            m_size++;
	            return pos;
	        }

	        // TODO(bgruber): more insert overloads

	        // TODO(bgruber): emplace

	        LLAMA_FN_HOST_ACC_INLINE auto erase(iterator pos) -> iterator
	        {
	            std::copy(pos + 1, end(), pos);
	            m_size--;
	            return pos;
	        }

	        // TODO(bgruber): more erase overloads

	        // TODO(bgruber): T here is probably a RecordRef. We could also allow any struct that is storable to the
	        // view via RecordRef::store().
	        template<typename T>
	        // NOLINTNEXTLINE(readability-identifier-naming)
	        LLAMA_FN_HOST_ACC_INLINE void push_back(T&& t)
	        {
	            if(const auto cap = capacity(); m_size == cap)
	                reserve(std::max(cap + cap / 2, m_size + 1));

	            m_view[m_size++] = std::forward<T>(t);
	        }

	        // TODO(bgruber): emplace_back

	        // NOLINTNEXTLINE(readability-identifier-naming)
	        LLAMA_FN_HOST_ACC_INLINE void pop_back()
	        {
	            m_size--;
	        }

	        template<typename RecordRef = One<RecordDim>>
	        LLAMA_FN_HOST_ACC_INLINE void resize(size_type count, const RecordRef& value = {})
	        {
	            reserve(count);
	            for(size_type i = m_size; i < count; i++)
	                m_view[i] = value;
	            m_size = count;
	        }

	        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const Vector& a, const Vector& b) -> bool
	        {
	            if(a.m_size != b.m_size)
	                return false;
	            return std::equal(a.begin(), a.end(), b.begin());
	        }

	        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const Vector& a, const Vector& b) -> bool
	        {
	            return !(a == b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const Vector& a, const Vector& b) -> bool
	        {
	            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
	        }

	        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const Vector& a, const Vector& b) -> bool
	        {
	            return !(b < a);
	        }

	        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const Vector& a, const Vector& b) -> bool
	        {
	            return b < a;
	        }

	        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const Vector& a, const Vector& b) -> bool
	        {
	            return !(a < b);
	        }

	        LLAMA_FN_HOST_ACC_INLINE friend void swap(Vector& a, Vector& b) noexcept
	        {
	            a.swap(b);
	        }

	    private:
	        LLAMA_FN_HOST_ACC_INLINE void changeCapacity(size_type cap)
	        {
	            auto newView = allocViewUninitialized<Mapping>(Mapping{typename Mapping::ArrayExtents{cap}});
	            auto b = begin();
	            std::copy(begin(), b + std::min(m_size, cap), newView.begin());
	            using std::swap;
	            swap(m_view, newView); // depends on move semantic of View
	        }

	        LLAMA_FN_HOST_ACC_INLINE void swap(Vector& other) noexcept
	        {
	            using std::swap;
	            swap(m_view, other.m_view); // depends on move semantic of View
	            swap(m_size, other.m_size);
	        }

	        ViewType m_view = {};
	        size_type m_size = 0;
	    };
	} // namespace llama
	// ==
	// == ./include/llama/Vector.hpp ==
	// ============================================================================

// #include "View.hpp"    // amalgamate: file already inlined
// #include "macros.hpp"    // amalgamate: file already inlined
// #include "mapping/AoS.hpp"    // amalgamate: file already inlined
// #include "mapping/AoSoA.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/mapping/BitPackedFloat.hpp ==
	// ==
	// Copyright 2023 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
		// ============================================================================
		// == ./include/llama/mapping/BitPackedInt.hpp ==
		// ==
		// Copyright 2023 Bernhard Manfred Gruber
		// SPDX-License-Identifier: MPL-2.0

		// #pragma once
		// #include "../Core.hpp"    // amalgamate: file already inlined
		// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
		// #include "Common.hpp"    // amalgamate: file already inlined

		// #include <climits>    // amalgamate: file already included
		// #include <type_traits>    // amalgamate: file already included

		namespace llama::mapping
		{
		    LLAMA_EXPORT
		    enum class SignBit
		    {
		        Keep,
		        Discard
		    };

		    namespace internal
		    {
		        template<typename Integral>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto makeMask(Integral bits) -> Integral
		        {
		            return bits >= sizeof(Integral) * CHAR_BIT ? ~Integral{0} : (Integral{1} << bits) - 1u;
		        }

		        template<bool KeepSignBit, typename Integral, typename StoredIntegral>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto bitunpack(
		            const StoredIntegral* ptr,
		            StoredIntegral bitOffset,
		            StoredIntegral bitCount) -> Integral
		        {
		            constexpr auto bitsPerIntegral = static_cast<StoredIntegral>(sizeof(Integral) * CHAR_BIT);
		            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
		            static_assert(bitsPerIntegral <= bitsPerStoredIntegral);
		            assert(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
		#ifdef __clang__
		            // this is necessary to silence the clang static analyzer
		            __builtin_assume(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
		#endif

		            const auto* p = ptr + bitOffset / bitsPerStoredIntegral;
		            const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;
		            //            assert(p < endPtr);
		            auto v = p[0] >> innerBitOffset;

		            const auto innerBitEndOffset = innerBitOffset + bitCount;
		            if(innerBitEndOffset <= bitsPerStoredIntegral)
		            {
		                const auto mask = makeMask(bitCount);
		                v &= mask;
		            }
		            else
		            {
		                const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
		                const auto bitsLoaded = bitsPerStoredIntegral - innerBitOffset;
		                const auto mask = makeMask(excessBits);
		                //                assert(p + 1 < endPtr);
		                v |= (p[1] & mask) << bitsLoaded;
		            }
		            if constexpr(std::is_signed_v<Integral> && KeepSignBit)
		            {
		                // perform sign extension
		                if((v & (StoredIntegral{1} << (bitCount - 1))) && bitCount < bitsPerStoredIntegral)
		                    v |= ~StoredIntegral{0} << bitCount;
		            }
		            return static_cast<Integral>(v);
		        }

		        template<bool KeepSignBit, typename StoredIntegral, typename Integral>
		        LLAMA_FN_HOST_ACC_INLINE constexpr void bitpack(
		            StoredIntegral* ptr,
		            StoredIntegral bitOffset,
		            StoredIntegral bitCount,
		            Integral value)
		        {
		            constexpr auto bitsPerIntegral = static_cast<StoredIntegral>(sizeof(Integral) * CHAR_BIT);
		            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
		            static_assert(bitsPerIntegral <= bitsPerStoredIntegral);
		            assert(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
		#ifdef __clang__
		            // this is necessary to silence the clang static analyzer
		            __builtin_assume(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
		#endif

		            // NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
		            const auto unsignedValue = static_cast<StoredIntegral>(value);
		            const auto mask = makeMask(bitCount);
		            StoredIntegral valueBits;
		            if constexpr(std::is_signed_v<Integral> && KeepSignBit)
		            {
		                const auto magnitudeMask = makeMask(bitCount - 1);
		                const auto isSigned = value < 0;
		                valueBits = (StoredIntegral{isSigned} << (bitCount - 1)) | (unsignedValue & magnitudeMask);
		            }
		            else
		            {
		                valueBits = unsignedValue & mask;
		            }

		            auto* p = ptr + bitOffset / bitsPerStoredIntegral;
		            const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;

		            {
		                const auto clearMask = ~(mask << innerBitOffset);
		                //                assert(p < endPtr);
		                auto mem = p[0] & clearMask; // clear previous bits
		                mem |= valueBits << innerBitOffset; // write new bits
		                p[0] = mem;
		            }

		            const auto innerBitEndOffset = innerBitOffset + bitCount;
		            if(innerBitEndOffset > bitsPerStoredIntegral)
		            {
		                const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
		                const auto bitsWritten = bitsPerStoredIntegral - innerBitOffset;
		                const auto clearMask = ~makeMask(excessBits);
		                //                assert(p + 1 < endPtr);
		                auto mem = p[1] & clearMask; // clear previous bits
		                mem |= valueBits >> bitsWritten; // write new bits
		                p[1] = mem;
		            }
		        }

		        template<typename Integral, typename StoredIntegral>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto bitunpack1(const StoredIntegral* ptr, StoredIntegral bitOffset)
		            -> Integral
		        {
		            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
		            const auto bit
		                = (ptr[bitOffset / bitsPerStoredIntegral] >> (bitOffset % bitsPerStoredIntegral)) & StoredIntegral{1};
		            return static_cast<Integral>(bit);
		        }

		        template<typename StoredIntegral, typename Integral>
		        LLAMA_FN_HOST_ACC_INLINE constexpr void bitpack1(StoredIntegral* ptr, StoredIntegral bitOffset, Integral value)
		        {
		            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
		            const auto bitOff = bitOffset % bitsPerStoredIntegral;
		            auto& dst = ptr[bitOffset / bitsPerStoredIntegral];
		            dst &= ~(StoredIntegral{1} << bitOff); // clear bit
		            const auto bit = (static_cast<StoredIntegral>(value) & StoredIntegral{1});
		            dst |= (bit << bitOff); // set bit
		        }

		        /// A proxy type representing a reference to a reduced precision integral value, stored in a buffer at a
		        /// specified bit offset.
		        /// @tparam Integral Integral data type which can be loaded and store through this reference.
		        /// @tparam StoredIntegralCV Integral type used for storing the bits with CV qualifiers.
		        /// @tparam SizeType Type used to store sizes and offsets.
		        template<typename Integral, typename StoredIntegralCV, typename VHBits, typename SizeType, SignBit SignBit>
		        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
		        struct BitPackedIntRef
		            : private VHBits
		            , ProxyRefOpMixin<BitPackedIntRef<Integral, StoredIntegralCV, VHBits, SizeType, SignBit>, Integral>
		        {
		        private:
		            using StoredIntegral = std::remove_cv_t<StoredIntegralCV>;
		            StoredIntegralCV* ptr;
		            SizeType bitOffset;

		        public:
		            using value_type = Integral;

		            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedIntRef(
		                StoredIntegralCV* ptr,
		                SizeType bitOffset,
		                VHBits vhBits)
		                : VHBits{vhBits}
		                , ptr{ptr}
		                , bitOffset{bitOffset}
		            {
		            }

		            BitPackedIntRef(const BitPackedIntRef&) = default;

		            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
		            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const BitPackedIntRef& other) -> BitPackedIntRef&
		            {
		                *this = static_cast<value_type>(other);
		                return *this;
		            }

		            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
		            LLAMA_FN_HOST_ACC_INLINE constexpr operator Integral() const
		            {
		                // fast path for single bits without sign handling
		                if constexpr(std::is_empty_v<VHBits>)
		                {
		                    if constexpr(VHBits::value() == 1 && (std::is_unsigned_v<Integral> || SignBit == SignBit::Discard))
		                    {
		                        return bitunpack1<Integral>(ptr, static_cast<StoredIntegral>(bitOffset));
		                    }
		                }

		                return bitunpack<SignBit == SignBit::Keep, Integral>(
		                    ptr,
		                    static_cast<StoredIntegral>(bitOffset),
		                    static_cast<StoredIntegral>(VHBits::value()));
		            }

		            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Integral value) -> BitPackedIntRef&
		            {
		                // fast path for single bits without sign handling
		                if constexpr(std::is_empty_v<VHBits>)
		                {
		                    if constexpr(VHBits::value() == 1 && (std::is_unsigned_v<Integral> || SignBit == SignBit::Discard))
		                    {
		                        bitpack1(ptr, static_cast<StoredIntegral>(bitOffset), value);
		                    }
		                }

		                bitpack<SignBit == SignBit::Keep>(
		                    ptr,
		                    static_cast<StoredIntegral>(bitOffset),
		                    static_cast<StoredIntegral>(VHBits::value()),
		                    value);
		                return *this;
		            }
		        };

		        template<typename A, typename B>
		        using HasLargerSize = mp_bool<sizeof(A) < sizeof(B)>;

		        template<typename RecordDim>
		        using LargestIntegral = mp_max_element<FlatRecordDim<RecordDim>, HasLargerSize>;

		        template<typename RecordDim>
		        using StoredUnsignedFor = std::
		            conditional_t<(sizeof(LargestIntegral<RecordDim>) > sizeof(std::uint32_t)), std::uint64_t, std::uint32_t>;

		        template<
		            typename TArrayExtents,
		            typename TRecordDim,
		            typename Bits,
		            SignBit SignBit,
		            typename TLinearizeArrayIndexFunctor,
		            typename TStoredIntegral>
		        struct BitPackedIntCommon
		            : MappingBase<TArrayExtents, TRecordDim>
		            , protected llama::internal::BoxedValue<Bits>
		        {
		            using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
		            using StoredIntegral = TStoredIntegral;

		            static_assert(std::is_integral_v<StoredIntegral>);
		            static_assert(std::is_unsigned_v<StoredIntegral>);

		            // We could allow more integer types as storage type, but that needs to be thought through carefully
		            static_assert(
		                std::is_same_v<StoredIntegral, std::uint32_t> || std::is_same_v<StoredIntegral, std::uint64_t>);

		        protected:
		            using Base = MappingBase<TArrayExtents, TRecordDim>;
		            using VHBits = llama::internal::BoxedValue<Bits>;
		            using size_type = typename TArrayExtents::value_type;

		            template<typename T>
		            using IsAllowedFieldType = mp_or<std::is_integral<T>, std::is_enum<T>>;

		            static_assert(
		                mp_all_of<FlatRecordDim<TRecordDim>, IsAllowedFieldType>::value,
		                "All record dimension field types must be integral");

		            template<typename T>
		            using IsFieldTypeSmallerOrEqualStorageIntegral = mp_bool<sizeof(T) <= sizeof(StoredIntegral)>;

		            static_assert(
		                mp_all_of<FlatRecordDim<TRecordDim>, IsFieldTypeSmallerOrEqualStorageIntegral>::value,
		                "The integral type used for storage must be at least as big as the type of the values to retrieve");

		        public:
		            LLAMA_FN_HOST_ACC_INLINE
		            constexpr auto bits() const -> size_type
		            {
		                return static_cast<size_type>(VHBits::value());
		            }

		            template<typename B = Bits, std::enable_if_t<isConstant<B>, int> = 0>
		            LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntCommon(
		                TArrayExtents extents = {},
		                Bits bits = {},
		                TRecordDim = {})
		                : Base(extents)
		                , VHBits{bits}
		            {
		                static_assert(VHBits::value() > 0);
		                mp_for_each<mp_transform<mp_identity, FlatRecordDim<TRecordDim>>>(
		                    [&](auto t)
		                    {
		                        using FieldType = typename decltype(t)::type;
		                        static_assert(
		                            static_cast<std::size_t>(VHBits::value()) <= sizeof(FieldType) * CHAR_BIT,
		                            "Storage bits must not be greater than bits of field type");
		                        static_assert(
		                            VHBits::value() >= 2
		                                || std::is_unsigned_v<FieldType> || SignBit == llama::mapping::SignBit::Discard,
		                            "When keeping the sign bit, Bits must be at least 2 with signed integers in the record "
		                            "dimension");
		                    });
		            }

		            template<typename B = Bits, std::enable_if_t<!isConstant<B>, int> = 0>
		            LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntCommon(
		                TArrayExtents extents,
		                Bits bits,
		                TRecordDim = {})
		                : Base(extents)
		                , VHBits{bits}
		            {
		#ifdef __CUDA_ARCH__
		                assert(VHBits::value() > 0);
		#else
		                if(VHBits::value() <= 0)
		                    throw std::invalid_argument("BitPackedInt* Bits must not be zero");
		#endif
		                mp_for_each<mp_transform<mp_identity, FlatRecordDim<TRecordDim>>>(
		                    [&](auto t)
		                    {
		                        using FieldType [[maybe_unused]] = typename decltype(t)::type;
		#ifdef __CUDA_ARCH__
		                        assert(VHBits::value() <= sizeof(FieldType) * CHAR_BIT);
		#else
		                        if(static_cast<std::size_t>(VHBits::value()) > sizeof(FieldType) * CHAR_BIT)
		                            throw std::invalid_argument(
		                                "BitPackedInt* Bits must not be larger than any field type in the record dimension");
		                        if(!(VHBits::value() >= 2
		                             || std::is_unsigned_v<FieldType> || SignBit == llama::mapping::SignBit::Discard))
		                            throw std::invalid_argument("When keeping the sign bit, Bits must be at least 2 with "
		                                                        "signed integers in the record "
		                                                        "dimension");
		#endif
		                    });
		            }

		            template<std::size_t... RecordCoords>
		            static constexpr auto isComputed(RecordCoord<RecordCoords...>)
		            {
		                return true;
		            }
		        };
		    } // namespace internal

		    /// Struct of array mapping using bit packing to reduce size/precision of integral data types. If your record
		    /// dimension contains non-integral types, split them off using the \ref Split mapping first.
		    /// \tparam Bits If Bits is llama::Constant<N>, the compile-time N specifies the number of bits to use. If Bits is
		    /// an integral type T, the number of bits is specified at runtime, passed to the constructor and stored as type T.
		    /// Must not be zero and must not be bigger than the bits of TStoredIntegral.
		    /// @tparam SignBit When set to SignBit::Discard, discards the sign bit when storing signed integers. All
		    /// numbers will be read back positive.
		    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
		    /// how big the linear domain gets.
		    /// \tparam TStoredIntegral Integral type used as storage of reduced precision integers. Must be std::uint32_t or
		    /// std::uint64_t.
		    LLAMA_EXPORT
		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        typename Bits = typename TArrayExtents::value_type,
		        SignBit SignBit = SignBit::Keep,
		        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
		        typename TStoredIntegral = internal::StoredUnsignedFor<TRecordDim>>
		    struct BitPackedIntSoA
		        : internal::BitPackedIntCommon<
		              TArrayExtents,
		              TRecordDim,
		              Bits,
		              SignBit,
		              TLinearizeArrayIndexFunctor,
		              TStoredIntegral>
		    {
		    private:
		        using Base = internal::
		            BitPackedIntCommon<TArrayExtents, TRecordDim, Bits, SignBit, TLinearizeArrayIndexFunctor, TStoredIntegral>;

		    public:
		        using Base::Base;
		        using typename Base::size_type;
		        using VHBits = typename Base::VHBits; // use plain using declaration with nvcc >= 11.8

		        static constexpr std::size_t blobCount = mp_size<FlatRecordDim<TRecordDim>>::value;

		        LLAMA_FN_HOST_ACC_INLINE
		        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
		        {
		            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(TStoredIntegral) * CHAR_BIT);
		            const auto bitsNeeded = TLinearizeArrayIndexFunctor{}.size(Base::extents()) * VHBits::value();
		            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
		        }

		        template<std::size_t... RecordCoords, typename Blobs>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
		            typename Base::ArrayIndex ai,
		            RecordCoord<RecordCoords...>,
		            Blobs& blobs) const
		        {
		            constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
		            const auto bitOffset = TLinearizeArrayIndexFunctor{}(ai, Base::extents()) * VHBits::value();

		            using QualifiedStoredIntegral = CopyConst<Blobs, TStoredIntegral>;
		            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
		            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
		            return internal::BitPackedIntRef<DstType, QualifiedStoredIntegral, VHBits, size_type, SignBit>{
		                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
		                bitOffset,
		                static_cast<const VHBits&>(*this)};
		            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
		        }
		    };

		    /// Binds parameters to a \ref BitPackedIntSoA mapping except for array and record dimension, producing a quoted
		    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
		    LLAMA_EXPORT
		    template<
		        typename Bits = void,
		        SignBit SignBit = SignBit::Keep,
		        typename LinearizeArrayIndexFunctor = mapping::LinearizeArrayIndexRight,
		        typename StoredIntegral = void>
		    struct BindBitPackedIntSoA
		    {
		        template<typename ArrayExtents, typename RecordDim>
		        using fn = BitPackedIntSoA<
		            ArrayExtents,
		            RecordDim,
		            std::conditional_t<!std::is_void_v<Bits>, Bits, typename ArrayExtents::value_type>,
		            SignBit,
		            LinearizeArrayIndexFunctor,
		            std::conditional_t<
		                !std::is_void_v<StoredIntegral>,
		                StoredIntegral,
		                internal::StoredUnsignedFor<RecordDim>>>;
		    };

		    LLAMA_EXPORT
		    template<typename Mapping>
		    inline constexpr bool isBitPackedIntSoA = false;

		    LLAMA_EXPORT
		    template<
		        typename ArrayExtents,
		        typename RecordDim,
		        typename Bits,
		        SignBit SignBit,
		        typename LinearizeArrayIndexFunctor,
		        typename StoredIntegral>
		    inline constexpr bool isBitPackedIntSoA<
		        BitPackedIntSoA<ArrayExtents, RecordDim, Bits, SignBit, LinearizeArrayIndexFunctor, StoredIntegral>>
		        = true;

		    /// Array of struct mapping using bit packing to reduce size/precision of integral data types. If your record
		    /// dimension contains non-integral types, split them off using the \ref Split mapping first.
		    /// \tparam Bits If Bits is llama::Constant<N>, the compile-time N specifies the number of bits to use. If Bits is
		    /// an integral type T, the number of bits is specified at runtime, passed to the constructor and stored as type T.
		    /// Must not be zero and must not be bigger than the bits of TStoredIntegral.
		    /// @tparam SignBit When set to SignBit::Discard, discards the sign bit when storing signed integers. All
		    /// numbers will be read back positive.
		    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
		    /// how big the linear domain gets.
		    /// \tparam PermuteFields Defines how the record dimension's fields should be permuted. See \ref
		    //  PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref PermuteFieldsDecreasingAlignment and
		    //  \ref PermuteFieldsMinimizePadding.
		    /// \tparam TStoredIntegral Integral type used as storage of reduced precision integers. Must be std::uint32_t or
		    /// std::uint64_t.
		    LLAMA_EXPORT
		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        typename Bits = typename TArrayExtents::value_type,
		        SignBit SignBit = SignBit::Keep,
		        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
		        template<typename> typename PermuteFields = PermuteFieldsInOrder,
		        typename TStoredIntegral = internal::StoredUnsignedFor<TRecordDim>>
		    struct BitPackedIntAoS
		        : internal::BitPackedIntCommon<
		              TArrayExtents,
		              TRecordDim,
		              Bits,
		              SignBit,
		              TLinearizeArrayIndexFunctor,
		              TStoredIntegral>
		    {
		    private:
		        using Base = internal::
		            BitPackedIntCommon<TArrayExtents, TRecordDim, Bits, SignBit, TLinearizeArrayIndexFunctor, TStoredIntegral>;

		    public:
		        using Base::Base;
		        using typename Base::size_type;
		        using VHBits = typename Base::VHBits; // use plain using declaration with nvcc >= 11.8

		        using Permuter = PermuteFields<TRecordDim>;
		        static constexpr std::size_t blobCount = 1;

		        LLAMA_FN_HOST_ACC_INLINE
		        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
		        {
		            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(TStoredIntegral) * CHAR_BIT);
		            const auto bitsNeeded = TLinearizeArrayIndexFunctor{}.size(Base::extents())
		                * static_cast<size_type>(VHBits::value()) * static_cast<size_type>(flatFieldCount<TRecordDim>);
		            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
		        }

		        template<std::size_t... RecordCoords, typename Blobs>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
		            typename Base::ArrayIndex ai,
		            RecordCoord<RecordCoords...>,
		            Blobs& blobs) const
		        {
		            constexpr auto flatFieldIndex = static_cast<size_type>(
		                Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>);
		            const auto bitOffset = ((TLinearizeArrayIndexFunctor{}(ai, Base::extents())
		                                     * static_cast<size_type>(flatFieldCount<TRecordDim>))
		                                    + flatFieldIndex)
		                * static_cast<size_type>(VHBits::value());

		            using QualifiedStoredIntegral = CopyConst<Blobs, TStoredIntegral>;
		            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
		            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
		            return internal::BitPackedIntRef<DstType, QualifiedStoredIntegral, VHBits, size_type, SignBit>{
		                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[0][0]),
		                bitOffset,
		                static_cast<const VHBits&>(*this)};
		            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
		        }
		    };

		    /// Binds parameters to a \ref BitPackedIntAoS mapping except for array and record dimension, producing a quoted
		    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
		    LLAMA_EXPORT
		    template<
		        typename Bits = void,
		        SignBit SignBit = SignBit::Keep,
		        typename LinearizeArrayIndexFunctor = mapping::LinearizeArrayIndexRight,
		        template<typename> typename PermuteFields = PermuteFieldsInOrder,
		        typename StoredIntegral = void>
		    struct BindBitPackedIntAoS
		    {
		        template<typename ArrayExtents, typename RecordDim>
		        using fn = BitPackedIntAoS<
		            ArrayExtents,
		            RecordDim,
		            std::conditional_t<!std::is_void_v<Bits>, Bits, typename ArrayExtents::value_type>,
		            SignBit,
		            LinearizeArrayIndexFunctor,
		            PermuteFields,
		            std::conditional_t<
		                !std::is_void_v<StoredIntegral>,
		                StoredIntegral,
		                internal::StoredUnsignedFor<RecordDim>>>;
		    };

		    LLAMA_EXPORT
		    template<typename Mapping>
		    inline constexpr bool isBitPackedIntAoS = false;

		    template<
		        typename ArrayExtents,
		        typename RecordDim,
		        typename Bits,
		        SignBit SignBit,
		        typename LinearizeArrayIndexFunctor,
		        template<typename>
		        typename PermuteFields,
		        typename StoredIntegral>
		    inline constexpr bool isBitPackedIntAoS<BitPackedIntAoS<
		        ArrayExtents,
		        RecordDim,
		        Bits,
		        SignBit,
		        LinearizeArrayIndexFunctor,
		        PermuteFields,
		        StoredIntegral>>
		        = true;
		} // namespace llama::mapping
		// ==
		// == ./include/llama/mapping/BitPackedInt.hpp ==
		// ============================================================================

	// #include "Common.hpp"    // amalgamate: file already inlined

	// #include <algorithm>    // amalgamate: file already included
	// #include <climits>    // amalgamate: file already included
	#include <cstdint>
	// #include <cstring>    // amalgamate: file already included
	// #include <limits>    // amalgamate: file already included
	// #include <type_traits>    // amalgamate: file already included

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template<typename T>
	        struct FloatBitTraits;

	        template<>
	        struct FloatBitTraits<float>
	        {
	            inline static constexpr unsigned mantissa = 23;
	            inline static constexpr unsigned exponent = 8;
	        };

	        template<>
	        struct FloatBitTraits<double>
	        {
	            inline static constexpr unsigned mantissa = 52;
	            inline static constexpr unsigned exponent = 11;
	        };

	        template<typename Integral>
	        LLAMA_FN_HOST_ACC_INLINE auto repackFloat(
	            Integral inFloat,
	            unsigned inMantissaBits,
	            unsigned inExponentBits,
	            unsigned outMantissaBits,
	            unsigned outExponentBits) -> Integral
	        {
	            const Integral inMantissaMask = (Integral{1} << inMantissaBits) - 1u;
	            const Integral inExponentMask = (Integral{1} << inExponentBits) - 1u;

	            Integral inMantissa = inFloat & inMantissaMask;
	            const Integral inExponent = (inFloat >> inMantissaBits) & inExponentMask;
	            const Integral inSign = inFloat >> inExponentBits >> inMantissaBits;

	            const Integral outExponentMask = (Integral{1} << outExponentBits) - 1u;
	            Integral outExponent;
	            if(inExponent == inExponentMask) [[LLAMA_UNLIKELY]]
	                outExponent = outExponentMask; // propagate +/- inf/nan
	            else if(inExponent == 0) [[LLAMA_UNLIKELY]]
	                outExponent = 0; // propagate -/+ zero
	            else
	            {
	                const int outExponentMax = 1 << (outExponentBits - 1); // NOLINT(hicpp-signed-bitwise)
	                const int outExponentMin = -outExponentMax + 1;
	                const int outExponentBias = outExponentMax - 1;
	                const int inExponentBias = (1 << (inExponentBits - 1)) - 1; // NOLINT(hicpp-signed-bitwise)

	                const int exponent = static_cast<int>(inExponent) - inExponentBias;
	                const auto clampedExponent = std::clamp(exponent, outExponentMin, outExponentMax);
	                if(clampedExponent == outExponentMin || clampedExponent == outExponentMax)
	                    inMantissa = 0; // when exponent changed, let value become inf and not nan
	                outExponent = clampedExponent + outExponentBias;
	            }
	            assert(outExponent < (1u << outExponentBits));

	            const Integral packedMantissa = inMantissaBits > outMantissaBits
	                ? inMantissa >> (inMantissaBits - outMantissaBits)
	                : inMantissa << (outMantissaBits - inMantissaBits);
	            const Integral packedExponent = outExponent << outMantissaBits;
	            const Integral packedSign = inSign << outExponentBits << outMantissaBits;

	            const auto outFloat = static_cast<Integral>(packedMantissa | packedExponent | packedSign);
	            return outFloat;
	        }

	        // TODO(bgruber): Boost.Hana generalizes these sorts of computations on mixed constants and values
	        template<typename E, typename M>
	        LLAMA_FN_HOST_ACC_INLINE auto integBits(E e, M m)
	        {
	            return llama::internal::BoxedValue{e.value() + m.value() + 1};
	        }

	        template<auto E, auto M>
	        LLAMA_FN_HOST_ACC_INLINE auto integBits(
	            llama::internal::BoxedValue<Constant<E>>,
	            llama::internal::BoxedValue<Constant<M>>)
	        {
	            return llama::internal::BoxedValue<Constant<E + M + 1>>{};
	        }

	        /// A proxy type representing a reference to a reduced precision floating-point value, stored in a buffer at a
	        /// specified bit offset.
	        /// @tparam Float Floating-point data type which can be loaded and store through this reference.
	        /// @tparam StoredIntegralCV Integral type used for storing the bits with CV qualifiers.
	        /// @tparam SizeType Type used to store sizes and offsets.
	        template<typename Float, typename StoredIntegralCV, typename VHExp, typename VHMan, typename SizeType>
	        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
	        struct LLAMA_DECLSPEC_EMPTY_BASES BitPackedFloatRef
	            : private VHExp
	            , private VHMan
	            , ProxyRefOpMixin<BitPackedFloatRef<Float, StoredIntegralCV, VHExp, VHMan, SizeType>, Float>
	        {
	        private:
	            static_assert(
	                std::is_same_v<Float, float> || std::is_same_v<Float, double>,
	                "Types other than float or double are not implemented yet");
	            static_assert(
	                std::numeric_limits<Float>::is_iec559,
	                "Only IEEE754/IEC559 floating point formats are implemented");

	            using FloatBits = std::conditional_t<std::is_same_v<Float, float>, std::uint32_t, std::uint64_t>;

	            BitPackedIntRef<
	                FloatBits,
	                StoredIntegralCV,
	                decltype(integBits(std::declval<VHExp>(), std::declval<VHMan>())),
	                SizeType,
	                SignBit::Discard>
	                intref;

	        public:
	            using value_type = Float;

	            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedFloatRef(
	                StoredIntegralCV* p,
	                SizeType bitOffset,
	                VHExp vhExp,
	                VHMan vhMan)
	                : VHExp{vhExp}
	                , VHMan{vhMan}
	                , intref{
	                      p,
	                      bitOffset,
	                      integBits(vhExp, vhMan),
	                  }
	            {
	            }

	            BitPackedFloatRef(const BitPackedFloatRef&) = default;

	            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
	            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const BitPackedFloatRef& other) -> BitPackedFloatRef&
	            {
	                *this = static_cast<value_type>(other);
	                return *this;
	            }

	            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	            LLAMA_FN_HOST_ACC_INLINE constexpr operator Float() const
	            {
	                using Bits = FloatBitTraits<Float>;
	                const FloatBits packedFloat = intref;
	                const FloatBits unpackedFloat
	                    = repackFloat(packedFloat, VHMan::value(), VHExp::value(), Bits::mantissa, Bits::exponent);
	                Float f;
	                std::memcpy(&f, &unpackedFloat, sizeof(Float));
	                return f;
	            }

	            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Float f) -> BitPackedFloatRef&
	            {
	                using Bits = FloatBitTraits<Float>;
	                FloatBits unpackedFloat = 0;
	                std::memcpy(&unpackedFloat, &f, sizeof(Float));
	                const FloatBits packedFloat
	                    = repackFloat(unpackedFloat, Bits::mantissa, Bits::exponent, VHMan::value(), VHExp::value());
	                intref = packedFloat;
	                return *this;
	            }
	        };

	        template<typename RecordDim>
	        using StoredIntegralFor
	            = std::conditional_t<mp_contains<FlatRecordDim<RecordDim>, double>::value, std::uint64_t, std::uint32_t>;
	    } // namespace internal

	    // TODO(bgruber): I would like to allow zero mantissa bits, which would then no longer support INF. Likewise,
	    // support to skip the sign bit would also be great.
	    /// Struct of array mapping using bit packing to reduce size/precision of floating-point data types. The bit layout
	    /// is [1 sign bit, exponentBits bits from the exponent, mantissaBits bits from the mantissa]+ and tries to follow
	    /// IEEE 754. Infinity and NAN are supported. If the packed exponent bits are not big enough to hold a number, it
	    /// will be set to infinity (preserving the sign). If your record dimension contains non-floating-point types,
	    /// split them off using the \ref Split mapping first.
	    /// \tparam ExponentBits If ExponentBits is llama::Constant<N>, the compile-time N specifies the number of bits to
	    /// use to store the exponent. If ExponentBits is llama::Value<T>, the number of bits is specified at runtime,
	    /// passed to the constructor and stored as type T. Must not be zero.
	    /// \tparam MantissaBits Like ExponentBits but for the mantissa bits. Must not be zero (otherwise values turn INF).
	    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    /// \tparam TStoredIntegral Integral type used as storage of reduced precision floating-point values.
	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        typename ExponentBits = typename TArrayExtents::value_type,
	        typename MantissaBits = ExponentBits,
	        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        typename TStoredIntegral = internal::StoredIntegralFor<TRecordDim>>
	    struct LLAMA_DECLSPEC_EMPTY_BASES BitPackedFloatSoA
	        : MappingBase<TArrayExtents, TRecordDim>
	        , llama::internal::BoxedValue<ExponentBits, 0>
	        , llama::internal::BoxedValue<MantissaBits, 1>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using VHExp = llama::internal::BoxedValue<ExponentBits, 0>;
	        using VHMan = llama::internal::BoxedValue<MantissaBits, 1>;
	        using size_type = typename TArrayExtents::value_type;

	    public:
	        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
	        using StoredIntegral = TStoredIntegral;
	        static constexpr std::size_t blobCount = mp_size<FlatRecordDim<TRecordDim>>::value;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto exponentBits() const -> size_type
	        {
	            return static_cast<size_type>(VHExp::value());
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto mantissaBits() const -> size_type
	        {
	            return static_cast<size_type>(VHMan::value());
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr explicit BitPackedFloatSoA(
	            TArrayExtents extents = {},
	            ExponentBits exponentBits = {},
	            MantissaBits mantissaBits = {},
	            TRecordDim = {})
	            : Base(extents)
	            , VHExp{exponentBits}
	            , VHMan{mantissaBits}
	        {
	            assert(this->exponentBits() > 0);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
	        {
	            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(StoredIntegral) * CHAR_BIT);
	            const auto bitsNeeded
	                = LinearizeArrayIndexFunctor{}.size(Base::extents()) * (exponentBits() + mantissaBits() + 1);
	            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
	        {
	            return true;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
	            typename Base::ArrayIndex ai,
	            RecordCoord<RecordCoords...>,
	            Blobs& blobs) const
	        {
	            constexpr auto blob = llama::flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
	            const auto bitOffset
	                = LinearizeArrayIndexFunctor{}(ai, Base::extents()) * (exponentBits() + mantissaBits() + 1);

	            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
	            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
	            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	            return internal::BitPackedFloatRef<DstType, QualifiedStoredIntegral, VHExp, VHMan, size_type>{
	                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
	                bitOffset,
	                static_cast<const VHExp&>(*this),
	                static_cast<const VHMan&>(*this)};
	            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	        }
	    };

	    /// Binds parameters to a \ref BitPackedFloatSoA mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<
	        typename ExponentBits = unsigned,
	        typename MantissaBits = ExponentBits,
	        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        typename StoredIntegral = void>
	    struct BindBitPackedFloatSoA
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = BitPackedFloatSoA<
	            ArrayExtents,
	            RecordDim,
	            ExponentBits,
	            MantissaBits,
	            LinearizeArrayIndexFunctor,
	            std::conditional_t<
	                !std::is_void_v<StoredIntegral>,
	                StoredIntegral,
	                internal::StoredIntegralFor<RecordDim>>>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isBitPackedFloatSoA = false;

	    LLAMA_EXPORT
	    template<typename... Ts>
	    inline constexpr bool isBitPackedFloatSoA<BitPackedFloatSoA<Ts...>> = true;

	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        typename ExponentBits = typename TArrayExtents::value_type,
	        typename MantissaBits = ExponentBits,
	        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        template<typename> typename PermuteFields = PermuteFieldsInOrder,
	        typename TStoredIntegral = internal::StoredIntegralFor<TRecordDim>>
	    struct LLAMA_DECLSPEC_EMPTY_BASES BitPackedFloatAoS
	        : MappingBase<TArrayExtents, TRecordDim>
	        , llama::internal::BoxedValue<ExponentBits, 0>
	        , llama::internal::BoxedValue<MantissaBits, 1>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using VHExp = llama::internal::BoxedValue<ExponentBits, 0>;
	        using VHMan = llama::internal::BoxedValue<MantissaBits, 1>;
	        using size_type = typename TArrayExtents::value_type;

	    public:
	        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
	        using StoredIntegral = TStoredIntegral;

	        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
	        static constexpr std::size_t blobCount = 1;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto exponentBits() const -> size_type
	        {
	            return static_cast<size_type>(VHExp::value());
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto mantissaBits() const -> size_type
	        {
	            return static_cast<size_type>(VHMan::value());
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr explicit BitPackedFloatAoS(
	            TArrayExtents extents = {},
	            ExponentBits exponentBits = {},
	            MantissaBits mantissaBits = {},
	            TRecordDim = {})
	            : Base(extents)
	            , VHExp{exponentBits}
	            , VHMan{mantissaBits}
	        {
	            assert(this->exponentBits() > 0);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
	        {
	            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(StoredIntegral) * CHAR_BIT);
	            const auto bitsNeeded = TLinearizeArrayIndexFunctor{}.size(Base::extents())
	                * static_cast<size_type>(exponentBits() + mantissaBits() + 1)
	                * static_cast<size_type>(flatFieldCount<TRecordDim>);
	            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
	        {
	            return true;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
	            typename Base::ArrayIndex ai,
	            RecordCoord<RecordCoords...>,
	            Blobs& blobs) const
	        {
	            constexpr auto flatFieldIndex = static_cast<size_type>(
	                Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>);
	            const auto bitOffset = ((TLinearizeArrayIndexFunctor{}(ai, Base::extents())
	                                     * static_cast<size_type>(flatFieldCount<TRecordDim>))
	                                    + flatFieldIndex)
	                * static_cast<size_type>(exponentBits() + mantissaBits() + 1);

	            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
	            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
	            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	            return internal::BitPackedFloatRef<DstType, QualifiedStoredIntegral, VHExp, VHMan, size_type>{
	                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[0][0]),
	                bitOffset,
	                static_cast<const VHExp&>(*this),
	                static_cast<const VHMan&>(*this)};
	            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	        }
	    };

	    LLAMA_EXPORT
	    template<
	        typename ExponentBits = unsigned,
	        typename MantissaBits = ExponentBits,
	        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
	        template<typename> typename PermuteFields = PermuteFieldsInOrder,
	        typename StoredIntegral = void>
	    struct BindBitPackedFloatAoS
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = BitPackedFloatAoS<
	            ArrayExtents,
	            RecordDim,
	            ExponentBits,
	            MantissaBits,
	            LinearizeArrayIndexFunctor,
	            PermuteFields,
	            std::conditional_t<
	                !std::is_void_v<StoredIntegral>,
	                StoredIntegral,
	                internal::StoredIntegralFor<RecordDim>>>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isBitPackedFloatAoS = false;

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        typename ExponentBits,
	        typename MantissaBits,
	        typename LinearizeArrayIndexFunctor,
	        template<typename>
	        typename PermuteFields,
	        typename StoredIntegral>
	    inline constexpr bool isBitPackedFloatAoS<BitPackedFloatAoS<
	        ArrayExtents,
	        RecordDim,
	        ExponentBits,
	        MantissaBits,
	        LinearizeArrayIndexFunctor,
	        PermuteFields,
	        StoredIntegral>>
	        = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/BitPackedFloat.hpp ==
	// ============================================================================

// #include "mapping/BitPackedInt.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/mapping/Bytesplit.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
	// #include "Common.hpp"    // amalgamate: file already inlined

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template<typename T>
	        using ReplaceByByteArray = std::byte[sizeof(T)];

	        template<typename RecordDim>
	        using SplitBytes = TransformLeaves<RecordDim, ReplaceByByteArray>;
	    } // namespace internal

	    /// Meta mapping splitting each field in the record dimension into an array of bytes and mapping the resulting
	    /// record dimension using a further mapping.
	    LLAMA_EXPORT
	    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
	    struct Bytesplit : private InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>
	    {
	        using Inner = InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>;

	        using ArrayExtents = typename Inner::ArrayExtents;
	        using RecordDim = TRecordDim; // hide Inner::RecordDim
	        using Inner::blobCount;

	        using Inner::blobSize;
	        using Inner::extents;

	    private:
	        using ArrayIndex = typename TArrayExtents::Index;

	    public:
	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr explicit Bytesplit(TArrayExtents extents, TRecordDim = {}) : Inner(extents)
	        {
	        }

	        template<typename... Args>
	        LLAMA_FN_HOST_ACC_INLINE constexpr explicit Bytesplit(std::tuple<Args...> innerMappingArgs, TRecordDim = {})
	            : Inner(std::make_from_tuple<Inner>(innerMappingArgs))
	        {
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
	        {
	            return true;
	        }

	        template<typename RC, typename BlobArray>
	        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
	        struct Reference : ProxyRefOpMixin<Reference<RC, BlobArray>, GetType<TRecordDim, RC>>
	        {
	        private:
	            const Inner& inner;
	            ArrayIndex ai;
	            BlobArray& blobs;

	        public:
	            using value_type = GetType<TRecordDim, RC>;

	            LLAMA_FN_HOST_ACC_INLINE constexpr Reference(const Inner& innerMapping, ArrayIndex ai, BlobArray& blobs)
	                : inner(innerMapping)
	                , ai(ai)
	                , blobs(blobs)
	            {
	            }

	            Reference(const Reference&) = default;

	            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
	            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const Reference& other) -> Reference&
	            {
	                *this = static_cast<value_type>(other);
	                return *this;
	            }

	            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	            LLAMA_FN_HOST_ACC_INLINE constexpr operator value_type() const
	            {
	#ifdef _MSC_VER
	                // MSVC workaround. Without this, MSVC deduces the last template parameter of mapToMemory wrongly
	                BlobArray& blobs = this->blobs;
	#endif

	                value_type v;
	                auto* p = reinterpret_cast<std::byte*>(&v);
	                mp_for_each<mp_iota_c<sizeof(value_type)>>(
	                    [&](auto ic) LLAMA_LAMBDA_INLINE
	                    {
	                        constexpr auto i = decltype(ic)::value;
	                        auto&& ref = mapToMemory(inner, ai, Cat<RC, RecordCoord<i>>{}, blobs);

	                        p[i] = ref;
	                    });
	                return v;
	            }

	            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(value_type v) -> Reference&
	            {
	#ifdef _MSC_VER
	                // MSVC workaround. Without this, MSVC deduces the last template parameter of mapToMemory wrongly
	                BlobArray& blobs = this->blobs;
	#endif

	                auto* p = reinterpret_cast<std::byte*>(&v);
	                mp_for_each<mp_iota_c<sizeof(value_type)>>(
	                    [&](auto ic) LLAMA_LAMBDA_INLINE
	                    {
	                        constexpr auto i = decltype(ic)::value;

	                        auto&& ref = mapToMemory(inner, ai, Cat<RC, RecordCoord<i>>{}, blobs);
	                        ref = p[i];
	                    });
	                return *this;
	            }
	        };

	        template<std::size_t... RecordCoords, typename BlobArray>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(ArrayIndex ai, RecordCoord<RecordCoords...>, BlobArray& blobs)
	            const
	        {
	            return Reference<RecordCoord<RecordCoords...>, BlobArray>{*this, ai, blobs};
	        }
	    };

	    /// Binds parameters to a \ref Bytesplit mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<template<typename, typename> typename InnerMapping>
	    struct BindBytesplit
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = Bytesplit<ArrayExtents, RecordDim, InnerMapping>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isBytesplit = false;

	    LLAMA_EXPORT
	    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
	    inline constexpr bool isBytesplit<Bytesplit<TArrayExtents, TRecordDim, InnerMapping>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/Bytesplit.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/Byteswap.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../Core.hpp"    // amalgamate: file already inlined
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
	// #include "Common.hpp"    // amalgamate: file already inlined
		// ============================================================================
		// == ./include/llama/mapping/Projection.hpp ==
		// ==
		// Copyright 2022 Bernhard Manfred Gruber
		// SPDX-License-Identifier: MPL-2.0

		// #pragma once
		// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
		// #include "../View.hpp"    // amalgamate: file already inlined
		// #include "Common.hpp"    // amalgamate: file already inlined

		namespace llama::mapping
		{
		    namespace internal
		    {
		        template<typename F>
		        struct UnaryFunctionTraits
		        {
		            static_assert(sizeof(F) == 0, "F is not an unary function");
		        };

		        template<typename Arg, typename Ret>
		        struct UnaryFunctionTraits<Ret (*)(Arg)>
		        {
		            using ArgumentType = Arg;
		            using ReturnType = Ret;
		        };

		        template<typename ProjectionMap, typename Coord, typename RecordDimType>
		        auto projectionOrVoidImpl()
		        {
		            if constexpr(mp_map_contains<ProjectionMap, Coord>::value)
		                return mp_identity<mp_second<mp_map_find<ProjectionMap, Coord>>>{};
		            else if constexpr(mp_map_contains<ProjectionMap, RecordDimType>::value)
		                return mp_identity<mp_second<mp_map_find<ProjectionMap, RecordDimType>>>{};
		            else
		                return mp_identity<void>{};
		        }

		        template<typename ProjectionMap, typename Coord, typename RecordDimType>
		        using ProjectionOrVoid = typename decltype(projectionOrVoidImpl<ProjectionMap, Coord, RecordDimType>())::type;

		        template<typename ProjectionMap>
		        struct MakeReplacerProj
		        {
		            template<typename Coord, typename RecordDimType>
		            static auto replacedTypeProj()
		            {
		                using Projection = ProjectionOrVoid<ProjectionMap, Coord, RecordDimType>;
		                if constexpr(std::is_void_v<Projection>)
		                    return mp_identity<RecordDimType>{};
		                else
		                {
		                    using LoadFunc = UnaryFunctionTraits<decltype(&Projection::load)>;
		                    using StoreFunc = UnaryFunctionTraits<decltype(&Projection::store)>;

		                    static_assert(std::is_same_v<typename LoadFunc::ReturnType, RecordDimType>);
		                    static_assert(std::is_same_v<typename StoreFunc::ArgumentType, RecordDimType>);
		                    static_assert(std::is_same_v<typename LoadFunc::ArgumentType, typename StoreFunc::ReturnType>);

		                    return mp_identity<typename StoreFunc::ReturnType>{};
		                }
		            }

		            template<typename Coord, typename RecordDimType>
		            using fn = typename decltype(replacedTypeProj<Coord, RecordDimType>())::type;
		        };

		        template<typename RecordDim, typename ProjectionMap>
		        using ReplaceTypesByProjectionResults
		            = TransformLeavesWithCoord<RecordDim, MakeReplacerProj<ProjectionMap>::template fn>;

		        template<typename Reference, typename Projection>
		        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
		        struct ProjectionReference
		            : ProxyRefOpMixin<
		                  ProjectionReference<Reference, Projection>,
		                  decltype(Projection::load(std::declval<Reference>()))>
		        {
		        private:
		            Reference storageRef;

		        public:
		            using value_type = decltype(Projection::load(std::declval<Reference>()));

		            LLAMA_FN_HOST_ACC_INLINE constexpr explicit ProjectionReference(Reference storageRef)
		                : storageRef{storageRef}
		            {
		            }

		            ProjectionReference(const ProjectionReference&) = default;

		            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
		            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const ProjectionReference& other) -> ProjectionReference&
		            {
		                *this = static_cast<value_type>(other);
		                return *this;
		            }

		            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
		            LLAMA_FN_HOST_ACC_INLINE constexpr operator value_type() const
		            {
		                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
		                return Projection::load(storageRef);
		                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
		            }

		            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(value_type v) -> ProjectionReference&
		            {
		                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
		                storageRef = Projection::store(v);
		                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
		                return *this;
		            }
		        };
		    } // namespace internal

		    /// Mapping that projects types in the record domain to different types. Projections are executed during load and
		    /// store.
		    /// @tparam TProjectionMap A type list of binary type lists (a map) specifing a projection (map value) for a type
		    /// or the type at a \ref RecordCoord (map key). A projection is a type with two functions:
		    /// struct Proj {
		    ///   static auto load(auto&& fromMem);
		    ///   static auto store(auto&& toMem);
		    /// };
		    LLAMA_EXPORT
		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        template<typename, typename>
		        typename InnerMapping,
		        typename TProjectionMap>
		    struct Projection
		        : private InnerMapping<TArrayExtents, internal::ReplaceTypesByProjectionResults<TRecordDim, TProjectionMap>>
		    {
		        using Inner
		            = InnerMapping<TArrayExtents, internal::ReplaceTypesByProjectionResults<TRecordDim, TProjectionMap>>;
		        using ProjectionMap = TProjectionMap;
		        using ArrayExtents = typename Inner::ArrayExtents;
		        using RecordDim = TRecordDim; // hide Inner::RecordDim
		        using Inner::blobCount;
		        using Inner::blobSize;
		        using Inner::extents;
		        using Inner::Inner;

		    protected:
		        using ArrayIndex = typename ArrayExtents::Index;

		    public:
		        template<typename RecordCoord>
		        LLAMA_FN_HOST_ACC_INLINE static constexpr auto isComputed(RecordCoord) -> bool
		        {
		            return !std::is_void_v<
		                internal::ProjectionOrVoid<ProjectionMap, RecordCoord, GetType<RecordDim, RecordCoord>>>;
		        }

		        template<std::size_t... RecordCoords, typename BlobArray>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
		            ArrayIndex ai,
		            RecordCoord<RecordCoords...> rc,
		            BlobArray& blobs) const
		        {
		            static_assert(isComputed(rc));
		            using RecordDimType = GetType<RecordDim, RecordCoord<RecordCoords...>>;
		            using Reference = decltype(mapToMemory(static_cast<const Inner&>(*this), ai, rc, blobs));
		            using Projection = internal::ProjectionOrVoid<ProjectionMap, RecordCoord<RecordCoords...>, RecordDimType>;
		            static_assert(!std::is_void_v<Projection>);
		            Reference r = mapToMemory(static_cast<const Inner&>(*this), ai, rc, blobs);

		            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
		            return internal::ProjectionReference<Reference, Projection>{r};
		            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
		        }

		        template<std::size_t... RecordCoords>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> rc = {})
		            const -> NrAndOffset<typename ArrayExtents::value_type>
		        {
		            static_assert(!isComputed(rc));
		            return Inner::blobNrAndOffset(ai, rc);
		        }
		    };

		    /// Binds parameters to a \ref Projection mapping except for array and record dimension, producing a quoted
		    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
		    LLAMA_EXPORT
		    template<template<typename, typename> typename InnerMapping, typename ProjectionMap>
		    struct BindProjection
		    {
		        template<typename ArrayExtents, typename RecordDim>
		        using fn = Projection<ArrayExtents, RecordDim, InnerMapping, ProjectionMap>;
		    };

		    LLAMA_EXPORT
		    template<typename Mapping>
		    inline constexpr bool isProjection = false;

		    LLAMA_EXPORT
		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        template<typename, typename>
		        typename InnerMapping,
		        typename ReplacementMap>
		    inline constexpr bool isProjection<Projection<TArrayExtents, TRecordDim, InnerMapping, ReplacementMap>> = true;
		} // namespace llama::mapping
		// ==
		// == ./include/llama/mapping/Projection.hpp ==
		// ============================================================================


	namespace llama::mapping
	{
	    namespace internal
	    {
	        // TODO(bgruber): replace by std::byteswap in C++23
	        template<typename T>
	        LLAMA_FN_HOST_ACC_INLINE auto byteswap(T t) -> T
	        {
	            if constexpr(sizeof(T) == 1)
	                return t;
	            else
	            {
	                llama::Array<std::byte, sizeof(T)> arr{};
	                std::memcpy(&arr, &t, sizeof(T));

	                for(std::size_t i = 0; i < sizeof(T) / 2; i++)
	                {
	                    const auto a = arr[i];
	                    const auto b = arr[sizeof(T) - 1 - i];
	                    arr[i] = b;
	                    arr[sizeof(T) - 1 - i] = a;
	                }

	                std::memcpy(&t, &arr, sizeof(T));
	                return t;
	            }
	        }

	        template<typename T>
	        struct ByteswapProjection
	        {
	            LLAMA_FN_HOST_ACC_INLINE static auto load(T v) -> T
	            {
	                return byteswap(v);
	            }

	            LLAMA_FN_HOST_ACC_INLINE static auto store(T v) -> T
	            {
	                return byteswap(v);
	            }
	        };

	        template<typename T>
	        using MakeByteswapProjectionPair = mp_list<T, ByteswapProjection<T>>;

	        template<typename RecordDim>
	        using MakeByteswapProjectionMap
	            = mp_transform<MakeByteswapProjectionPair, mp_unique<FlatRecordDim<RecordDim>>>;
	    } // namespace internal

	    /// Mapping that swaps the byte order of all values when loading/storing.
	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim, template<typename, typename> typename InnerMapping>
	    struct Byteswap : Projection<ArrayExtents, RecordDim, InnerMapping, internal::MakeByteswapProjectionMap<RecordDim>>
	    {
	    private:
	        using Base = Projection<ArrayExtents, RecordDim, InnerMapping, internal::MakeByteswapProjectionMap<RecordDim>>;

	    public:
	        using Base::Base;
	    };

	    /// Binds parameters to a \ref ChangeType mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<template<typename, typename> typename InnerMapping>
	    struct BindByteswap
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = Byteswap<ArrayExtents, RecordDim, InnerMapping>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isByteswap = false;

	    LLAMA_EXPORT
	    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
	    inline constexpr bool isByteswap<Byteswap<TArrayExtents, TRecordDim, InnerMapping>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/Byteswap.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/ChangeType.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already inlined
	// #include "Common.hpp"    // amalgamate: file already inlined
	// #include "Projection.hpp"    // amalgamate: file already inlined

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template<typename UserT, typename StoredT>
	        struct ChangeTypeProjection
	        {
	            LLAMA_FN_HOST_ACC_INLINE static auto load(StoredT v) -> UserT
	            {
	                return static_cast<UserT>(v); // we could allow stronger casts here
	            }

	            LLAMA_FN_HOST_ACC_INLINE static auto store(UserT v) -> StoredT
	            {
	                return static_cast<StoredT>(v); // we could allow stronger casts here
	            }
	        };

	        template<typename RecordDim>
	        struct MakeProjectionPair
	        {
	            template<typename Key>
	            static auto recordDimType()
	            {
	                if constexpr(isRecordCoord<Key>)
	                    return mp_identity<GetType<RecordDim, Key>>{};
	                else
	                    return mp_identity<Key>{};
	            }

	            template<typename Pair, typename Key = mp_first<Pair>, typename StoredT = mp_second<Pair>>
	            using fn = mp_list<Key, ChangeTypeProjection<typename decltype(recordDimType<Key>())::type, StoredT>>;
	        };

	        template<typename RecordDim, typename ReplacementMap>
	        using MakeProjectionMap = mp_transform<MakeProjectionPair<RecordDim>::template fn, ReplacementMap>;
	    } // namespace internal

	    /// Mapping that changes the type in the record domain for a different one in storage. Conversions happen during
	    /// load and store.
	    /// @tparam ReplacementMap A type list of binary type lists (a map) specifiying which type or the type at a \ref
	    /// RecordCoord (map key) to replace by which other type (mapped value).
	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        template<typename, typename>
	        typename InnerMapping,
	        typename ReplacementMap>
	    struct ChangeType
	        : Projection<ArrayExtents, RecordDim, InnerMapping, internal::MakeProjectionMap<RecordDim, ReplacementMap>>
	    {
	    private:
	        using Base = Projection<
	            ArrayExtents,
	            RecordDim,
	            InnerMapping,
	            internal::MakeProjectionMap<RecordDim, ReplacementMap>>;

	    public:
	        using Base::Base;
	    };

	    /// Binds parameters to a \ref ChangeType mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<template<typename, typename> typename InnerMapping, typename ReplacementMap>
	    struct BindChangeType
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = ChangeType<ArrayExtents, RecordDim, InnerMapping, ReplacementMap>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isChangeType = false;

	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        template<typename, typename>
	        typename InnerMapping,
	        typename ReplacementMap>
	    inline constexpr bool isChangeType<ChangeType<TArrayExtents, TRecordDim, InnerMapping, ReplacementMap>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/ChangeType.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/FieldAccessCount.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../StructName.hpp"    // amalgamate: file already inlined
	// #include "Common.hpp"    // amalgamate: file already inlined

	#include <cstdio>
	#include <iomanip>
	#include <iostream>

	namespace llama::mapping
	{
	    LLAMA_EXPORT
	    template<typename CountType>
	    struct AccessCounts
	    {
	        union
	        {
	            CountType memLocsComputed;
	            CountType reads;
	        };
	        CountType writes;
	    };

	    namespace internal
	    {
	        template<typename Value, typename Ref, typename Count>
	        struct FieldAccessCountReference : ProxyRefOpMixin<FieldAccessCountReference<Value, Ref, Count>, Value>
	        {
	            using value_type = Value;

	            template<typename RefFwd>
	            LLAMA_FN_HOST_ACC_INLINE constexpr FieldAccessCountReference(RefFwd&& r, AccessCounts<Count>* hits)
	                : r(std::forward<RefFwd>(r))
	                , hits(hits)
	            {
	                static_assert(std::is_same_v<std::remove_reference_t<Ref>, std::remove_reference_t<RefFwd>>);
	            }

	            FieldAccessCountReference(const FieldAccessCountReference&) = default;
	            FieldAccessCountReference(FieldAccessCountReference&&) noexcept = default;
	            auto operator=(FieldAccessCountReference&& ref) noexcept -> FieldAccessCountReference& = default;
	            ~FieldAccessCountReference() = default;

	            LLAMA_FN_HOST_ACC_INLINE auto operator=(const FieldAccessCountReference& ref) -> FieldAccessCountReference&
	            {
	                if(&ref != this)
	                {
	                    internal::atomicInc(hits->writes);
	                    r = static_cast<value_type>(ref);
	                }
	                return *this;
	            }

	            LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type value) -> FieldAccessCountReference&
	            {
	                internal::atomicInc(hits->writes);
	                r = value;
	                return *this;
	            }

	            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	            LLAMA_FN_HOST_ACC_INLINE operator value_type() const
	            {
	                internal::atomicInc(hits->reads);
	                return static_cast<value_type>(r);
	            }

	        private:
	            Ref r;
	            AccessCounts<Count>* hits;
	        };
	    } // namespace internal

	    /// Forwards all calls to the inner mapping. Counts all accesses made through this mapping and allows printing a
	    /// summary.
	    /// @tparam Mapping The type of the inner mapping.
	    /// @tparam TCountType The type used for counting the number of accesses.
	    /// @tparam MyCodeHandlesProxyReferences If false, FieldAccessCount will avoid proxy references but can then only
	    /// count the number of address computations
	    LLAMA_EXPORT
	    template<typename Mapping, typename TCountType = std::size_t, bool MyCodeHandlesProxyReferences = true>
	    struct FieldAccessCount : Mapping
	    {
	    private:
	        using size_type = typename Mapping::ArrayExtents::value_type;

	    public:
	        using RecordDim = typename Mapping::RecordDim;
	        using CountType = TCountType;
	        inline static constexpr bool myCodeHandlesProxyReferences = MyCodeHandlesProxyReferences;

	        struct FieldHitsArray : Array<AccessCounts<CountType>, flatFieldCount<RecordDim>>
	        {
	            LLAMA_FN_HOST_ACC_INLINE auto total() const -> AccessCounts<CountType>
	            {
	                AccessCounts<CountType> total{};
	                for(const auto& ac : *this)
	                {
	                    if constexpr(MyCodeHandlesProxyReferences)
	                    {
	                        total.reads += ac.reads;
	                        total.writes += ac.writes;
	                    }
	                    else
	                        total.memLocsComputed += ac.memLocsComputed;
	                }
	                return total;
	            }

	            struct TotalBytes
	            {
	                CountType totalRead;
	                CountType totalWritten;
	            };

	            /// When MyCodeHandlesProxyReferences is true, return a pair of the total read and written bytes. If false,
	            /// returns the total bytes of accessed data as a single value.
	            LLAMA_FN_HOST_ACC_INLINE auto totalBytes() const
	            {
	                CountType r = 0;
	                CountType w = 0; // NOLINT(misc-const-correctness)
	                forEachLeafCoord<RecordDim>(
	                    [&](auto rc) LLAMA_LAMBDA_INLINE
	                    {
	                        const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
	                        const auto fieldSize = sizeof(GetType<RecordDim, decltype(rc)>);
	                        if constexpr(MyCodeHandlesProxyReferences)
	                        {
	                            r += (*this)[i].reads * fieldSize;
	                            w += (*this)[i].writes * fieldSize;
	                        }
	                        else
	                            r += (*this)[i].memLocsComputed * fieldSize;
	                    });
	                if constexpr(MyCodeHandlesProxyReferences)
	                    return TotalBytes{r, w};
	                else
	                    return r;
	            }
	        };

	        inline static constexpr auto blobCount = Mapping::blobCount + 1;

	        constexpr FieldAccessCount() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        explicit FieldAccessCount(Mapping mapping) : Mapping(std::move(mapping))
	        {
	        }

	        template<typename... Args>
	        LLAMA_FN_HOST_ACC_INLINE explicit FieldAccessCount(Args&&... innerArgs)
	            : Mapping(std::forward<Args>(innerArgs)...)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize(size_type blobIndex) const -> size_type
	        {
	            if(blobIndex < size_type{Mapping::blobCount})
	                return inner().blobSize(blobIndex);
	            return sizeof(FieldHitsArray);
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
	        {
	            return true;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto compute(
	            typename Mapping::ArrayExtents::Index ai,
	            RecordCoord<RecordCoords...> rc,
	            Blobs& blobs) const -> decltype(auto)
	        {
	            static_assert(
	                !std::is_const_v<Blobs>,
	                "Cannot access (even just reading) data through FieldAccessCount from const blobs/view, since we need "
	                "to write "
	                "the access counts");

	            auto& hits = fieldHits(blobs)[+flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>];
	            decltype(auto) ref = mapToMemory(inner(), ai, rc, blobs); // T& or proxy reference (value)
	            if constexpr(MyCodeHandlesProxyReferences)
	            {
	                using Value = GetType<RecordDim, decltype(rc)>;
	                using Ref = decltype(ref);
	                return internal::FieldAccessCountReference<Value, Ref, CountType>{std::forward<Ref>(ref), &hits};
	            }
	            else
	            {
	                internal::atomicInc(hits.memLocsComputed);
	                return ref;
	            }
	        }

	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
	        template<typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto fieldHits(const Blobs& blobs) const -> const FieldHitsArray&
	        {
	            return reinterpret_cast<const FieldHitsArray&>(*&blobs[blobCount - 1][0]);
	        }

	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
	        template<typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto fieldHits(Blobs& blobs) const -> FieldHitsArray&
	        {
	            return reinterpret_cast<FieldHitsArray&>(*&blobs[blobCount - 1][0]);
	        }

	        template<typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE void printFieldHits(const Blobs& blobs) const
	        {
	            printFieldHits(fieldHits(blobs));
	        }

	        LLAMA_FN_HOST_ACC_INLINE void printFieldHits(const FieldHitsArray& hits) const
	        {
	#ifdef __CUDA_ARCH__
	            printFieldHitsDevice(hits);
	#else
	            printFieldHitsHost(hits);
	#endif
	        }

	    private:
	        static constexpr auto columnWidth = 10;
	        static constexpr auto sizeColumnWidth = 5;

	        void printFieldHitsHost(const FieldHitsArray& hits) const
	        {
	            if constexpr(MyCodeHandlesProxyReferences)
	                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
	                          << std::setw(sizeColumnWidth) << "Size" << std::right << std::setw(columnWidth) << "Reads"
	                          << ' ' << std::right << std::setw(columnWidth) << "Writes" << '\n';
	            else
	                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
	                          << std::setw(sizeColumnWidth) << "Size" << std::right << std::setw(columnWidth)
	                          << "Mlocs cmp" << '\n';
	            forEachLeafCoord<RecordDim>(
	                [&](auto rc)
	                {
	                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
	                    const auto fieldSize = sizeof(GetType<RecordDim, decltype(rc)>);
	                    if constexpr(MyCodeHandlesProxyReferences)
	                        std::cout << std::left << std::setw(columnWidth) << prettyRecordCoord<RecordDim>(rc) << ' '
	                                  << std::right << std::setw(sizeColumnWidth) << fieldSize << std::right
	                                  << std::setw(columnWidth) << hits[i].reads << ' ' << std::right
	                                  << std::setw(columnWidth) << hits[i].writes << '\n';
	                    else
	                        std::cout << std::left << std::setw(columnWidth) << prettyRecordCoord<RecordDim>(rc) << ' '
	                                  << std::right << std::setw(sizeColumnWidth) << fieldSize << std::right
	                                  << std::setw(columnWidth) << hits[i].memLocsComputed << '\n';
	                });
	            const auto total = hits.totalBytes();
	            if constexpr(MyCodeHandlesProxyReferences)
	            {
	                const auto [rsize, runit] = prettySize(total.totalRead);
	                const auto [wsize, wunit] = prettySize(total.totalWritten);
	                std::cout << std::left << std::setw(columnWidth) << "Total" << ' ' << std::right
	                          << std::setw(sizeColumnWidth) << ' ' << std::right << std::setw(columnWidth) << rsize
	                          << runit << ' ' << std::right << std::setw(columnWidth - 2) << wsize << wunit << '\n';
	            }
	            else
	            {
	                const auto [size, unit] = prettySize(total);
	                std::cout << std::left << std::setw(columnWidth) << "Total" << ' ' << std::right
	                          << std::setw(sizeColumnWidth) << ' ' << std::right << std::setw(columnWidth) << size << unit
	                          << '\n';
	            }
	            std::cout << std::internal;
	        }

	        LLAMA_ACC void printFieldHitsDevice(const FieldHitsArray& hits) const
	        {
	            if constexpr(MyCodeHandlesProxyReferences)
	            {
	                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                printf(
	                    "%*s %*s %*s %*s\n",
	                    columnWidth,
	                    "Field",
	                    sizeColumnWidth,
	                    "Size",
	                    columnWidth,
	                    "Reads",
	                    columnWidth,
	                    "Writes");
	            }
	            else
	            {
	                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                printf("%*s %*s %*s\n", columnWidth, "Field", sizeColumnWidth, "Size", columnWidth, "Mlocs cmp");
	            }
	            forEachLeafCoord<RecordDim>(
	                [&](auto rc)
	                {
	                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
	                    const auto fieldSize = sizeof(GetType<RecordDim, decltype(rc)>);
	                    constexpr auto fieldName = prettyRecordCoord<RecordDim>(rc);
	                    char fieldNameZT[fieldName.size() + 1]{}; // nvcc does not handle the %*.*s parameter correctly
	                    llama::internal::constexprCopy(fieldName.begin(), fieldName.end(), fieldNameZT);
	                    if constexpr(MyCodeHandlesProxyReferences)
	                    {
	                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                        printf(
	                            "%*.s %*lu %*lu %*lu\n",
	                            columnWidth,
	                            fieldNameZT,
	                            sizeColumnWidth,
	                            fieldSize,
	                            columnWidth,
	                            static_cast<unsigned long>(hits[i].reads),
	                            columnWidth,
	                            static_cast<unsigned long>(hits[i].writes));
	                    }
	                    else
	                    {
	                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                        printf(
	                            "%*.s %*lu %*lu\n",
	                            columnWidth,
	                            fieldNameZT,
	                            sizeColumnWidth,
	                            fieldSize,
	                            columnWidth,
	                            static_cast<unsigned long>(hits[i].memLocsComputed));
	                    }
	                });

	            const auto total = hits.totalBytes();
	            if constexpr(MyCodeHandlesProxyReferences)
	            {
	                const auto [rsize, runit] = prettySize(total.totalRead);
	                const auto [wsize, wunit] = prettySize(total.totalWritten);
	                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                printf(
	                    "%*s %*s %*f%s %*f%s\n",
	                    columnWidth,
	                    "Total",
	                    sizeColumnWidth,
	                    "",
	                    columnWidth,
	                    rsize,
	                    runit,
	                    columnWidth - 2,
	                    wsize,
	                    wunit);
	            }
	            else
	            {
	                const auto [size, unit] = prettySize(total);
	                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                printf("%*s %*s %*f%s\n", columnWidth, "Total", sizeColumnWidth, "", columnWidth, size, unit);
	            }
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto inner() const -> const Mapping&
	        {
	            return static_cast<const Mapping&>(*this);
	        }
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isFieldAccessCount = false;

	    LLAMA_EXPORT
	    template<typename Mapping, typename CountType, bool MyCodeHandlesProxyReferences>
	    inline constexpr bool isFieldAccessCount<FieldAccessCount<Mapping, CountType, MyCodeHandlesProxyReferences>>
	        = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/FieldAccessCount.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/Heatmap.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../View.hpp"    // amalgamate: file already inlined
	// #include "Common.hpp"    // amalgamate: file already inlined

	// #include <array>    // amalgamate: file already included
	// #include <atomic>    // amalgamate: file already included
	#include <sstream>
	// #include <vector>    // amalgamate: file already included
	#if __has_include(<span>)
	#    include <span>
	#endif

	namespace llama::mapping
	{
	    /// Forwards all calls to the inner mapping. Counts all accesses made to blocks inside the blobs, allowing to
	    /// extract a heatmap.
	    /// @tparam Mapping The type of the inner mapping.
	    /// @tparam Granularity The granularity in bytes on which to could accesses. A value of 1 counts every byte.
	    /// individually. A value of e.g. 64, counts accesses per 64 byte block.
	    /// @tparam TCountType Data type used to count the number of accesses. Atomic increments must be supported for this
	    /// type.
	    LLAMA_EXPORT
	    template<
	        typename Mapping,
	        typename Mapping::ArrayExtents::value_type Granularity = 1,
	        typename TCountType = std::size_t>
	    struct Heatmap : private Mapping
	    {
	        static_assert(!hasAnyComputedField<Mapping>, "Heatmaps for computed mappings are not implemented.");

	    private:
	        using size_type = typename Mapping::ArrayExtents::value_type;
	        using ArrayIndex = typename Mapping::ArrayExtents::Index;

	    public:
	        using Inner = Mapping;
	        inline static constexpr std::size_t granularity = Granularity;
	        using CountType = TCountType;
	        using ArrayExtents = typename Mapping::ArrayExtents;
	        using RecordDim = typename Mapping::RecordDim;

	        // We duplicate every blob of the inner mapping with a shadow blob, where we count the accesses
	        inline static constexpr std::size_t blobCount = Mapping::blobCount * 2;

	        constexpr Heatmap() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        explicit Heatmap(Mapping mapping) : Mapping(std::move(mapping))
	        {
	        }

	        template<typename... Args>
	        LLAMA_FN_HOST_ACC_INLINE explicit Heatmap(Args&&... innerArgs) : Mapping(std::forward<Args>(innerArgs)...)
	        {
	        }

	#if defined(__cpp_lib_concepts) && defined(__NVCOMPILER)
	        // nvc++ fails to find extents() from the base class when trying to satisfy the Mapping concept
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> typename Mapping::ArrayExtents
	        {
	            return static_cast<const Mapping&>(*this).extents();
	        }
	#else
	        using Mapping::extents;
	#endif

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize(size_type blobIndex) const -> size_type
	        {
	            if(blobIndex < size_type{Mapping::blobCount})
	                return Mapping::blobSize(blobIndex);
	            return blockHitsSize(blobIndex - size_type{Mapping::blobCount}) * sizeof(CountType);
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
	        {
	            return true;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto compute(ArrayIndex ai, RecordCoord<RecordCoords...> rc, Blobs& blobs) const
	            -> decltype(auto)
	        {
	            static_assert(
	                !std::is_const_v<Blobs>,
	                "Cannot access (even just reading) data through Heatmap from const blobs/view, since we need to write "
	                "the access counts");

	            const auto [nr, offset] = Mapping::blobNrAndOffset(ai, rc);
	            using Type = GetType<typename Mapping::RecordDim, RecordCoord<RecordCoords...>>;

	            auto* hits = blockHitsPtr(nr, blobs);
	            for(size_type i = 0; i < divCeil(size_type{sizeof(Type)}, Granularity); i++)
	                internal::atomicInc(hits[offset / Granularity + i]);

	            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	            return reinterpret_cast<CopyConst<std::remove_reference_t<decltype(blobs[nr][offset])>, Type>&>(
	                blobs[nr][offset]);
	            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	        }

	        // Returns the size of the block hits buffer for blob forBlobI in block counts.
	        LLAMA_FN_HOST_ACC_INLINE auto blockHitsSize(size_type forBlobI) const -> size_type
	        {
	            assert(forBlobI < Mapping::blobCount);
	            return divCeil(Mapping::blobSize(forBlobI), Granularity);
	        }

	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
	        template<typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto blockHitsPtr(size_type forBlobI, Blobs& blobs) const
	            -> CopyConst<Blobs, CountType>*
	        {
	            return reinterpret_cast<CopyConst<Blobs, CountType>*>(&blobs[size_type{Mapping::blobCount} + forBlobI][0]);
	        }

	#ifdef __cpp_lib_span
	        template<typename Blobs>
	        auto blockHits(size_type forBlobI, Blobs& blobs) const -> std::span<CopyConst<Blobs, CountType>>
	        {
	            return {blockHitsPtr(forBlobI, blobs), blockHitsSize(forBlobI)};
	        }
	#endif

	    private:
	        static auto trimBlobRight(const CountType* bh, std::size_t size)
	        {
	            while(size > 0 && bh[size - 1] == 0)
	                size--;
	            return size;
	        }

	    public:
	        /// Writes a data file suitable for gnuplot containing the heatmap data. You can use the script provided by
	        /// \ref gnuplotScript to plot this data file.
	        /// @param blobs The blobs of the view containing this mapping
	        /// @param os The stream to write the data to. Should be some form of std::ostream.
	        template<typename Blobs, typename OStream>
	        void writeGnuplotDataFileAscii(
	            const Blobs& blobs,
	            OStream&& os,
	            bool trimEnd = true,
	            std::size_t wrapAfterBlocks = 64) const
	        {
	            for(std::size_t i = 0; i < Mapping::blobCount; i++)
	            {
	                auto* bh = blockHitsPtr(i, blobs);
	                auto size = blockHitsSize(i);
	                if(trimEnd)
	                    size = trimBlobRight(bh, size);
	                for(size_type j = 0; j < size; j++)
	                {
	                    if(j > 0)
	                        os << (j % wrapAfterBlocks == 0 ? '\n' : ' ');
	                    os << bh[j];
	                }
	                for(size_type j = size; j < roundUpToMultiple(size, wrapAfterBlocks); j++)
	                    os << " 0";
	                os << '\n';
	            }
	        }

	        template<typename Blobs, typename OStream>
	        void writeGnuplotDataFileBinary(
	            const Blobs& blobs,
	            OStream&& os,
	            bool trimEnd = true,
	            std::size_t afterBlobRoundUpTo = 64) const
	        {
	            for(std::size_t i = 0; i < Mapping::blobCount; i++)
	            {
	                auto* bh = blockHitsPtr(i, blobs);
	                auto size = blockHitsSize(i);
	                if(trimEnd)
	                    size = trimBlobRight(bh, size);
	                os.write(reinterpret_cast<const char*>(bh), size * sizeof(CountType));

	                // round up before starting next blob
	                CountType zero = 0;
	                for(size_type j = size; j < roundUpToMultiple(size, afterBlobRoundUpTo); j++)
	                    os.write(reinterpret_cast<const char*>(&zero), sizeof(CountType));
	            }
	        }

	        /// An example script for plotting the ASCII heatmap data using gnuplot.
	        static constexpr std::string_view gnuplotScriptAscii = R"(#!/bin/bash
	gnuplot -p <<EOF
	file = '${1:-plot.bin}'

	set xtics format ""
	set x2tics autofreq 32
	set yrange [] reverse
	set link x2; set link y2
	set x2label "Byte"
	plot file matrix with image pixels axes x2y1
	EOF
	)";

	        /// An example script for plotting the binary heatmap data using gnuplot.
	        static constexpr std::string_view gnuplotScriptBinary = R"(#!/bin/bash
	gnuplot -p <<EOF
	file = '${1:-plot.bin}'
	rowlength = '${2:-64}'
	maxrows = '${3:-all}'
	format = '${4:-%uint64}'

	counts = system('stat -c "%s" ${1:-plot.bin}')/8
	rows = counts/rowlength
	rows = maxrows eq 'all' ? rows : (rows < maxrows ? rows : maxrows)

	set xtics format ""
	set x2tics autofreq 32
	set yrange [] reverse
	set link x2; set link y2
	set x2label "Byte"
	plot file binary array=(rowlength,rows) format=format with image pixels axes x2y1
	EOF
	)";
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isHeatmap = false;

	    LLAMA_EXPORT
	    template<typename Mapping, typename Mapping::ArrayExtents::value_type Granularity, typename CountType>
	    inline constexpr bool isHeatmap<Heatmap<Mapping, Granularity, CountType>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/Heatmap.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/Null.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already inlined

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template<typename T>
	        struct NullReference : ProxyRefOpMixin<NullReference<T>, T>
	        {
	            using value_type = T;

	            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	            LLAMA_FN_HOST_ACC_INLINE constexpr operator T() const
	            {
	                return T{}; // this might not be the best design decision
	            }

	            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(T) -> NullReference&
	            {
	                return *this;
	            }
	        };
	    } // namespace internal

	    /// The Null mappings maps all elements to nothing. Writing data through a reference obtained from the Null mapping
	    /// discards the value. Reading through such a reference returns a default constructed object.
	    LLAMA_EXPORT
	    template<typename TArrayExtents, typename TRecordDim>
	    struct Null : MappingBase<TArrayExtents, TRecordDim>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using size_type = typename TArrayExtents::value_type;

	    public:
	        static constexpr std::size_t blobCount = 0;

	        using Base::Base;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
	        {
	            return 0;
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
	        {
	            return true;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
	            typename Base::ArrayIndex,
	            RecordCoord<RecordCoords...>,
	            Blobs&) const
	        {
	            using FieldType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
	            return internal::NullReference<FieldType>{};
	        }
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isNull = false;

	    LLAMA_EXPORT
	    template<typename ArrayExtents, typename RecordDim>
	    inline constexpr bool isNull<Null<ArrayExtents, RecordDim>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/Null.hpp ==
	// ============================================================================

// #include "mapping/One.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/mapping/PermuteArrayIndex.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already inlined

	namespace llama::mapping
	{
	    /// Meta mapping permuting the array indices before forwarding to another mapping. The array extents are not
	    /// changed.
	    /// @tparam Permutation The pack of integrals describing the permutation of the array indices. The inner mapping
	    /// will be called with an ArrayIndex{ai[Permutation]...}.
	    LLAMA_EXPORT
	    template<typename Mapping, std::size_t... Permutation>
	    struct PermuteArrayIndex : Mapping
	    {
	    private:
	        using size_type = typename Mapping::ArrayExtents::value_type;
	        using ArrayIndex = typename Mapping::ArrayExtents::Index;

	    public:
	        using Inner = Mapping;

	        constexpr PermuteArrayIndex() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        explicit PermuteArrayIndex(Mapping mapping) : Mapping(std::move(mapping))
	        {
	        }

	        template<typename... Args>
	        LLAMA_FN_HOST_ACC_INLINE explicit PermuteArrayIndex(Args&&... innerArgs)
	            : Mapping(std::forward<Args>(innerArgs)...)
	        {
	        }

	        static_assert(
	            sizeof...(Permutation) == ArrayIndex::rank,
	            "The number of integral arguments to PermuteArrayIndex must be the same as ArrayExtents::rank");

	        template<std::size_t... RCs>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RCs...> rc = {}) const
	            -> NrAndOffset<size_type>
	        {
	            return Inner::blobNrAndOffset(ArrayIndex{ai[Permutation]...}, rc);
	        }

	        template<std::size_t... RCs, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto compute(ArrayIndex ai, RecordCoord<RCs...> rc, Blobs& blobs) const
	            -> decltype(auto)
	        {
	            return Inner::compute(ArrayIndex{ai[Permutation]...}, rc, blobs);
	        }
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    PermuteArrayIndex(Mapping) -> PermuteArrayIndex<Mapping>;

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isPermuteArrayIndex = false;

	    LLAMA_EXPORT
	    template<typename Mapping, std::size_t... Permutation>
	    inline constexpr bool isPermuteArrayIndex<PermuteArrayIndex<Mapping, Permutation...>> = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/PermuteArrayIndex.hpp ==
	// ============================================================================

// #include "mapping/Projection.hpp"    // amalgamate: file already inlined
// #include "mapping/SoA.hpp"    // amalgamate: file already inlined
	// ============================================================================
	// == ./include/llama/mapping/Split.hpp ==
	// ==
	// Copyright 2022 Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../View.hpp"    // amalgamate: file already inlined
	// #include "Common.hpp"    // amalgamate: file already inlined

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template<typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
	        auto partitionRecordDim(Record<Fields...>, RecordCoord<FirstCoord, Coords...>)
	        {
	            using Rec = Record<Fields...>;
	            if constexpr(sizeof...(Coords) == 0)
	            {
	                using Part1 = Record<mp_at_c<Rec, FirstCoord>>;
	                using Part2 = mp_erase_c<Rec, FirstCoord, FirstCoord + 1>;
	                return mp_list<Part1, Part2>{};
	            }
	            else
	            {
	                using FieldTag = GetTag<Rec, RecordCoord<FirstCoord>>;
	                using FieldType = GetType<Rec, RecordCoord<FirstCoord>>;
	                using InnerPartition = decltype(partitionRecordDim(FieldType{}, RecordCoord<Coords...>{}));
	                using Part1 = Record<Field<FieldTag, mp_first<InnerPartition>>>;
	                using Part2 = mp_replace_at_c<Rec, FirstCoord, Field<FieldTag, mp_second<InnerPartition>>>;
	                return mp_list<Part1, Part2>{};
	            }
	        }

	        template<typename Acc, typename TagList>
	        struct PartitionFoldOpImpl
	        {
	            using Part1Before = mp_first<Acc>;
	            using Part2Before = mp_second<Acc>;
	            using R = decltype(partitionRecordDim(Part2Before{}, GetCoordFromTags<Part2Before, TagList>{}));
	            using Part1After = mp_first<R>;
	            using Part2After = mp_second<R>;

	            using type = mp_list<MergedRecordDims<Part1Before, Part1After>, Part2After>;
	        };

	        template<typename Acc, typename TagList>
	        using PartitionFoldOp = typename PartitionFoldOpImpl<Acc, TagList>::type;

	        template<typename... Fields, typename... RCs>
	        auto partitionRecordDim(Record<Fields...>, mp_list<RCs...>)
	        {
	            static_assert((isRecordCoord<RCs> && ...));
	            using Initial = mp_list<Record<>, Record<Fields...>>; // initially, nothing selected for mapping 1
	            return mp_fold<mp_list<GetTags<Record<Fields...>, RCs>...>, Initial, PartitionFoldOp>{};
	        }

	        template<typename RC, typename RecordCoordForMapping1>
	        inline constexpr bool isSelected = recordCoordCommonPrefixIsSame<RecordCoordForMapping1, RC>;

	        template<typename RC, typename... RecordCoordsForMapping1>
	        inline constexpr bool isSelected<RC, mp_list<RecordCoordsForMapping1...>>
	            = (isSelected<RC, RecordCoordsForMapping1> || ...);

	        template<typename RecordDim, typename Selector>
	        struct ReplaceTagListsByCoords;

	        template<typename RecordDim, std::size_t... RCs>
	        struct ReplaceTagListsByCoords<RecordDim, RecordCoord<RCs...>>
	        {
	            using type = RecordCoord<RCs...>;
	        };

	        template<typename RecordDim, typename... Args>
	        struct ReplaceTagListsByCoords<RecordDim, mp_list<Args...>>
	        {
	            static auto f()
	            {
	                if constexpr(((mp_is_list<Args>::value || isRecordCoord<Args>) &&...))
	                    // Args is a pack of tag lists/record coords
	                    return mp_list<GetCoordFromTags<RecordDim, Args>...>{};
	                else
	                    // Args is a single tag lists
	                    return GetCoordFromTags<RecordDim, Args...>{};
	            }

	            using type = decltype(f());
	        };
	    } // namespace internal

	    /// Mapping which splits off a part of the record dimension and maps it differently then the rest.
	    /// \tparam TSelectorForMapping1 Selects a part of the record dimension to be mapped by MappingTemplate1. Can be a
	    /// \ref RecordCoord, a type list of RecordCoords, a type list of tags (selecting one field), or a type list of
	    /// type list of tags (selecting one field per sub list). dimension to be mapped differently.
	    /// \tparam MappingTemplate1 The mapping used for the selected part of the record dimension.
	    /// \tparam MappingTemplate2 The mapping used for the not selected part of the record dimension.
	    /// \tparam SeparateBlobs If true, both pieces of the record dimension are mapped to separate blobs.
	    LLAMA_EXPORT
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        typename TSelectorForMapping1,
	        template<typename...>
	        typename MappingTemplate1,
	        template<typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs = false>
	    struct Split
	    {
	        static_assert(isRecord<TRecordDim>, "Cannot split a scalar record dim");

	        using ArrayExtents = TArrayExtents;
	        using ArrayIndex = typename ArrayExtents::Index;
	        using RecordDim = TRecordDim;

	        using SelectorForMapping1 = TSelectorForMapping1;
	        using NormalizedSelectorForMapping1 =
	            typename internal::ReplaceTagListsByCoords<RecordDim, SelectorForMapping1>::type;
	        using RecordDimPartitions
	            = decltype(internal::partitionRecordDim(RecordDim{}, NormalizedSelectorForMapping1{}));
	        using RecordDim1 = mp_first<RecordDimPartitions>;
	        using RecordDim2 = mp_second<RecordDimPartitions>;

	        using Mapping1 = MappingTemplate1<ArrayExtents, RecordDim1>;
	        using Mapping2 = MappingTemplate2<ArrayExtents, RecordDim2>;

	        static constexpr std::size_t blobCount = SeparateBlobs ? Mapping1::blobCount + Mapping2::blobCount : 1;
	        static_assert(SeparateBlobs || Mapping1::blobCount == 1);
	        static_assert(SeparateBlobs || Mapping2::blobCount == 1);

	    private:
	        using size_type = typename ArrayExtents::value_type;
	        static constexpr size_type m1bc = static_cast<size_type>(Mapping1::blobCount);

	    public:
	        constexpr Split() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr explicit Split(ArrayExtents extents) : mapping1(extents), mapping2(extents)
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr Split(Mapping1 mapping1, Mapping2 mapping2)
	            : mapping1(std::move(mapping1))
	            , mapping2(std::move(mapping2))
	        {
	        }

	        template<typename... Args1, typename... Args2>
	        LLAMA_FN_HOST_ACC_INLINE constexpr Split(std::tuple<Args1...> mappingArgs1, std::tuple<Args2...> mappingArgs2)
	            : mapping1(std::make_from_tuple<Mapping1>(mappingArgs1))
	            , mapping2(std::make_from_tuple<Mapping2>(mappingArgs2))
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
	        {
	            return mapping1.extents();
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize([[maybe_unused]] size_type i) const -> size_type
	        {
	            if constexpr(SeparateBlobs)
	            {
	                if(i < m1bc)
	                    return mapping1.blobSize(i);
	                return mapping2.blobSize(i - m1bc);
	            }
	            else
	                return mapping1.blobSize(0) + mapping2.blobSize(0);
	        }

	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> = {}) const
	            -> NrAndOffset<size_type>
	        {
	            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;

	            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, NormalizedSelectorForMapping1>)
	                return mapping1.blobNrAndOffset(ai, GetCoordFromTags<RecordDim1, Tags>{});
	            else
	            {
	                auto nrAndOffset = mapping2.blobNrAndOffset(ai, GetCoordFromTags<RecordDim2, Tags>{});
	                if constexpr(SeparateBlobs)
	                    nrAndOffset.nr += m1bc;
	                else
	                {
	                    for(size_type i = 0; i < m1bc; i++)
	                        nrAndOffset.offset += mapping1.blobSize(i);
	                }
	                return nrAndOffset;
	            }
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>) -> bool
	        {
	            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;
	            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, NormalizedSelectorForMapping1>)
	                return llama::isComputed<Mapping1, GetCoordFromTags<RecordDim1, Tags>>;
	            else
	                return llama::isComputed<Mapping2, GetCoordFromTags<RecordDim2, Tags>>;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(ArrayIndex ai, RecordCoord<RecordCoords...>, Blobs& blobs)
	            const
	        {
	            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;
	            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, NormalizedSelectorForMapping1>)
	                return mapping1.compute(ai, GetCoordFromTags<RecordDim1, Tags>{}, blobs);
	            else
	            {
	                // only pass on blobs for mapping 2, so it can index starting from 0
	                auto* blobs2 = &blobs[0] + m1bc;
	                return mapping2.compute(ai, GetCoordFromTags<RecordDim2, Tags>{}, blobs2);
	            }
	        }

	        Mapping1 mapping1;
	        Mapping2 mapping2;
	    };

	    /// Binds parameters to a \ref Split mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    LLAMA_EXPORT
	    template<
	        typename SelectorForMapping1,
	        template<typename...>
	        typename MappingTemplate1,
	        template<typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs = false>
	    struct BindSplit
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn
	            = Split<ArrayExtents, RecordDim, SelectorForMapping1, MappingTemplate1, MappingTemplate2, SeparateBlobs>;
	    };

	    LLAMA_EXPORT
	    template<typename Mapping>
	    inline constexpr bool isSplit = false;

	    LLAMA_EXPORT
	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        typename SelectorForMapping1,
	        template<typename...>
	        typename MappingTemplate1,
	        template<typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs>
	    inline constexpr bool
	        isSplit<Split<ArrayExtents, RecordDim, SelectorForMapping1, MappingTemplate1, MappingTemplate2, SeparateBlobs>>
	        = true;
	} // namespace llama::mapping
	// ==
	// == ./include/llama/mapping/Split.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./include/llama/mapping/tree/Mapping.hpp ==
	// ==
	// Copyright 2020 Alexander Matthes, Bernhard Manfred Gruber
	// SPDX-License-Identifier: MPL-2.0

	// #pragma once
	// #include "../Common.hpp"    // amalgamate: file already inlined
		// ============================================================================
		// == ./include/llama/mapping/tree/Functors.hpp ==
		// ==
		// Copyright 2020 Alexander Matthes, Bernhard Manfred Gruber
		// SPDX-License-Identifier: MPL-2.0

		// #pragma once
			// ============================================================================
			// == ./include/llama/mapping/tree/TreeFromDimensions.hpp ==
			// ==
			// Copyright 2020 Alexander Matthes, Bernhard Manfred Gruber
			// SPDX-License-Identifier: MPL-2.0
			// #pragma once
			// #include "../../Core.hpp"    // amalgamate: file already inlined
			// #include "../../Tuple.hpp"    // amalgamate: file already inlined

			// #include <cstddef>    // amalgamate: file already included
			// #include <string>    // amalgamate: file already included
			// #include <type_traits>    // amalgamate: file already included

			namespace llama::mapping::tree
			{
			    template<typename T>
			    inline constexpr auto one = 1;

			    template<>
			    inline constexpr auto one<mp_size_t<1>> = mp_size_t<1>{};

			    template<typename TIdentifier, typename TType, typename CountType = std::size_t>
			    struct Leaf
			    {
			        using Identifier = TIdentifier;
			        using Type = TType;

			        const CountType count = one<CountType>;
			    };

			    template<typename TIdentifier, typename TChildrenTuple, typename CountType = std::size_t>
			    struct Node
			    {
			        using Identifier = TIdentifier;
			        using ChildrenTuple = TChildrenTuple;

			        const CountType count = one<CountType>;
			        const ChildrenTuple childs = {};
			    };

			    template<std::size_t ChildIndex = 0, typename ArrayIndexType = std::size_t>
			    struct TreeCoordElement
			    {
			        static constexpr mp_size_t<ChildIndex> childIndex = {};
			        const ArrayIndexType arrayIndex = {};
			    };

			    template<std::size_t... Coords>
			    using TreeCoord = Tuple<TreeCoordElement<Coords, mp_size_t<0>>...>;

			    namespace internal
			    {
			        template<typename... Coords, std::size_t... Is>
			        auto treeCoordToString(Tuple<Coords...> treeCoord, std::index_sequence<Is...>) -> std::string
			        {
			            auto s
			                = ((std::to_string(get<Is>(treeCoord).arrayIndex) + ":" + std::to_string(get<Is>(treeCoord).childIndex)
			                    + ", ")
			                   + ...);
			            s.resize(s.length() - 2);
			            return s;
			        }
			    } // namespace internal

			    template<typename TreeCoord>
			    auto treeCoordToString(TreeCoord treeCoord) -> std::string
			    {
			        return std::string("[ ")
			            + internal::treeCoordToString(treeCoord, std::make_index_sequence<std::tuple_size_v<TreeCoord>>{})
			            + std::string(" ]");
			    }

			    namespace internal
			    {
			        template<typename Tag, typename RecordDim, typename CountType>
			        struct CreateTreeElement
			        {
			            using type = Leaf<Tag, RecordDim, mp_size_t<1>>;
			        };

			        template<typename Tag, typename... Fields, typename CountType>
			        struct CreateTreeElement<Tag, Record<Fields...>, CountType>
			        {
			            using type = Node<
			                Tag,
			                Tuple<typename CreateTreeElement<GetFieldTag<Fields>, GetFieldType<Fields>, mp_size_t<1>>::type...>,
			                CountType>;
			        };

			        template<typename Tag, typename ChildType, std::size_t Count, typename CountType>
			        struct CreateTreeElement<Tag, ChildType[Count], CountType>
			        {
			            template<std::size_t... Is>
			            static auto createChildren(std::index_sequence<Is...>)
			            {
			                return Tuple<typename CreateTreeElement<RecordCoord<Is>, ChildType, mp_size_t<1>>::type...>{};
			            }

			            using type = Node<Tag, decltype(createChildren(std::make_index_sequence<Count>{})), CountType>;
			        };

			        template<typename Leaf, std::size_t Count>
			        struct WrapInNNodes
			        {
			            using type = Node<NoName, Tuple<typename WrapInNNodes<Leaf, Count - 1>::type>>;
			        };

			        template<typename Leaf>
			        struct WrapInNNodes<Leaf, 0>
			        {
			            using type = Leaf;
			        };

			        template<typename RecordDim>
			        using TreeFromRecordDimImpl = typename CreateTreeElement<NoName, RecordDim, std::size_t>::type;
			    } // namespace internal

			    template<typename RecordDim>
			    using TreeFromRecordDim = internal::TreeFromRecordDimImpl<RecordDim>;

			    template<typename ArrayExtents, typename RecordDim>
			    using TreeFromDimensions =
			        typename internal::WrapInNNodes<internal::TreeFromRecordDimImpl<RecordDim>, ArrayExtents::rank - 1>::type;

			    template<typename RecordDim, typename V, std::size_t N, std::size_t Pos = 0>
			    LLAMA_FN_HOST_ACC_INLINE auto createTree(const ArrayIndex<V, N>& size)
			    {
			        if constexpr(Pos == N - 1)
			            return TreeFromRecordDim<RecordDim>{
			                static_cast<std::size_t>(size[N - 1])}; // FIXME(bgruber): propagate index type
			        else
			        {
			            Tuple inner{createTree<RecordDim, V, N, Pos + 1>(size)}; // NOLINT(misc-const-correctness)
			            return Node<NoName, decltype(inner)>{
			                static_cast<std::size_t>(size[Pos]),
			                inner}; // FIXME(bgruber): propagate index type
			        }
			    };

			    namespace internal
			    {
			        template<
			            typename ArrayIndex,
			            std::size_t... ADIndices,
			            std::size_t FirstRecordCoord,
			            std::size_t... RecordCoords>
			        LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(
			            const ArrayIndex& ai,
			            std::index_sequence<ADIndices...>,
			            RecordCoord<FirstRecordCoord, RecordCoords...>)
			        {
			            return Tuple{
			                TreeCoordElement<(ADIndices == ArrayIndex::rank - 1 ? FirstRecordCoord : 0)>{static_cast<std::size_t>(
			                    ai[ADIndices])}..., // TODO(bgruber): we should keep the integer type from the array index
			                TreeCoordElement<RecordCoords, mp_size_t<0>>{}...,
			                TreeCoordElement<0, mp_size_t<0>>{}};
			        }
			    } // namespace internal

			    template<typename RecordCoord, typename ArrayIndex>
			    LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(const ArrayIndex& ai)
			    {
			        return internal::createTreeCoord(ai, std::make_index_sequence<ArrayIndex::rank>{}, RecordCoord{});
			    }
			} // namespace llama::mapping::tree
			// ==
			// == ./include/llama/mapping/tree/TreeFromDimensions.hpp ==
			// ============================================================================


		namespace llama::mapping::tree::functor
		{
		    /// Functor for \ref tree::Mapping. Does nothing with the mapping tree. Is used for testing.
		    struct Idem
		    {
		        template<typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree& tree) const -> Tree
		        {
		            return tree;
		        }

		        template<typename Tree, typename TreeCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree&) const
		            -> TreeCoord
		        {
		            return basicCoord;
		        }

		        template<typename Tree, typename TreeCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree&) const
		            -> TreeCoord
		        {
		            return resultCoord;
		        }
		    };

		    /// Functor for \ref tree::Mapping. Moves all run time parts to the leaves, creating a SoA layout.
		    struct LeafOnlyRT
		    {
		        template<typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(Tree tree) const
		        {
		            return basicToResultImpl(tree, 1);
		        }

		        template<typename Tree, typename BasicCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const BasicCoord& basicCoord, const Tree& tree) const
		        {
		            return basicCoordToResultCoordImpl(basicCoord, tree);
		        }

		        template<typename Tree, typename ResultCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const ResultCoord& resultCoord, const Tree& /*tree*/)
		            const -> ResultCoord
		        {
		            return resultCoord;
		        }

		    private:
		        template<typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
		            const Node<Identifier, Type, CountType>& node,
		            std::size_t arraySize)
		        {
		            auto children = tupleTransform(
		                node.childs,
		                [&](auto element) { return basicToResultImpl(element, LLAMA_COPY(node.count) * arraySize); });
		            return Node<Identifier, decltype(children), mp_size_t<1>>{{}, children};
		        }

		        template<typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
		            const Leaf<Identifier, Type, CountType>& leaf,
		            std::size_t arraySize)
		        {
		            return Leaf<Identifier, Type, std::size_t>{LLAMA_COPY(leaf.count) * arraySize};
		        }

		        template<typename BasicCoord, typename NodeOrLeaf>
		        LLAMA_FN_HOST_ACC_INLINE static auto basicCoordToResultCoordImpl(
		            const BasicCoord& basicCoord,
		            const NodeOrLeaf& nodeOrLeaf,
		            std::size_t arraySize = 0)
		        {
		            if constexpr(std::tuple_size_v<BasicCoord> == 1)
		                return Tuple{TreeCoordElement<BasicCoord::FirstElement::childIndex>{
		                    arraySize + LLAMA_COPY(basicCoord.first().arrayIndex)}};
		            else
		            {
		                const auto& branch = get<BasicCoord::FirstElement::childIndex>(nodeOrLeaf.childs);
		                auto first = TreeCoordElement<BasicCoord::FirstElement::childIndex, mp_size_t<0>>{};

		                return tupleCat(
		                    Tuple{first},
		                    basicCoordToResultCoordImpl(
		                        basicCoord.rest(),
		                        branch,
		                        (arraySize + LLAMA_COPY(basicCoord.first().arrayIndex)) * LLAMA_COPY(branch.count)));
		            }
		        }
		    };

		    namespace internal
		    {
		        template<typename TreeCoord, typename Node>
		        LLAMA_FN_HOST_ACC_INLINE auto getNode(const Node& node)
		        {
		            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
		                return node;
		            else
		                return getNode<typename TreeCoord::RestTuple>(get<TreeCoord::FirstElement::childIndex>(node.childs));
		        }

		        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
		            const Node<Identifier, Type, CountType>& tree,
		            std::size_t newValue)
		        {
		            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
		                return Node<Identifier, Type>{newValue, tree.childs};
		            else
		            {
		                auto current = get<TreeCoord::FirstElement::childIndex>(tree.childs);
		                auto replacement = changeNodeRuntime<typename TreeCoord::RestTuple>(current, newValue);
		                auto children = tupleReplace<TreeCoord::FirstElement::childIndex>(tree.childs, replacement);
		                return Node<Identifier, decltype(children)>{tree.count, children};
		            }
		        }

		        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
		            const Leaf<Identifier, Type, CountType>& /*tree*/,
		            std::size_t newValue)
		        {
		            return Leaf<Identifier, Type, std::size_t>{newValue};
		        }

		        struct ChangeNodeChildsRuntimeFunctor
		        {
		            const std::size_t newValue;

		            template<typename Identifier, typename Type, typename CountType>
		            LLAMA_FN_HOST_ACC_INLINE auto operator()(const Node<Identifier, Type, CountType>& element) const
		            {
		                return Node<Identifier, Type, std::size_t>{element.count * newValue, element.childs};
		            }

		            template<typename Identifier, typename Type, typename CountType>
		            LLAMA_FN_HOST_ACC_INLINE auto operator()(const Leaf<Identifier, Type, CountType>& element) const
		            {
		                return Leaf<Identifier, Type, std::size_t>{element.count * newValue};
		            }
		        };

		        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
		            const Node<Identifier, Type, CountType>& tree,
		            std::size_t newValue)
		        {
		            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
		            {
		                auto children = tupleTransform(tree.childs, ChangeNodeChildsRuntimeFunctor{newValue});
		                return Node<Identifier, decltype(children)>{tree.count, children};
		            }
		            else
		            {
		                auto current = get<TreeCoord::FirstElement::childIndex>(tree.childs);
		                auto replacement = changeNodeChildsRuntime<typename TreeCoord::RestTuple>(current, newValue);
		                auto children = tupleReplace<TreeCoord::FirstElement::childIndex>(tree.childs, replacement);
		                return Node<Identifier, decltype(children)>{tree.count, children};
		            }
		        }

		        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
		        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
		            const Leaf<Identifier, Type, CountType>& tree,
		            std::size_t /*newValue*/)
		        {
		            return tree;
		        }
		    } // namespace internal

		    /// Functor for \ref tree::Mapping. Move the run time part of a node one level down in direction of the leaves by
		    /// the given amount (runtime or compile time value).
		    /// \tparam TreeCoord tree coordinate in the mapping tree which's run time part shall be moved down one level
		    /// \see tree::Mapping
		    template<typename TreeCoord, typename Amount = std::size_t>
		    struct MoveRTDown
		    {
		        const Amount amount = {};

		        template<typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree& tree) const
		        {
		            return internal::changeNodeChildsRuntime<TreeCoord>(
		                internal::changeNodeRuntime<TreeCoord>(
		                    tree,
		                    // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
		                    (internal::getNode<TreeCoord>(tree).count + amount - 1) / amount),
		                amount);
		        }

		        template<typename Tree, typename BasicCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const BasicCoord& basicCoord, const Tree& tree) const
		        {
		            return basicCoordToResultCoordImpl<TreeCoord>(basicCoord, tree);
		        }

		        template<typename Tree, typename ResultCoord>
		        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const ResultCoord& resultCoord, const Tree&) const
		            -> ResultCoord
		        {
		            return resultCoord;
		        }

		    private:
		        template<typename InternalTreeCoord, typename BasicCoord, typename Tree>
		        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoordImpl(const BasicCoord& basicCoord, const Tree& tree) const
		        {
		            if constexpr(std::is_same_v<InternalTreeCoord, Tuple<>>)
		            {
		                if constexpr(std::is_same_v<BasicCoord, Tuple<>>)
		                    return Tuple{};
		                else
		                {
		                    const auto& childTree = get<BasicCoord::FirstElement::childIndex>(tree.childs);
		                    const auto rt1 = basicCoord.first().arrayIndex / amount;
		                    const auto rt2 = basicCoord.first().arrayIndex % amount * childTree.count
		                        + basicCoord.rest().first().arrayIndex;
		                    auto rt1Child = TreeCoordElement<BasicCoord::FirstElement::childIndex>{rt1};
		                    auto rt2Child = TreeCoordElement<BasicCoord::RestTuple::FirstElement::childIndex>{rt2};
		                    return tupleCat(Tuple{rt1Child}, tupleCat(Tuple{rt2Child}, popFront(basicCoord.rest())));
		                }
		            }
		            else
		            {
		                if constexpr(InternalTreeCoord::FirstElement::childIndex != BasicCoord::FirstElement::childIndex)
		                    return basicCoord;
		                else
		                {
		                    auto rest = basicCoordToResultCoordImpl<typename InternalTreeCoord::RestTuple>(
		                        popFront(basicCoord),
		                        get<BasicCoord::FirstElement::childIndex>(tree.childs));
		                    return tupleCat(Tuple{basicCoord.first()}, rest);
		                }
		            }
		        }
		    };

		    template<typename TreeCoord, std::size_t Amount>
		    using MoveRTDownFixed = MoveRTDown<TreeCoord, mp_size_t<Amount>>;
		} // namespace llama::mapping::tree::functor
		// ==
		// == ./include/llama/mapping/tree/Functors.hpp ==
		// ============================================================================

	// #include "TreeFromDimensions.hpp"    // amalgamate: file already inlined
		// ============================================================================
		// == ./include/llama/mapping/tree/toString.hpp ==
		// ==
		// Copyright 2020 Alexander Matthes, Bernhard Manfred Gruber
		// SPDX-License-Identifier: MPL-2.0

		// #pragma once
		// #include "TreeFromDimensions.hpp"    // amalgamate: file already inlined

		// #include <string>    // amalgamate: file already included

		namespace llama::mapping::tree
		{
		    template<typename T>
		    auto toString(T) -> std::string
		    {
		        return "Unknown";
		    }

		    // handles array indices
		    template<std::size_t I>
		    inline auto toString(RecordCoord<I>) -> std::string
		    {
		        return "";
		    }

		    inline auto toString(NoName) -> std::string
		    {
		        return "";
		    }

		    template<typename... Elements>
		    auto toString(Tuple<Elements...> tree) -> std::string
		    {
		        if constexpr(sizeof...(Elements) > 1)
		            return toString(tree.first()) + " , " + toString(tree.rest());
		        else
		            return toString(tree.first());
		    }

		    namespace internal
		    {
		        inline void replaceAll(std::string& str, const std::string& search, const std::string& replace)
		        {
		            std::string::size_type i = 0;
		            while((i = str.find(search, i)) != std::string::npos)
		            {
		                str.replace(i, search.length(), replace);
		                i += replace.length();
		            }
		        }

		        template<typename NodeOrLeaf>
		        auto countAndIdentToString(const NodeOrLeaf& nodeOrLeaf) -> std::string
		        {
		            auto r = std::to_string(nodeOrLeaf.count);
		            if constexpr(std::is_same_v<std::decay_t<decltype(nodeOrLeaf.count)>, std::size_t>)
		                r += "R"; // runtime
		            else
		                r += "C"; // compile time
		            r += std::string{" * "} + toString(typename NodeOrLeaf::Identifier{});
		            return r;
		        }
		    } // namespace internal

		    template<typename Identifier, typename Type, typename CountType>
		    auto toString(const Node<Identifier, Type, CountType>& node) -> std::string
		    {
		        return internal::countAndIdentToString(node) + "[ " + toString(node.childs) + " ]";
		    }

		    template<typename Identifier, typename Type, typename CountType>
		    auto toString(const Leaf<Identifier, Type, CountType>& leaf) -> std::string
		    {
		        auto raw = std::string{llama::structName<Type>()};
		#ifdef _MSC_VER
		        internal::replaceAll(raw, " __cdecl(void)", "");
		#endif
		#ifdef __GNUG__
		        internal::replaceAll(raw, " ()", "");
		#endif
		        return internal::countAndIdentToString(leaf) + "(" + raw + ")";
		    }
		} // namespace llama::mapping::tree
		// ==
		// == ./include/llama/mapping/tree/toString.hpp ==
		// ============================================================================


	// #include <type_traits>    // amalgamate: file already included

	namespace llama::mapping::tree
	{
	    namespace internal
	    {
	        template<typename Tree, typename TreeOperationList>
	        struct MergeFunctors
	        {
	        };

	        template<typename Tree, typename... Operations>
	        struct MergeFunctors<Tree, Tuple<Operations...>>
	        {
	            mp_first<Tuple<Operations...>> operation = {};
	            using ResultTree = decltype(operation.basicToResult(Tree()));
	            ResultTree treeAfterOp;
	            MergeFunctors<ResultTree, mp_drop_c<Tuple<Operations...>, 1>> next = {};

	            MergeFunctors() = default;

	            LLAMA_FN_HOST_ACC_INLINE
	            MergeFunctors(const Tree& tree, const Tuple<Operations...>& treeOperationList)
	                : operation(treeOperationList.first())
	                , treeAfterOp(operation.basicToResult(tree))
	                , next(treeAfterOp, popFront(treeOperationList))
	            {
	            }

	            LLAMA_FN_HOST_ACC_INLINE
	            auto basicToResult(const Tree& tree) const
	            {
	                if constexpr(sizeof...(Operations) > 1)
	                    return next.basicToResult(treeAfterOp);
	                else if constexpr(sizeof...(Operations) == 1)
	                    return operation.basicToResult(tree);
	                else
	                    return tree;
	            }

	            template<typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree& tree) const
	            {
	                if constexpr(sizeof...(Operations) >= 1)
	                    return next.basicCoordToResultCoord(
	                        operation.basicCoordToResultCoord(basicCoord, tree),
	                        treeAfterOp);
	                else
	                    return basicCoord;
	            }

	            template<typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree& tree) const
	            {
	                if constexpr(sizeof...(Operations) >= 1)
	                    return next.resultCoordToBasicCoord(
	                        operation.resultCoordToBasicCoord(resultCoord, tree),
	                        operation.basicToResult(tree));
	                else
	                    return resultCoord;
	            }
	        };

	        template<typename Tree>
	        struct MergeFunctors<Tree, Tuple<>>
	        {
	            MergeFunctors() = default;

	            LLAMA_FN_HOST_ACC_INLINE
	            MergeFunctors(const Tree&, const Tuple<>&)
	            {
	            }

	            LLAMA_FN_HOST_ACC_INLINE
	            auto basicToResult(const Tree& tree) const
	            {
	                return tree;
	            }

	            template<typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree& /*tree*/)
	                const -> TreeCoord
	            {
	                return basicCoord;
	            }

	            template<typename TreeCoord>
	            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree& /*tree*/)
	                const -> TreeCoord
	            {
	                return resultCoord;
	            }
	        };

	        template<typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t;

	        template<typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t;

	        template<typename... Children, std::size_t... Is, typename Count>
	        LLAMA_FN_HOST_ACC_INLINE auto getChildrenBlobSize(
	            const Tuple<Children...>& childs,
	            std::index_sequence<Is...> /*ii*/,
	            const Count& count) -> std::size_t
	        {
	            return count * (getTreeBlobSize(get<Is>(childs)) + ...);
	        }

	        template<typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t
	        {
	            constexpr std::size_t childCount = mp_size<std::decay_t<decltype(node.childs)>>::value;
	            return getChildrenBlobSize(node.childs, std::make_index_sequence<childCount>{}, LLAMA_COPY(node.count));
	        }

	        template<typename Identifier, typename Type, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t
	        {
	            return leaf.count * sizeof(Type);
	        }

	        template<typename Childs, typename CountType>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Childs& childs, const CountType& count) -> std::size_t
	        {
	            return getTreeBlobSize(Node<NoName, Childs, CountType>{count, childs});
	        }

	        template<std::size_t MaxPos, typename Identifier, typename Type, typename CountType, std::size_t... Is>
	        LLAMA_FN_HOST_ACC_INLINE auto sumChildrenSmallerThan(
	            const Node<Identifier, Type, CountType>& node,
	            std::index_sequence<Is...>) -> std::size_t
	        {
	            return ((getTreeBlobSize(get<Is>(node.childs)) * (Is < MaxPos)) + ...);
	        }

	        template<typename Tree, typename... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobByte(const Tree& tree, const Tuple<Coords...>& treeCoord)
	            -> std::size_t
	        {
	            const auto firstArrayIndex = treeCoord.first().arrayIndex;
	            if constexpr(sizeof...(Coords) > 1)
	            {
	                constexpr auto firstChildIndex = decltype(treeCoord.first().childIndex)::value;
	                return getTreeBlobSize(tree.childs, firstArrayIndex)
	                    + sumChildrenSmallerThan<firstChildIndex>(
	                           tree,
	                           std::make_index_sequence<std::tuple_size_v<typename Tree::ChildrenTuple>>{})
	                    + getTreeBlobByte(get<firstChildIndex>(tree.childs), treeCoord.rest());
	            }
	            else
	                return sizeof(typename Tree::Type) * firstArrayIndex;
	        }
	    } // namespace internal

	    /// An experimental attempt to provide a general purpose description of a mapping. \ref Array and record
	    /// dimensions are represented by a compile time tree data structure. This tree is mapped into memory by means of a
	    /// breadth-first tree traversal. By specifying additional tree operations, the tree can be modified at compile
	    /// time before being mapped to memory.
	    template<typename TArrayExtents, typename TRecordDim, typename TreeOperationList>
	    struct Mapping : private TArrayExtents
	    {
	        using ArrayExtents = TArrayExtents;
	        using ArrayIndex = typename ArrayExtents::Index;
	        using RecordDim = TRecordDim;

	        // TODO(bgruber): , support more than one blob
	        static constexpr std::size_t blobCount = 1;

	    private:
	        using size_type = typename ArrayExtents::value_type;

	    public:
	        using BasicTree = TreeFromDimensions<ArrayExtents, RecordDim>;
	        using MergedFunctors = internal::MergeFunctors<BasicTree, TreeOperationList>;
	        BasicTree basicTree;
	        MergedFunctors mergedFunctors;

	        using ResultTree = decltype(mergedFunctors.basicToResult(basicTree));
	        ResultTree resultTree;

	        Mapping() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        Mapping(ArrayExtents extents, TreeOperationList treeOperationList, RecordDim = {})
	            : ArrayExtents(extents)
	            , basicTree(createTree<RecordDim>(extents.toArray()))
	            , mergedFunctors(basicTree, treeOperationList)
	            , resultTree(mergedFunctors.basicToResult(basicTree))
	        {
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto extents() const -> ArrayExtents
	        {
	            return static_cast<const ArrayExtents&>(*this);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto blobSize(size_type const) const -> size_type
	        {
	            // TODO(bgruber): propagate use of size_type
	            return internal::getTreeBlobSize(resultTree);
	        }

	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> = {}) const
	            -> NrAndOffset<size_type>
	        {
	            // TODO(bgruber): propagate use of size_type
	            const auto basicTreeCoord = createTreeCoord<RecordCoord<RecordCoords...>>(ai);
	            const auto resultTreeCoord = mergedFunctors.basicCoordToResultCoord(basicTreeCoord, basicTree);
	            const auto offset = static_cast<size_type>(internal::getTreeBlobByte(
	                resultTree,
	                resultTreeCoord)); // FIXME(bgruber): size_type should be propagated through getTreeBlobByte
	            return {0, offset};
	        }
	    };
	} // namespace llama::mapping::tree
	// ==
	// == ./include/llama/mapping/tree/Mapping.hpp ==
	// ============================================================================


#if defined(__NVCC__)
#    ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#        pragma nv_diag_default 940
#    else
#        pragma diag_default 940
#    endif
#endif
#ifdef __NVCOMPILER
#    pragma push
#    pragma diag_default 941
#endif
// ==
// == ./include/llama/llama.hpp ==
// ============================================================================

// ============================================================================
// == ./include/llama/Proofs.hpp ==
// ==
// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

// #pragma once
// #include "ArrayIndexRange.hpp"    // amalgamate: file already inlined
// #include "Core.hpp"    // amalgamate: file already inlined

namespace llama
{
// FIXME(bgruber): this test is actually not correct, because __cpp_constexpr_dynamic_alloc only guarantees constexpr
// std::allocator
#ifdef __cpp_constexpr_dynamic_alloc
    namespace internal
    {
        template<typename T>
        struct DynArray
        {
            constexpr DynArray() = default;

            constexpr explicit DynArray(std::size_t n) : data(new T[n]{})
            {
            }

            DynArray(const DynArray&) = delete;
            DynArray(DynArray&&) = delete;
            auto operator=(const DynArray&) -> DynArray& = delete;
            auto operator=(DynArray&&) -> DynArray& = delete;

            constexpr ~DynArray()
            {
                delete[] data;
            }

            constexpr void resize(std::size_t n)
            {
                delete[] data;
                data = new T[n]{};
            }

            T* data = nullptr; // TODO(bgruber): replace by std::unique_ptr in C++23
        };
    } // namespace internal

    /// Proofs by exhaustion of the array and record dimensions, that all values mapped to memory do not overlap.
    // Unfortunately, this only works for smallish array dimensions, because of compiler limits on constexpr evaluation
    // depth.
    LLAMA_EXPORT
    template<typename Mapping>
    constexpr auto mapsNonOverlappingly(const Mapping& m) -> bool
    {
        internal::DynArray<internal::DynArray<std::uint64_t>> blobByteMapped(m.blobCount);
        for(std::size_t i = 0; i < m.blobCount; i++)
            blobByteMapped.data[i].resize(divCeil(m.blobSize(i), std::size_t{64}));

        auto testAndSet = [&](auto blob, auto offset) constexpr
        {
            const auto bit = std::uint64_t{1} << (offset % 64);
            if(blobByteMapped.data[blob].data[offset / 64] & bit)
                return true;
            blobByteMapped.data[blob].data[offset / 64] |= bit;
            return false;
        };

        bool collision = false;
        forEachLeafCoord<typename Mapping::RecordDim>(
            [&](auto rc) constexpr
            {
                if(collision)
                    return;
                for(auto ai : ArrayIndexRange{m.extents()})
                {
                    using Type = GetType<typename Mapping::RecordDim, decltype(rc)>;
                    const auto [blob, offset] = m.blobNrAndOffset(ai, rc);
                    for(std::size_t b = 0; b < sizeof(Type); b++)
                        if(testAndSet(blob, offset + b))
                        {
                            collision = true;
                            break;
                        }
                }
            });
        return !collision;
    }
#endif

    /// Proofs by exhaustion of the array and record dimensions, that at least PieceLength elements are always stored
    /// contiguously.
    // Unfortunately, this only works for smallish array dimensions, because of compiler limits on constexpr evaluation
    // depth.
    LLAMA_EXPORT
    template<std::size_t PieceLength, typename Mapping>
    constexpr auto mapsPiecewiseContiguous(const Mapping& m) -> bool
    {
        bool collision = false;
        forEachLeafCoord<typename Mapping::RecordDim>(
            [&](auto rc) constexpr
            {
                std::size_t flatIndex = 0;
                std::size_t lastBlob = std::numeric_limits<std::size_t>::max();
                std::size_t lastOffset = std::numeric_limits<std::size_t>::max();
                for(auto ai : ArrayIndexRange{m.extents()})
                {
                    using Type = GetType<typename Mapping::RecordDim, decltype(rc)>;
                    const auto [blob, offset] = m.blobNrAndOffset(ai, rc);
                    if(flatIndex % PieceLength != 0 && (lastBlob != blob || lastOffset + sizeof(Type) != offset))
                    {
                        collision = true;
                        break;
                    }
                    lastBlob = blob;
                    lastOffset = offset;
                    flatIndex++;
                }
            });
        return !collision;
    }
} // namespace llama
// ==
// == ./include/llama/Proofs.hpp ==
// ============================================================================


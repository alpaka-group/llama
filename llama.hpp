#pragma once

// ============================================================================
// == ./Concepts.hpp ==
// ==
// #pragma once
	// ============================================================================
	// == ./Array.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
		// ============================================================================
		// == ./macros.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

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
		#    if defined(__NVCC__)
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
		#    define LLAMA_PRAGMA(tokens) _Pragma(#    tokens)
		#endif

		#ifndef LLAMA_UNROLL
		#    if defined(__NVCC__) || defined(__NVCOMPILER) || defined(__clang__) || defined(__INTEL_LLVM_COMPILER)
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
		#    if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
		#        define LLAMA_ACC __device__
		#    elif defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER)
		#        define LLAMA_ACC
		#    else
		#        define LLAMA_ACC
		#        warning LLAMA_HOST_ACC is only defined empty for this compiler
		#    endif
		#endif

		#ifndef LLAMA_HOST_ACC
		#    if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
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
		#    ifdef __CUDACC__
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
		#    ifdef __CUDACC__
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

		#if defined(_MSC_VER)
		#    define LLAMA_FORCE_INLINE_RECURSIVE __pragma(inline_depth(255))
		#else
		/// Forces the compiler to recursively inline the call hiearchy started by the subsequent function call.
		#    define LLAMA_FORCE_INLINE_RECURSIVE
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
		// ==
		// == ./macros.hpp ==
		// ============================================================================


	#include <ostream>
	#include <tuple>

	namespace llama
	{
	    /// Array class like `std::array` but suitable for use with offloading devices like GPUs.
	    /// \tparam T type if array elements.
	    /// \tparam N rank of the array.
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
	            return &element[N];
	        }

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto end() const -> const T*
	        {
	            return &element[N];
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
	    };

	    template<typename First, typename... Args>
	    Array(First, Args... args) -> Array<First, sizeof...(Args) + 1>;

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

	    template<typename T, std::size_t N>
	    LLAMA_FN_HOST_ACC_INLINE constexpr auto product(Array<T, N> a) -> T
	    {
	        T prod = 1;
	        for(auto s : a)
	            prod *= s;
	        return prod;
	    }

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

	namespace std
	{
	    template<typename T, size_t N>
	    struct tuple_size<llama::Array<T, N>> : integral_constant<size_t, N> // NOLINT(cert-dcl58-cpp)
	    {
	    };

	    template<size_t I, typename T, size_t N>
	    struct tuple_element<I, llama::Array<T, N>> // NOLINT(cert-dcl58-cpp)
	    {
	        using type = T;
	    };
	} // namespace std
	// ==
	// == ./Array.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./Core.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
		// ============================================================================
		// == ./ArrayExtents.hpp ==
		// ==
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "Array.hpp"    // amalgamate: file already expanded
			// ============================================================================
			// == ./Meta.hpp ==
			// ==
			// SPDX-License-Identifier: GPL-3.0-or-later

			// #pragma once
			#include <boost/mp11.hpp>

			#if BOOST_MP11_VERSION < 107300
			//  Copyright 2015 Peter Dimov.
			//
			//  Distributed under the Boost Software License, Version 1.0.
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

			namespace boost::mp11
			{
			    namespace detail
			    {
			        template<class L2>
			        struct mp_flatten_impl
			        {
			            template<class T>
			            using fn = mp_if<mp_similar<L2, T>, T, mp_list<T>>;
			        };
			    } // namespace detail

			    template<class L, class L2 = mp_clear<L>>
			    using mp_flatten = mp_apply<mp_append, mp_push_front<mp_transform_q<detail::mp_flatten_impl<L2>, L>, mp_clear<L>>>;
			} // namespace boost::mp11
			#endif

			namespace llama
			{
			    namespace internal
			    {
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
			        struct ReplacePlaceholdersImpl<boost::mp11::mp_arg<I>, Args...>
			        {
			            using type = boost::mp11::mp_at_c<boost::mp11::mp_list<Args...>, I>;
			        };

			        template<template<typename...> typename E, typename... Ts, typename... Args>
			        struct ReplacePlaceholdersImpl<E<Ts...>, Args...>
			        {
			            using type = E<typename ReplacePlaceholdersImpl<Ts, Args...>::type...>;
			        };
			    } // namespace internal

			    template<typename Expression, typename... Args>
			    using ReplacePlaceholders = typename internal::ReplacePlaceholdersImpl<Expression, Args...>::type;
			} // namespace llama
			// ==
			// == ./Meta.hpp ==
			// ============================================================================


		#include <limits>
		#include <type_traits>

		namespace llama
		{
		    // TODO(bgruber): make this an alias in C++20, when we have CTAD for aliases
		    /// Represents a run-time index into the array dimensions.
		    /// \tparam Dim Compile-time number of dimensions.
		    template<typename T, std::size_t Dim>
		    struct ArrayIndex : Array<T, Dim>
		    {
		        static constexpr std::size_t rank = Dim;
		    };

		    // allow comparing ArrayIndex with different size types:
		    template<std::size_t Dim, typename TA, typename TB>
		    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(ArrayIndex<TA, Dim> a, ArrayIndex<TB, Dim> b) -> bool
		    {
		        for(std::size_t i = 0; i < Dim; ++i)
		            if(a[i] != b[i])
		                return false;
		        return true;
		    }

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

		    template<typename... Args>
		    ArrayIndex(Args...)
		        -> ArrayIndex<typename internal::IndexTypeFromArgs<std::size_t, Args...>::type, sizeof...(Args)>;
		} // namespace llama

		template<typename V, size_t N>
		struct std::tuple_size<llama::ArrayIndex<V, N>> : std::integral_constant<size_t, N> // NOLINT(cert-dcl58-cpp)
		{
		};

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

		    /// Used as a template argument to \ref ArrayExtents to mark a dynamic extent.
		    inline constexpr auto dyn = internal::Dyn{};

		    /// ArrayExtents holding compile and runtime indices. This is conceptually equivalent to the std::extent of
		    /// std::mdspan (@see: https://wg21.link/P0009) including the changes to make the size_type controllable (@see:
		    /// https://wg21.link/P2553).
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
		            using namespace boost::mp11;
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
		            return boost::mp11::mp_with_index<rank>(i, [&](auto ic) { return get<decltype(ic)::value>(); });
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

		    template<typename... Args>
		    ArrayExtents(Args... args)
		        -> ArrayExtents<typename internal::IndexTypeFromArgs<std::size_t, Args...>::type, (Args{}, dyn)...>;

		    static_assert(std::is_trivially_default_constructible_v<ArrayExtents<std::size_t, 1>>);
		    static_assert(std::is_trivially_copy_constructible_v<ArrayExtents<std::size_t, 1>>);
		    static_assert(std::is_trivially_move_constructible_v<ArrayExtents<std::size_t, 1>>);
		    static_assert(std::is_trivially_copy_assignable_v<ArrayExtents<std::size_t, 1>>);
		    static_assert(std::is_trivially_move_assignable_v<ArrayExtents<std::size_t, 1>>);
		    static_assert(std::is_empty_v<ArrayExtents<std::size_t, 1>>);

		    template<typename SizeTypeA, SizeTypeA... SizesA, typename SizeTypeB, SizeTypeB... SizesB>
		    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(
		        ArrayExtents<SizeTypeA, SizesA...> a,
		        ArrayExtents<SizeTypeB, SizesB...> b) -> bool
		    {
		        return a.toArray() == b.toArray();
		    }

		    template<typename SizeTypeA, SizeTypeA... SizesA, typename SizeTypeB, SizeTypeB... SizesB>
		    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(
		        ArrayExtents<SizeTypeA, SizesA...> a,
		        ArrayExtents<SizeTypeB, SizesB...> b) -> bool
		    {
		        return !(a == b);
		    }

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

		    /// N-dimensional ArrayExtents where all N extents are Extent.
		    template<typename SizeType, std::size_t N, SizeType Extent>
		    using ArrayExtentsNCube = decltype(internal::makeArrayExtents<SizeType, Extent>(std::make_index_sequence<N>{}));

		    /// N-dimensional ArrayExtents where all values are dynamic.
		    template<typename SizeType, std::size_t N>
		    using ArrayExtentsDynamic = ArrayExtentsNCube<SizeType, N, dyn>;

		    template<typename SizeType, std::size_t Dim, typename Func, typename... OuterIndices>
		    LLAMA_FN_HOST_ACC_INLINE void forEachADCoord(
		        [[maybe_unused]] ArrayIndex<SizeType, Dim> adSize,
		        Func&& func,
		        OuterIndices... outerIndices)
		    {
		        if constexpr(Dim > 0)
		            for(SizeType i = 0; i < adSize[0]; i++)
		                forEachADCoord(
		                    ArrayIndex<SizeType, Dim - 1>{popFront(adSize)},
		                    std::forward<Func>(func),
		                    outerIndices...,
		                    i);
		        else
		            std::forward<Func>(func)(ArrayIndex<SizeType, sizeof...(outerIndices)>{outerIndices...});
		    }

		    template<typename SizeType, SizeType... Sizes, typename Func>
		    LLAMA_FN_HOST_ACC_INLINE void forEachADCoord(ArrayExtents<SizeType, Sizes...> extents, Func&& func)
		    {
		        forEachADCoord(extents.toArray(), std::forward<Func>(func));
		    }
		} // namespace llama

		template<typename SizeType, SizeType... Sizes>
		struct std::tuple_size<llama::ArrayExtents<SizeType, Sizes...>> // NOLINT(cert-dcl58-cpp)
		    : std::integral_constant<std::size_t, sizeof...(Sizes)>
		{
		};

		template<typename SizeType, std::size_t I, SizeType... Sizes>
		struct std::tuple_element<I, llama::ArrayExtents<SizeType, Sizes...>> // NOLINT(cert-dcl58-cpp)
		{
		    using type = SizeType;
		};
		// ==
		// == ./ArrayExtents.hpp ==
		// ============================================================================

	// #include "Meta.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./RecordCoord.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "Meta.hpp"    // amalgamate: file already expanded

		#include <array>
		// #include <ostream>    // amalgamate: file already included
		// #include <type_traits>    // amalgamate: file already included

		namespace llama
		{
		    /// Represents a coordinate for a record inside the record dimension tree.
		    /// \tparam Coords... the compile time coordinate.
		    template<std::size_t... Coords>
		    struct RecordCoord
		    {
		        /// The list of integral coordinates as `boost::mp11::mp_list`.
		        using List = boost::mp11::mp_list_c<std::size_t, Coords...>;

		        static constexpr std::size_t front = boost::mp11::mp_front<List>::value;
		        static constexpr std::size_t back = boost::mp11::mp_back<List>::value;
		        static constexpr std::size_t size = sizeof...(Coords);
		    };

		    template<>
		    struct RecordCoord<>
		    {
		        using List = boost::mp11::mp_list_c<std::size_t>;

		        static constexpr std::size_t size = 0;
		    };

		    template<std::size_t... CoordsA, std::size_t... CoordsB>
		    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<CoordsA...>, RecordCoord<CoordsB...>)
		    {
		        return false;
		    }

		    template<std::size_t... Coords>
		    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<Coords...>, RecordCoord<Coords...>)
		    {
		        return true;
		    }

		    template<std::size_t... CoordsA, std::size_t... CoordsB>
		    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(RecordCoord<CoordsA...> a, RecordCoord<CoordsB...> b)
		    {
		        return !(a == b);
		    }

		    template<typename T>
		    inline constexpr bool isRecordCoord = false;

		    template<std::size_t... Coords>
		    inline constexpr bool isRecordCoord<RecordCoord<Coords...>> = true;

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
		            }
		            ();
		            return RecordCoord<coord>{};
		        }
		    } // namespace literals

		    /// Converts a type list of integral constants into a \ref RecordCoord.
		    template<typename L>
		    using RecordCoordFromList = internal::mp_unwrap_values_into<L, RecordCoord>;

		    /// Concatenate a set of \ref RecordCoord%s.
		    template<typename... RecordCoords>
		    using Cat = RecordCoordFromList<boost::mp11::mp_append<typename RecordCoords::List...>>;

		    /// Concatenate a set of \ref RecordCoord%s instances.
		    template<typename... RecordCoords>
		    LLAMA_FN_HOST_ACC_INLINE constexpr auto cat(RecordCoords...)
		    {
		        return Cat<RecordCoords...>{};
		    }

		    /// RecordCoord without first coordinate component.
		    template<typename RecordCoord>
		    using PopFront = RecordCoordFromList<boost::mp11::mp_pop_front<typename RecordCoord::List>>;

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
		    template<typename First, typename Second>
		    inline constexpr auto recordCoordCommonPrefixIsSame
		        = internal::recordCoordCommonPrefixIsSameImpl(First{}, Second{});
		} // namespace llama
		// ==
		// == ./RecordCoord.hpp ==
		// ============================================================================


	#include <iostream>
	#include <string>
	// #include <type_traits>    // amalgamate: file already included

	namespace llama
	{
	    /// Anonymous naming for a \ref Field.
	    struct NoName
	    {
	    };

	    /// A type list of \ref Field%s which may be used to define a record dimension.
	    template<typename... Fields>
	    struct Record
	    {
	    };

	    /// @brief Tells whether the given type is allowed as a field type in LLAMA. Such types need to be trivially
	    /// constructible and trivially destructible.
	    template<typename T>
	    inline constexpr bool isAllowedFieldType = std::is_trivially_destructible_v<T>;

	    /// Record dimension tree node which may either be a leaf or refer to a child tree presented as another \ref
	    /// Record.
	    /// \tparam Tag Name of the node. May be any type (struct, class).
	    /// \tparam Type Type of the node. May be one of three cases. 1. another sub tree consisting of a nested \ref
	    /// Record. 2. an array of static size of any type, in which case a Record with as many \ref Field as the array
	    /// size is created, named \ref RecordCoord specialized on consecutive numbers I. 3. A scalar type different from
	    /// \ref Record, making this node a leaf of this type.
	    template<typename Tag, typename Type>
	    struct Field
	    {
	        static_assert(isAllowedFieldType<Type>, "This field's type is not allowed");
	    };

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

	    template<typename Int>
	    NrAndOffset(Int, Int) -> NrAndOffset<Int>;

	    template<typename TA, typename TB>
	    auto operator==(const NrAndOffset<TA>& a, const NrAndOffset<TB>& b) -> bool
	    {
	        return a.nr == b.nr && a.offset == b.offset;
	    }

	    template<typename TA, typename TB>
	    auto operator!=(const NrAndOffset<TA>& a, const NrAndOffset<TB>& b) -> bool
	    {
	        return !(a == b);
	    }

	    /// Get the tag from a \ref Field.
	    template<typename Field>
	    using GetFieldTag = boost::mp11::mp_first<Field>;

	    /// Get the type from a \ref Field.
	    template<typename Field>
	    using GetFieldType = boost::mp11::mp_second<Field>;

	    template<typename T>
	    inline constexpr auto isRecord = false;

	    template<typename... Fields>
	    inline constexpr auto isRecord<Record<Fields...>> = true;

	    namespace internal
	    {
	        template<typename RecordDim, typename RecordCoord>
	        struct GetTagsImpl;

	        template<typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
	        struct GetTagsImpl<Record<Fields...>, RecordCoord<FirstCoord, Coords...>>
	        {
	            using Field = boost::mp11::mp_at_c<boost::mp11::mp_list<Fields...>, FirstCoord>;
	            using ChildTag = GetFieldTag<Field>;
	            using ChildType = GetFieldType<Field>;
	            using type
	                = boost::mp11::mp_push_front<typename GetTagsImpl<ChildType, RecordCoord<Coords...>>::type, ChildTag>;
	        };

	        template<typename ChildType, std::size_t Count, std::size_t FirstCoord, std::size_t... Coords>
	        struct GetTagsImpl<ChildType[Count], RecordCoord<FirstCoord, Coords...>>
	        {
	            using ChildTag = RecordCoord<FirstCoord>;
	            using type
	                = boost::mp11::mp_push_front<typename GetTagsImpl<ChildType, RecordCoord<Coords...>>::type, ChildTag>;
	        };

	        template<typename T>
	        struct GetTagsImpl<T, RecordCoord<>>
	        {
	            using type = boost::mp11::mp_list<>;
	        };
	    } // namespace internal

	    /// Get the tags of all \ref Field%s from the root of the record dimension tree until to the node identified by
	    /// \ref RecordCoord.
	    template<typename RecordDim, typename RecordCoord>
	    using GetTags = typename internal::GetTagsImpl<RecordDim, RecordCoord>::type;

	    namespace internal
	    {
	        template<typename RecordDim, typename RecordCoord>
	        struct GetTagImpl
	        {
	            using type = boost::mp11::mp_back<GetTags<RecordDim, RecordCoord>>;
	        };

	        template<typename RecordDim>
	        struct GetTagImpl<RecordDim, RecordCoord<>>
	        {
	            using type = NoName;
	        };
	    } // namespace internal

	    /// Get the tag of the \ref Field at a \ref RecordCoord inside the record dimension tree.
	    template<typename RecordDim, typename RecordCoord>
	    using GetTag = typename internal::GetTagImpl<RecordDim, RecordCoord>::type;

	    /// Is true if, starting at two coordinates in two record dimensions, all subsequent nodes in the record dimension
	    /// tree have the same tag.
	    /// \tparam RecordDimA First record dimension.
	    /// \tparam LocalA \ref RecordCoord based on StartA along which the tags are compared.
	    /// \tparam RecordDimB second record dimension.
	    /// \tparam LocalB \ref RecordCoord based on StartB along which the tags are compared.
	    template<typename RecordDimA, typename LocalA, typename RecordDimB, typename LocalB>
	    inline constexpr auto hasSameTags = []() constexpr
	    {
	        if constexpr(LocalA::size != LocalB::size)
	            return false;
	        else if constexpr(LocalA::size == 0 && LocalB::size == 0)
	            return true;
	        else
	            return std::is_same_v<GetTags<RecordDimA, LocalA>, GetTags<RecordDimB, LocalB>>;
	    }
	    ();

	    namespace internal
	    {
	        template<typename FieldList, typename Tag>
	        struct FindFieldByTag
	        {
	            template<typename Field>
	            using HasTag = std::is_same<GetFieldTag<Field>, Tag>;

	            static constexpr auto value = boost::mp11::mp_find_if<FieldList, HasTag>::value;
	        };

	        template<typename RecordDim, typename RecordCoord, typename... Tags>
	        struct GetCoordFromTagsImpl
	        {
	            static_assert(boost::mp11::mp_size<RecordDim>::value != 0, "Tag combination is not valid");
	        };

	        template<typename... Fields, std::size_t... ResultCoords, typename FirstTag, typename... Tags>
	        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<ResultCoords...>, FirstTag, Tags...>
	        {
	            static constexpr auto tagIndex = FindFieldByTag<boost::mp11::mp_list<Fields...>, FirstTag>::value;
	            static_assert(
	                tagIndex < sizeof...(Fields),
	                "FirstTag was not found inside this Record. Does your record dimension contain the tag you access "
	                "with?");

	            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Fields...>, tagIndex>>;

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
	        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<>, boost::mp11::mp_list<Tags...>>
	            : GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<>, Tags...>
	        {
	        };
	        template<typename ChildType, std::size_t Count, typename... Tags>
	        struct GetCoordFromTagsImpl<ChildType[Count], RecordCoord<>, boost::mp11::mp_list<Tags...>>
	            : GetCoordFromTagsImpl<ChildType[Count], RecordCoord<>, Tags...>
	        {
	        };
	    } // namespace internal

	    /// Converts a series of tags, or a list of tags, navigating down a record dimension into a \ref RecordCoord.
	    template<typename RecordDim, typename... Tags>
	    using GetCoordFromTags = typename internal::GetCoordFromTagsImpl<RecordDim, RecordCoord<>, Tags...>::type;

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
	            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Children...>, HeadCoord>>;
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
	    template<typename RecordDim, typename... RecordCoordOrTags>
	    using GetType = typename internal::GetTypeImpl<RecordDim, RecordCoordOrTags...>::type;

	    namespace internal
	    {
	        template<typename RecordDim, typename RecordCoord>
	        struct LeafRecordCoordsImpl;

	        template<typename T, std::size_t... RCs>
	        struct LeafRecordCoordsImpl<T, RecordCoord<RCs...>>
	        {
	            using type = boost::mp11::mp_list<RecordCoord<RCs...>>;
	        };

	        template<typename... Fields, std::size_t... RCs>
	        struct LeafRecordCoordsImpl<Record<Fields...>, RecordCoord<RCs...>>
	        {
	            template<std::size_t... Is>
	            static auto help(std::index_sequence<Is...>)
	            {
	                return boost::mp11::mp_append<
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
	                return boost::mp11::mp_append<
	                    typename LeafRecordCoordsImpl<Child, RecordCoord<RCs..., Is>>::type...>{};
	            }
	            using type = decltype(help(std::make_index_sequence<N>{}));
	        };
	    } // namespace internal

	    /// Returns a flat type list containing all record coordinates to all leaves of the given record dimension.
	    template<typename RecordDim>
	    using LeafRecordCoords = typename internal::LeafRecordCoordsImpl<RecordDim, RecordCoord<>>::type;

	    namespace internal
	    {
	        // adapted from boost::mp11, but with LLAMA_FN_HOST_ACC_INLINE
	        template<template<typename...> typename L, typename... T, typename F>
	        LLAMA_FN_HOST_ACC_INLINE constexpr void mpForEachInlined(L<T...>, F&& f)
	        {
	            using A = int[sizeof...(T)];
	            (void) A{((void) f(T{}), 0)...};
	        }
	    } // namespace internal

	    /// Iterates over the record dimension tree and calls a functor on each element.
	    /// \param functor Functor to execute at each element of. Needs to have `operator()` with a template parameter for
	    /// the \ref RecordCoord in the record dimension tree.
	    /// \param baseCoord \ref RecordCoord at which the iteration should be started. The functor is called on elements
	    /// beneath this coordinate.
	    template<typename RecordDim, typename Functor, std::size_t... Coords>
	    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafCoord(Functor&& functor, RecordCoord<Coords...> baseCoord)
	    {
	        LLAMA_FORCE_INLINE_RECURSIVE
	        internal::mpForEachInlined(
	            LeafRecordCoords<GetType<RecordDim, RecordCoord<Coords...>>>{},
	            [&](auto innerCoord) LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr)
	            { std::forward<Functor>(functor)(cat(baseCoord, innerCoord)); });
	    }

	    /// Iterates over the record dimension tree and calls a functor on each element.
	    /// \param functor Functor to execute at each element of. Needs to have `operator()` with a template parameter for
	    /// the \ref RecordCoord in the record dimension tree.
	    /// \param baseTags Tags used to define where the iteration should be started. The functor is called on elements
	    /// beneath this coordinate.
	    template<typename RecordDim, typename Functor, typename... Tags>
	    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafCoord(Functor&& functor, Tags... /*baseTags*/)
	    {
	        LLAMA_FORCE_INLINE_RECURSIVE
	        forEachLeafCoord<RecordDim>(std::forward<Functor>(functor), GetCoordFromTags<RecordDim, Tags...>{});
	    }

	    namespace internal
	    {
	        template<typename T>
	        struct FlattenRecordDimImpl
	        {
	            using type = boost::mp11::mp_list<T>;
	        };

	        template<typename... Fields>
	        struct FlattenRecordDimImpl<Record<Fields...>>
	        {
	            using type = boost::mp11::mp_append<typename FlattenRecordDimImpl<GetFieldType<Fields>>::type...>;
	        };
	        template<typename Child, std::size_t N>
	        struct FlattenRecordDimImpl<Child[N]>
	        {
	            using type = boost::mp11::mp_repeat_c<typename FlattenRecordDimImpl<Child>::type, N>;
	        };
	    } // namespace internal

	    /// Returns a flat type list containing all leaf field types of the given record dimension.
	    template<typename RecordDim>
	    using FlatRecordDim = typename internal::FlattenRecordDimImpl<RecordDim>::type;

	    /// The total number of fields in the recursively expanded record dimension.
	    template<typename RecordDim>
	    inline constexpr std::size_t flatFieldCount = 1;

	    template<typename... Children>
	    inline constexpr std::size_t flatFieldCount<
	        Record<Children...>> = (flatFieldCount<GetFieldType<Children>> + ... + 0);

	    template<typename Child, std::size_t N>
	    inline constexpr std::size_t flatFieldCount<Child[N]> = flatFieldCount<Child>* N;

	    namespace internal
	    {
	        template<std::size_t I, typename RecordDim>
	        inline constexpr std::size_t flatFieldCountBefore = 0;

	        template<typename... Children>
	        inline constexpr std::size_t flatFieldCountBefore<0, Record<Children...>> = 0;

	        // recursive formulation to benefit from template instantiation memoization
	        // this massively improves compilation time when this template is instantiated with a lot of different I
	        template<std::size_t I, typename... Children>
	        inline constexpr std::size_t flatFieldCountBefore<
	            I,
	            Record<
	                Children...>> = flatFieldCountBefore<I - 1, Record<Children...>> + flatFieldCount<GetFieldType<boost::mp11::mp_at_c<Record<Children...>, I - 1>>>;
	    } // namespace internal

	    /// The equivalent zero based index into a flat record dimension (\ref FlatRecordDim) of the given hierarchical
	    /// record coordinate.
	    template<typename RecordDim, typename RecordCoord>
	    inline constexpr std::size_t flatRecordCoord = 0;

	    template<typename T>
	    inline constexpr std::size_t flatRecordCoord<T, RecordCoord<>> = 0;

	    template<typename... Children, std::size_t I, std::size_t... Is>
	    inline constexpr std::size_t flatRecordCoord<
	        Record<Children...>,
	        RecordCoord<
	            I,
	            Is...>> = internal::
	                          flatFieldCountBefore<
	                              I,
	                              Record<
	                                  Children...>> + flatRecordCoord<GetFieldType<boost::mp11::mp_at_c<Record<Children...>, I>>, RecordCoord<Is...>>;

	    template<typename Child, std::size_t N, std::size_t I, std::size_t... Is>
	    inline constexpr std::size_t flatRecordCoord<Child[N], RecordCoord<I, Is...>> = flatFieldCount<Child>* I
	        + flatRecordCoord<Child, RecordCoord<Is...>>;

	    namespace internal
	    {
	        template<typename TypeList>
	        constexpr auto flatAlignOfImpl()
	        {
	            using namespace boost::mp11;

	            std::size_t maxAlign = 0;
	            mp_for_each<mp_transform<mp_identity, TypeList>>([&](auto e) constexpr {
	                using T = typename decltype(e)::type;
	                maxAlign = std::max(maxAlign, alignof(T));
	            });
	            return maxAlign;
	        }
	    } // namespace internal

	    /// The alignment of a type list if its elements would be in a normal struct.
	    template<typename TypeList>
	    inline constexpr std::size_t flatAlignOf = internal::flatAlignOfImpl<TypeList>();

	    /// The alignment of a type T.
	    template<typename T>
	    inline constexpr std::size_t alignOf = alignof(T);

	    /// The alignment of a record dimension if its fields would be in a normal struct.
	    template<typename... Fields>
	    inline constexpr std::size_t alignOf<Record<Fields...>> = flatAlignOf<FlatRecordDim<Record<Fields...>>>;

	    /// Returns the ceiling of a / b.
	    template<typename Integral>
	    [[nodiscard]] LLAMA_FN_HOST_ACC_INLINE constexpr auto divCeil(Integral a, Integral b) -> Integral
	    {
	        return (a + b - 1) / b;
	    }

	    /// Returns the integral n rounded up to be a multiple of mult.
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
	            using namespace boost::mp11;

	            std::size_t size = 0;
	            std::size_t maxAlign = 0; // NOLINT(misc-const-correctness)
	            mp_for_each<mp_transform<mp_identity, TypeList>>([&](auto e) constexpr {
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
	                size = roundUpToMultiple(size, maxAlign); // TODO(bgruber): we could use flatAlignOf<TypeList> here, at
	                                                          // the cost of more template instantiations
	            return size;
	        }

	        template<typename TypeList, std::size_t I, bool Align>
	        constexpr auto offsetOfImplWorkaround() -> std::size_t;
	    } // namespace internal

	    /// The size of a type list if its elements would be in a normal struct.
	    template<typename TypeList, bool Align, bool IncludeTailPadding = true>
	    inline constexpr std::size_t flatSizeOf = internal::sizeOfImpl<TypeList, Align, IncludeTailPadding>();

	    /// The size of a type T.
	    template<typename T, bool Align = false, bool IncludeTailPadding = true>
	    inline constexpr std::size_t sizeOf = sizeof(T);

	    /// The size of a record dimension if its fields would be in a normal struct.
	    template<typename... Fields, bool Align, bool IncludeTailPadding>
	    inline constexpr std::size_t sizeOf<Record<Fields...>, Align, IncludeTailPadding> = flatSizeOf<
	        FlatRecordDim<Record<Fields...>>,
	        Align,
	        IncludeTailPadding>;

	    /// The byte offset of an element in a type list ifs elements would be in a normal struct.
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
	                    = flatOffsetOf<TypeList, I - 1, Align> + sizeof(boost::mp11::mp_at_c<TypeList, I - 1>);
	                if constexpr(Align)
	                    offset = roundUpToMultiple(offset, alignof(boost::mp11::mp_at_c<TypeList, I>));
	                return offset;
	            }
	        }
	    } // namespace internal

	    /// The byte offset of an element in a record dimension if it would be a normal struct.
	    /// \tparam RecordDim Record dimension tree.
	    /// \tparam RecordCoord Record coordinate of an element inrecord dimension tree.
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
	                    boost::mp11::mp_same<
	                        typename TransformLeavesWithCoordImpl<RecordCoord<Is..., Js>, Child, TypeFunctor>::type...>::
	                        value,
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
	    template<typename RecordDim, template<typename, typename> typename FieldTypeFunctor>
	    using TransformLeavesWithCoord =
	        typename internal::TransformLeavesWithCoordImpl<RecordCoord<>, RecordDim, FieldTypeFunctor>::type;

	    /// Creates a new record dimension where each new leaf field's type is the result of applying FieldTypeFunctor to
	    /// the original leaf field's type.
	    template<typename RecordDim, template<typename> typename FieldTypeFunctor>
	    using TransformLeaves
	        = TransformLeavesWithCoord<RecordDim, internal::MakePassSecond<FieldTypeFunctor>::template fn>;

	    namespace internal
	    {
	        // TODO(bgruber): we might implement this better by expanding a record dim into a list of tag lists and then
	        // computing a real set union of the two tag list lists

	        template<typename A, typename B>
	        auto mergeRecordDimsImpl(boost::mp11::mp_identity<A> a, boost::mp11::mp_identity<B>)
	        {
	            static_assert(std::is_same_v<A, B>, "Cannot merge record and non-record or fields with different types");
	            return a;
	        }

	        template<typename A, std::size_t NA, typename B, std::size_t NB>
	        auto mergeRecordDimsImpl(
	            [[maybe_unused]] boost::mp11::mp_identity<A[NA]> a,
	            [[maybe_unused]] boost::mp11::mp_identity<B[NB]> b)
	        {
	            static_assert(std::is_same_v<A, B>, "Cannot merge arrays of different type");
	            if constexpr(NA < NB)
	                return b;
	            else
	                return a;
	        }

	        template<typename... FieldsA>
	        auto mergeRecordDimsImpl(boost::mp11::mp_identity<Record<FieldsA...>> a, boost::mp11::mp_identity<Record<>>)
	        {
	            return a;
	        }

	        template<
	            typename... FieldsA,
	            typename FieldB,
	            typename... FieldsB,
	            auto Pos = FindFieldByTag<Record<FieldsA...>, GetFieldTag<FieldB>>::value>
	        auto mergeRecordDimsImpl(
	            boost::mp11::mp_identity<Record<FieldsA...>>,
	            boost::mp11::mp_identity<Record<FieldB, FieldsB...>>)
	        {
	            using namespace boost::mp11;
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
	    template<typename RecordDimA, typename RecordDimB>
	    using MergedRecordDims = typename decltype(internal::mergeRecordDimsImpl(
	        boost::mp11::mp_identity<RecordDimA>{},
	        boost::mp11::mp_identity<RecordDimB>{}))::type;

	    /// Alias for ToT, adding `const` if FromT is const qualified.
	    template<typename FromT, typename ToT>
	    using CopyConst = std::conditional_t<std::is_const_v<FromT>, const ToT, ToT>;

	    /// Used as template argument to specify a constant/compile-time value.
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
	} // namespace llama
	// ==
	// == ./Core.hpp ==
	// ============================================================================

// #include "RecordCoord.hpp"    // amalgamate: file already expanded

// #include <type_traits>    // amalgamate: file already included

#if __has_include(<concepts>)
#    include <concepts>
#endif
namespace llama
{
#ifdef __cpp_lib_concepts
    // clang-format off
    template <typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayExtents;
        typename M::ArrayIndex;
        typename M::RecordDim;
        { m.extents() } -> std::same_as<typename M::ArrayExtents>;
        { +M::blobCount } -> std::same_as<std::size_t>;
        std::integral_constant<std::size_t, M::blobCount>{}; // validates constexpr-ness
        { m.blobSize(typename M::ArrayExtents::value_type{}) } -> std::same_as<typename M::ArrayExtents::value_type>;
    };

    template <typename M, typename RC>
    concept PhysicalField = requires(M m, typename M::ArrayIndex ai) {
        { m.blobNrAndOffset(ai, RC{}) } -> std::same_as<NrAndOffset<typename M::ArrayExtents::value_type>>;
    };

    template<typename M>
    struct MakeIsPhysical
    {
        template<typename RC>
        using fn = boost::mp11::mp_bool<PhysicalField<M, RC>>;
    };

    template<typename M>
    inline constexpr bool allFieldsArePhysical
        = boost::mp11::mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsPhysical<M>::template fn>::value;

    template <typename M>
    concept PhysicalMapping = Mapping<M> && allFieldsArePhysical<M>;

    template <typename R>
    concept LValueReference = std::is_lvalue_reference_v<R>;

    template <typename R>
    concept ProxyReference = requires(R r) {
        typename R::value_type;
        { static_cast<typename R::value_type>(r) } -> std::same_as<typename R::value_type>;
        { r = std::declval<typename R::value_type>() } -> std::same_as<R&>;
    };

    template <typename R>
    concept AnyReference = LValueReference<R> || ProxyReference<R>;

    template <typename R, typename T>
    concept AnyReferenceTo = (LValueReference<R> && std::is_same_v<std::remove_cvref_t<R>, T>) || (ProxyReference<R> && std::is_same_v<typename R::value_type, T>);

    template <typename M, typename RC>
    concept ComputedField = M::isComputed(RC{}) && requires(M m, typename M::ArrayIndex ai, Array<Array<std::byte, 1>, 1> blobs) {
        { m.compute(ai, RC{}, blobs) } -> AnyReferenceTo<GetType<typename M::RecordDim, RC>>;
    };

    template<typename M>
    struct MakeIsComputed
    {
        template<typename RC>
        using fn = boost::mp11::mp_bool<ComputedField<M, RC>>;
    };

    template<typename M>
    inline constexpr bool allFieldsAreComputed
        = boost::mp11::mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsComputed<M>::template fn>::value;

    template <typename M>
    concept FullyComputedMapping = Mapping<M> && allFieldsAreComputed<M>;

    template<
        typename M,
        typename LeafCoords = LeafRecordCoords<typename M::RecordDim>,
        std::size_t PhysicalCount = boost::mp11::mp_count_if<LeafCoords, MakeIsPhysical<M>::template fn>::value,
        std::size_t ComputedCount = boost::mp11::mp_count_if<LeafCoords, MakeIsComputed<M>::template fn>::value>
    inline constexpr bool allFieldsArePhysicalOrComputed
        = (PhysicalCount + ComputedCount) >= boost::mp11::mp_size<LeafCoords>::value&& PhysicalCount > 0
        && ComputedCount > 0; // == instead of >= would be better, but it's not easy to count correctly,
                              // because we cannot check whether the call to blobNrOrOffset()
                              // or compute() is actually valid

    template <typename M>
    concept PartiallyComputedMapping = Mapping<M> && allFieldsArePhysicalOrComputed<M>;

    template<typename B>
    concept Blob = requires(B b, std::size_t i) {
        // according to http://eel.is/c++draft/intro.object#3 only std::byte and unsigned char can provide storage for
        // other types
        std::is_same_v<decltype(b[i]), std::byte&> || std::is_same_v<decltype(b[i]), unsigned char&>;
    };

    template <typename BA>
    concept BlobAllocator = requires(BA ba, std::integral_constant<std::size_t, 16> alignment, std::size_t size) {
        { ba(alignment, size) } -> Blob;
    };
        // clang-format on
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
                decltype(std::declval<R&>() = std::declval<typename R::value_type>())>> : std::true_type
        {
        };
    } // namespace internal

    template<typename R>
#ifdef __cpp_lib_concepts
    inline constexpr bool isProxyReference = ProxyReference<R>;
#else
    inline constexpr bool isProxyReference = internal::IsProxyReferenceImpl<R>::value;
#endif
} // namespace llama
// ==
// == ./Concepts.hpp ==
// ============================================================================

// ============================================================================
// == ./Tuple.hpp ==
// ==
// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
// #include "Meta.hpp"    // amalgamate: file already expanded
// #include "macros.hpp"    // amalgamate: file already expanded

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

    template<typename... Elements>
    struct LLAMA_DECLSPEC_EMPTY_BASES Tuple
    {
    };

    /// Tuple class like `std::tuple` but suitable for use with offloading devices like GPUs.
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
                int> = 0>
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

    template<typename... Elements>
    Tuple(Elements...) -> Tuple<std::remove_cv_t<std::remove_reference_t<Elements>>...>;

    template<std::size_t I, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto get(Tuple<Elements...>& tuple) -> auto&
    {
        using Base [[maybe_unused]] // clang claims Base is unused ...
        = internal::TupleLeaf<sizeof...(Elements) - I, boost::mp11::mp_at_c<llama::Tuple<Elements...>, I>>;
        return tuple.Base::value();
    }

    template<std::size_t I, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto get(const Tuple<Elements...>& tuple) -> const auto&
    {
        using Base [[maybe_unused]] // clang claims Base is unused ...
        = internal::TupleLeaf<sizeof...(Elements) - I, boost::mp11::mp_at_c<llama::Tuple<Elements...>, I>>;
        return tuple.Base::value();
    }
} // namespace llama

template<typename... Elements>
struct std::tuple_size<llama::Tuple<Elements...>> // NOLINT(cert-dcl58-cpp)
{
    static constexpr auto value = sizeof...(Elements);
};

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

    template<typename... ElementsA, typename... ElementsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(const Tuple<ElementsA...>& a, const Tuple<ElementsB...>& b)
        -> bool
    {
        using namespace boost::mp11;
        if constexpr(sizeof...(ElementsA) == sizeof...(ElementsB))
            if constexpr(mp_apply<mp_all, mp_transform<std::is_same, mp_list<ElementsA...>, mp_list<ElementsB...>>>::
                             value)
                return internal::areEqual(a, b, std::make_index_sequence<sizeof...(ElementsA)>{});
        return false;
    }

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
            // FIXME(bgruber): nvcc fails to compile
            // Tuple{functor(get<Is>(tuple))...}
            return Tuple<decltype(functor(std::declval<Elements>()))...>{functor(get<Is>(tuple))...};
        }
    } // namespace internal

    /// Applies a functor to every element of a tuple, creating a new tuple with the result of the element
    /// transformations. The functor needs to implement a template `operator()` to which all tuple elements are passed.
    // TODO(bgruber): replace by mp11 version in Boost 1.74.
    template<typename... Elements, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleTransform(const Tuple<Elements...>& tuple, const Functor& functor)
    {
        return internal::tupleTransformHelper(std::make_index_sequence<sizeof...(Elements)>{}, tuple, functor);
    }

    /// Returns a copy of the tuple without the first element.
    template<typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto popFront(const Tuple<Elements...>& tuple)
    {
        return tuple.rest();
    }
} // namespace llama
// ==
// == ./Tuple.hpp ==
// ============================================================================

// ============================================================================
// == ./ProxyRefOpMixin.hpp ==
// ==
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
// #include "macros.hpp"    // amalgamate: file already expanded

namespace llama
{
    /// CRTP mixin for proxy reference types to support all compound assignment and increment/decrement operators.
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
    };
} // namespace llama
// ==
// == ./ProxyRefOpMixin.hpp ==
// ============================================================================

// ============================================================================
// == ./Copy.hpp ==
// ==
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
	// ============================================================================
	// == ./View.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
		// ============================================================================
		// == ./Accessors.hpp ==
		// ==
		// #pragma once
		// #include "Concepts.hpp"    // amalgamate: file already expanded
		// #include "macros.hpp"    // amalgamate: file already expanded

		#include <atomic>

		namespace llama::accessor
		{
		    /// Default accessor. Passes through the given reference.
		    struct Default
		    {
		        template<typename Reference>
		        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const -> Reference
		        {
		            return std::forward<Reference>(r);
		        }
		    };

		    /// Allows only read access and returns values instead of references to memory.
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

		    /// Allows only read access by qualifying the references to memory with const. Only works on l-value references.
		    struct Const
		    {
		        template<typename T>
		        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> const T&
		        {
		            return r;
		        }
		    };

		    /// Qualifies references to memory with __restrict. Only works on l-value references.
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
		    struct Atomic
		    {
		        template<typename T>
		        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> std::atomic_ref<T>
		        {
		            return std::atomic_ref<T>{r};
		        }
		    };
		#endif
		} // namespace llama::accessor
		// ==
		// == ./Accessors.hpp ==
		// ============================================================================

	// #include "Array.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./ArrayIndexRange.hpp ==
		// ==
		// #pragma once
		// #include "ArrayExtents.hpp"    // amalgamate: file already expanded
		// #include "Core.hpp"    // amalgamate: file already expanded
			// ============================================================================
			// == ./HasRanges.hpp ==
			// ==
			// SPDX-License-Identifier: GPL-3.0-or-later

			// #pragma once
			// TODO(bgruber): clang 10-15 (libstdc++ from gcc 11.2 or gcc 12.1) fail to compile this currently with the issue
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
			// == ./HasRanges.hpp ==
			// ============================================================================


		#include <algorithm>
		#include <iterator>
		// #include <limits>    // amalgamate: file already included
		#if CAN_USE_RANGES
		#    include <ranges>
		#endif

		namespace llama
		{
		    /// Iterator supporting \ref ArrayIndexRange.
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
		            return std::lexicographical_compare(
		                std::begin(a.current),
		                std::end(a.current),
		                std::begin(b.current),
		                std::end(b.current));
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
		// == ./ArrayIndexRange.hpp ==
		// ============================================================================

		// ============================================================================
		// == ./BlobAllocators.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "Array.hpp"    // amalgamate: file already expanded
		// #include "Concepts.hpp"    // amalgamate: file already expanded
		// #include "macros.hpp"    // amalgamate: file already expanded

		#include <cstddef>
		#include <memory>
		#include <vector>
		#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION < 11000
		#    include <boost/shared_ptr.hpp>
		#endif
		#if __has_include(<cuda_runtime.h>)
		#    include <cuda_runtime.h>
		#endif
		#if __has_include(<alpaka/alpaka.hpp>)
		#    include <alpaka/alpaka.hpp>
		#endif

		namespace llama::bloballoc
		{
		    /// Allocates statically sized memory for a \ref View, which is copied each time a \ref View is copied.
		    /// \tparam BytesToReserve the amount of memory to reserve.
		    template<std::size_t BytesToReserve>
		    struct Array
		    {
		        template<std::size_t Alignment>
		        LLAMA_FN_HOST_ACC_INLINE auto operator()(
		            std::integral_constant<std::size_t, Alignment>,
		            [[maybe_unused]] std::size_t count) const
		        {
		            assert(count == BytesToReserve);
		            struct alignas(Alignment) AlignedArray : llama::Array<std::byte, BytesToReserve>
		            {
		            };
		            return AlignedArray{};
		        }
		    };
		#ifdef __cpp_lib_concepts
		    static_assert(BlobAllocator<Array<64>>);
		#endif

		    /// Allocates heap memory managed by a `std::unique_ptr` for a \ref View. This memory can only be uniquely owned by
		    /// a single \ref View.
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
		    struct SharedPtr
		    {
		        // libc++ below 11.0.0 does not yet support shared_ptr with arrays
		        template<typename T>
		        using shared_ptr =
		#if defined(_LIBCPP_VERSION) && _LIBCPP_VERSION < 11000
		            boost::shared_ptr<T>;
		#else
		            std::shared_ptr<T>;
		#endif

		        template<std::size_t Alignment>
		        auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
		            -> shared_ptr<std::byte[]>
		        {
		            auto* ptr
		                = static_cast<std::byte*>(::operator new[](count * sizeof(std::byte), std::align_val_t{Alignment}));
		            auto deleter = [](std::byte* ptr) { ::operator delete[](ptr, std::align_val_t{Alignment}); };
		            return shared_ptr<std::byte[]>{ptr, deleter};
		        }
		    };
		#ifdef __cpp_lib_concepts
		    static_assert(BlobAllocator<SharedPtr>);
		#endif

		    /// An STL compatible allocator allowing to specify alignment.
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
		    struct CudaMalloc
		    {
		        template<std::size_t Alignment>
		        inline auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
		        {
		            std::byte* p = nullptr;
		            if(const auto code = cudaMalloc(&p, count); code != cudaSuccess)
		                throw std::runtime_error(std::string{"cudaMalloc failed with code "} + cudaGetErrorString(code));
		            if(reinterpret_cast<std::uintptr_t>(p) & (Alignment - 1 != 0u))
		                throw std::runtime_error{"cudaMalloc does not align sufficiently"};
		            auto deleter = [](void* p)
		            {
		                if(const auto code = cudaFree(p); code != cudaSuccess)
		                    throw std::runtime_error(std::string{"cudaFree failed with code "} + cudaGetErrorString(code));
		            };
		            return std::unique_ptr<std::byte[], decltype(deleter)>(p, deleter);
		        }
		    };
		#endif

		#if __has_include(<alpaka/alpaka.hpp>)
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
		#endif
		} // namespace llama::bloballoc
		// ==
		// == ./BlobAllocators.hpp ==
		// ============================================================================

	// #include "Concepts.hpp"    // amalgamate: file already expanded
	// #include "Core.hpp"    // amalgamate: file already expanded
	// #include "HasRanges.hpp"    // amalgamate: file already expanded
	// #include "macros.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/One.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "../Core.hpp"    // amalgamate: file already expanded
			// ============================================================================
			// == ./mapping/Common.hpp ==
			// ==
			// Copyright 2018 Alexander Matthes
			// SPDX-License-Identifier: GPL-3.0-or-later

			// #pragma once
			// #include "../Core.hpp"    // amalgamate: file already expanded

			// #include <atomic>    // amalgamate: file already included
			#include <climits>
			#ifndef __cpp_lib_atomic_ref
			#    include <boost/atomic/atomic_ref.hpp>
			#endif

			namespace llama::mapping
			{
			    template<typename TArrayExtents, typename TRecordDim>
			    struct MappingBase : private TArrayExtents
			    {
			        using ArrayExtents = TArrayExtents;
			        using ArrayIndex = typename ArrayExtents::Index;
			        using RecordDim = TRecordDim;
			        using size_type = typename ArrayExtents::value_type;

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

			    /// Functor that maps an \ref ArrayIndex into linear numbers the way C++ arrays work. The fast moving index of the
			    /// ArrayIndex object should be the last one. E.g. ArrayIndex<3> a; stores 3 indices where a[2] should be
			    /// incremented in the innermost loop.
			    struct LinearizeArrayDimsCpp
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

			    /// Functor that maps a \ref ArrayIndex into linear numbers the way Fortran arrays work. The fast moving index of
			    /// the ArrayIndex object should be the last one. E.g. ArrayIndex<3> a; stores 3 indices where a[2] should be
			    /// incremented in the innermost loop.
			    struct LinearizeArrayDimsFortran
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

			    /// Functor that maps an \ref ArrayIndex into linear numbers using the Z-order space filling curve (Morton codes).
			    struct LinearizeArrayDimsMorton
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

			    /// Flattens the record dimension in the order fields are written.
			    template<typename RecordDim>
			    struct FlattenRecordDimInOrder
			    {
			        using FlatRecordDim = llama::FlatRecordDim<RecordDim>;

			        template<std::size_t... RecordCoords>
			        static constexpr std::size_t flatIndex = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
			    };

			    /// Flattens the record dimension by sorting the fields according to a given predicate on the field types.
			    /// @tparam Less A binary predicate accepting two field types, which exposes a member value. Value must be true if
			    /// the first field type is less than the second one, otherwise false.
			    template<typename RecordDim, template<typename, typename> typename Less>
			    struct FlattenRecordDimSorted
			    {
			    private:
			        using FlatOrigRecordDim = llama::FlatRecordDim<RecordDim>;
			        using FlatSortedRecordDim = boost::mp11::mp_sort<FlatOrigRecordDim, Less>;

			        template<typename A, typename B>
			        using LessWithIndices
			            = Less<boost::mp11::mp_at<FlatOrigRecordDim, A>, boost::mp11::mp_at<FlatOrigRecordDim, B>>;

			        // A permutation from new FlatSortedRecordDim index to old FlatOrigRecordDim index
			        using PermutedIndices
			            = boost::mp11::mp_sort<boost::mp11::mp_iota<boost::mp11::mp_size<FlatOrigRecordDim>>, LessWithIndices>;

			        template<typename A, typename B>
			        using LessInvertPermutation = std::bool_constant<(
			            boost::mp11::mp_at<PermutedIndices, A>::value < boost::mp11::mp_at<PermutedIndices, B>::value)>;

			        // A permutation from old FlatOrigRecordDim index to new FlatSortedRecordDim index
			        using InversePermutedIndices = boost::mp11::
			            mp_sort<boost::mp11::mp_iota<boost::mp11::mp_size<FlatOrigRecordDim>>, LessInvertPermutation>;

			    public:
			        using FlatRecordDim = FlatSortedRecordDim;

			        template<std::size_t... RecordCoords>
			        static constexpr std::size_t flatIndex = []() constexpr
			        {
			            constexpr auto indexBefore = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
			            constexpr auto indexAfter = boost::mp11::mp_at_c<InversePermutedIndices, indexBefore>::value;
			            return indexAfter;
			        }
			        ();
			    };

			    namespace internal
			    {
			        template<typename A, typename B>
			        using LessAlignment = std::bool_constant<alignof(A) < alignof(B)>;

			        template<typename A, typename B>
			        using MoreAlignment = std::bool_constant<(alignof(A) > alignof(B))>;
			    } // namespace internal

			    /// Flattens and sorts the record dimension by increasing alignment of its fields.
			    template<typename RecordDim>
			    using FlattenRecordDimIncreasingAlignment = FlattenRecordDimSorted<RecordDim, internal::LessAlignment>;

			    /// Flattens and sorts the record dimension by decreasing alignment of its fields.
			    template<typename RecordDim>
			    using FlattenRecordDimDecreasingAlignment = FlattenRecordDimSorted<RecordDim, internal::MoreAlignment>;

			    /// Flattens and sorts the record dimension by the alignment of its fields to minimize padding.
			    template<typename RecordDim>
			    using FlattenRecordDimMinimizePadding = FlattenRecordDimIncreasingAlignment<RecordDim>;

			    namespace internal
			    {
			        template<typename CountType>
			        LLAMA_FN_HOST_ACC_INLINE void atomicInc(CountType& i)
			        {
			#ifdef __CUDA_ARCH__
			            // if you get an error here that there is no overload of atomicAdd, your CMAKE_CUDA_ARCHITECTURE might be
			            // too low or you need to use a smaller CountType for the Trace or Heatmap mapping.
			            atomicAdd(&i, CountType{1});
			#elif defined(__cpp_lib_atomic_ref)
			            ++std::atomic_ref<CountType>{i};
			#else
			            ++boost::atomic_ref<CountType>{i};
			#endif
			        }
			    } // namespace internal
			} // namespace llama::mapping
			// ==
			// == ./mapping/Common.hpp ==
			// ============================================================================


		namespace llama::mapping
		{
		    /// Maps all array dimension indices to the same location and layouts struct members consecutively. This mapping is
		    /// used for temporary, single element views.
		    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
		    /// If false, struct members are tightly packed.
		    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
		    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
		    /// \ref FlattenRecordDimMinimizePadding.
		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        bool AlignAndPad = true,
		        template<typename> typename FlattenRecordDim = FlattenRecordDimMinimizePadding>
		    struct One : MappingBase<TArrayExtents, TRecordDim>
		    {
		    private:
		        using Base = MappingBase<TArrayExtents, TRecordDim>;
		        using size_type = typename Base::size_type;

		    public:
		        inline static constexpr bool alignAndPad = AlignAndPad;
		        using Flattener = FlattenRecordDim<TRecordDim>;
		        static constexpr std::size_t blobCount = 1;

		        using Base::Base;

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
		        {
		            return flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad, false>; // no tail padding
		        }

		        template<std::size_t... RecordCoords>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
		            typename Base::ArrayIndex,
		            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
		        {
		            constexpr std::size_t flatFieldIndex =
		#ifdef __NVCC__
		                *& // mess with nvcc compiler state to workaround bug
		#endif
		                 Flattener::template flatIndex<RecordCoords...>;
		            constexpr auto offset
		                = static_cast<size_type>(flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, AlignAndPad>);
		            return {size_type{0}, offset};
		        }
		    };

		    /// One mapping preserving the alignment of the field types by inserting padding.
		    /// \see One
		    template<typename ArrayExtents, typename RecordDim>
		    using AlignedOne = One<ArrayExtents, RecordDim, true, FlattenRecordDimInOrder>;

		    /// One mapping preserving the alignment of the field types by inserting padding and permuting the field order to
		    /// minimize this padding.
		    /// \see One
		    template<typename ArrayExtents, typename RecordDim>
		    using MinAlignedOne = One<ArrayExtents, RecordDim, true, FlattenRecordDimMinimizePadding>;

		    /// One mapping packing the field types tightly, violating the types' alignment requirements.
		    /// \see One
		    template<typename ArrayExtents, typename RecordDim>
		    using PackedOne = One<ArrayExtents, RecordDim, false, FlattenRecordDimInOrder>;

		    /// Binds parameters to a \ref One mapping except for array and record dimension, producing a quoted
		    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
		    template<bool AlignAndPad, template<typename> typename FlattenRecordDim>
		    struct BindOne
		    {
		        template<typename ArrayExtents, typename RecordDim>
		        using fn = One<ArrayExtents, RecordDim, AlignAndPad, FlattenRecordDim>;
		    };

		    template<typename Mapping>
		    inline constexpr bool isOne = false;

		    template<typename ArrayExtents, typename RecordDim, bool AlignAndPad, template<typename> typename FlattenRecordDim>
		    inline constexpr bool isOne<One<ArrayExtents, RecordDim, AlignAndPad, FlattenRecordDim>> = true;
		} // namespace llama::mapping
		// ==
		// == ./mapping/One.hpp ==
		// ============================================================================


	// #include <type_traits>    // amalgamate: file already included

	namespace llama
	{
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
	#ifdef __cpp_lib_concepts
	    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
	#else
	    template<typename Mapping, typename Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
	#endif
	    LLAMA_FN_HOST_ACC_INLINE auto allocViewUninitialized(
	        Mapping mapping = {},
	        const Allocator& alloc = {},
	        Accessor accessor = {})
	        -> View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Accessor>
	    {
	        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
	        return {std::move(mapping), std::move(blobs), std::move(accessor)};
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
	    template<typename Mapping, typename RecordCoord>
	    inline constexpr bool isComputed = internal::IsComputed<Mapping, RecordCoord>::value;

	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 4
	    namespace internal
	    {
	        template<typename View>
	        struct NvccWorkaroundLambda
	        {
	            using RecordDim = typename View::RecordDim;
	            using ArrayIndex = typename View::ArrayIndex;

	            template<typename RecordCoord>
	            void operator()(RecordCoord rc) const
	            {
	                using FieldType = GetType<RecordDim, decltype(rc)>;
	                using RefType = decltype(view(ai)(rc));
	                // this handles physical and computed mappings
	                if constexpr(std::is_lvalue_reference_v<RefType>)
	                {
	                    new(&view(ai)(rc)) FieldType;
	                }
	                else if constexpr(isProxyReference<RefType>)
	                {
	                    view(ai)(rc) = FieldType{};
	                }
	            }

	            View& view;
	            ArrayIndex ai;
	        };
	    } // namespace internal
	#endif

	    /// Runs the constructor of all fields reachable through the given view. Computed fields are constructed if they
	    /// return l-value references. If the mapping is a computed
	    template<typename Mapping, typename BlobType, typename Accessor>
	    LLAMA_FN_HOST_ACC_INLINE void constructFields(View<Mapping, BlobType, Accessor>& view)
	    {
	        using View = View<Mapping, BlobType, Accessor>;
	        using RecordDim = typename View::RecordDim;
	        forEachADCoord(
	            view.mapping().extents(),
	            [&]([[maybe_unused]] typename View::ArrayIndex ai)
	            {
	                if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
	                {
	                    forEachLeafCoord<RecordDim>(
	#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 4
	                        internal::NvccWorkaroundLambda<View>{view, ai}
	#else
	                        [&](auto rc)
	                        {
	                            using FieldType = GetType<RecordDim, decltype(rc)>;
	                            using RefType = decltype(view(ai)(rc));
	                            // this handles physical and computed mappings
	                            if constexpr(isProxyReference<RefType>)
	                            {
	                                view(ai)(rc) = FieldType{};
	                            }
	                            else if constexpr(
	                                std::is_lvalue_reference_v<
	                                    RefType> && !std::is_const_v<std::remove_reference_t<RefType>>)
	                            {
	                                new(&view(ai)(rc)) FieldType;
	                            }
	                        }
	#endif
	                    );
	                }
	                else
	                {
	                    // this handles physical and computed mappings
	                    using RefType = decltype(view(ai));
	                    if constexpr(isProxyReference<RefType>)
	                    {
	                        view(ai) = RecordDim{};
	                    }
	                    else if constexpr(
	                        std::is_lvalue_reference_v<RefType> && !std::is_const_v<std::remove_reference_t<RefType>>)
	                    {
	                        new(&view(ai)) RecordDim;
	                    }
	                }
	            });
	    }

	    /// Creates a view based on the given mapping, e.g. \ref AoS or \ref :SoA. For allocating the view's underlying
	    /// memory, the specified allocator callable is used (or the default one, which is \ref bloballoc::Vector). The
	    /// allocator callable is called with the alignment and size of bytes to allocate for each blob of the mapping.
	    /// The constructors are run for all fields by calling \ref constructFields. This function is the preferred way to
	    /// create a \ref View. See also \ref allocViewUninitialized.
	#ifdef __cpp_lib_concepts
	    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
	#else
	    template<typename Mapping, typename Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
	#endif
	    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {}, Accessor accessor = {})
	        -> View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Accessor>
	    {
	        auto view = allocViewUninitialized(std::move(mapping), alloc, accessor);
	        constructFields(view);
	        return view;
	    }

	    /// Same as \ref allocViewStack but does not run field constructors.
	    template<std::size_t Dim, typename RecordDim>
	    LLAMA_FN_HOST_ACC_INLINE auto allocViewStackUninitialized() -> decltype(auto)
	    {
	        constexpr auto mapping = mapping::MinAlignedOne<ArrayExtentsNCube<int, Dim, 1>, RecordDim>{};
	        return allocViewUninitialized(mapping, bloballoc::Array<mapping.blobSize(0)>{});
	    }

	    /// Allocates a \ref View holding a single record backed by stack memory (\ref bloballoc::Array).
	    /// \tparam Dim Dimension of the \ref ArrayExtents of the \ref View.
	    template<std::size_t Dim, typename RecordDim>
	    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
	    {
	        auto view = allocViewStackUninitialized<Dim, RecordDim>();
	        constructFields(view);
	        return view;
	    }

	    template<typename View, typename BoundRecordCoord = RecordCoord<>, bool OwnView = false>
	    struct RecordRef;

	    /// A \ref RecordRef that owns and holds a single value.
	    template<typename RecordDim>
	    using One = RecordRef<decltype(allocViewStack<0, RecordDim>()), RecordCoord<>, true>;

	    /// Is true, if T is an instance of \ref One.
	    template<typename T>
	    inline constexpr bool isOne = false;

	    template<typename View, typename BoundRecordCoord>
	    inline constexpr bool isOne<RecordRef<View, BoundRecordCoord, true>> = true;

	    // TODO(bgruber): Higher dimensional iterators might not have good codegen. Multiple nested loops seem to be
	    // superior to a single iterator over multiple dimensions. At least compilers are able to produce better code.
	    // std::mdspan also discovered similar difficulties and there was a discussion in WG21 in Oulu 2016 to
	    // remove/postpone iterators from the design. In std::mdspan's design, the iterator iterated over the co-domain.
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
	    template<typename Mapping, std::size_t... Coords, typename Blobs>
	    LLAMA_FN_HOST_ACC_INLINE auto mapToMemory(
	        Mapping& mapping,
	        typename Mapping::ArrayIndex ai,
	        RecordCoord<Coords...> rc,
	        Blobs& blobs) -> decltype(auto)
	    {
	        if constexpr(llama::isComputed<Mapping, RecordCoord<Coords...>>)
	            return mapping.compute(ai, rc, blobs);
	        else
	        {
	            const auto [nr, offset] = mapping.blobNrAndOffset(ai, rc);
	            using Type = GetType<typename Mapping::RecordDim, RecordCoord<Coords...>>;
	            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	            return reinterpret_cast<CopyConst<std::remove_reference_t<decltype(blobs[nr][offset])>, Type>&>(
	                blobs[nr][offset]);
	            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	        }
	    }

	    /// Central LLAMA class holding memory for storage and giving access to values stored there defined by a mapping. A
	    /// view should be created using \ref allocView.
	    /// \tparam TMapping The mapping used by the view to map accesses into memory.
	    /// \tparam TBlobType The storage type used by the view holding memory.
	    /// \tparam TAccessor The accessor to use when an access is made through this view.
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
	        using ArrayIndex = typename Mapping::ArrayIndex;
	        using RecordDim = typename Mapping::RecordDim;
	        using Accessor = TAccessor;
	        using iterator = Iterator<View>;
	        using const_iterator = Iterator<const View>;
	        using size_type = typename ArrayExtents::value_type;

	        static_assert(
	            std::is_same_v<Mapping, std::decay_t<Mapping>>,
	            "Mapping must not be const qualified or a reference. Are you using decltype(...) as View template "
	            "argument?");
	        static_assert(
	            std::is_same_v<ArrayExtents, std::decay_t<ArrayExtents>>,
	            "Mapping::ArrayExtents must not be const qualified or a reference. Are you using decltype(...) as mapping "
	            "template argument?");

	        /// Performs default initialization of the blob array.
	        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
	        View() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        View(Mapping mapping, Array<BlobType, Mapping::blobCount> storageBlobs, Accessor accessor = {})
	            : Mapping(std::move(mapping))
	            , Accessor(std::move(accessor))
	            , storageBlobs(std::move(storageBlobs))
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
	            if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return RecordRef<const View>{ai, *this};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return access(ai, RecordCoord<>{});
	            }
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
	        {
	            if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return RecordRef<View>{ai, *this};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return access(ai, RecordCoord<>{});
	            }
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
	            LLAMA_FORCE_INLINE_RECURSIVE
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
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this)(ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...});
	        }

	        /// Retrieves the \ref RecordRef at the \ref ArrayIndex index constructed from the passed component
	        /// indices.
	        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) const -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this)(ai);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
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
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this)(index);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type index) -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return (*this)(index);
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto begin() -> iterator
	        {
	            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.begin(), this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto begin() const -> const_iterator
	        {
	            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.begin(), this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto end() -> iterator
	        {
	            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.end(), this};
	        }

	        LLAMA_FN_HOST_ACC_INLINE
	        auto end() const -> const_iterator
	        {
	            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.end(), this};
	        }

	        Array<BlobType, Mapping::blobCount> storageBlobs;

	    private:
	        template<typename TView, typename TBoundRecordCoord, bool OwnView>
	        friend struct RecordRef;

	        template<std::size_t... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto access(ArrayIndex ai, RecordCoord<Coords...> rc = {}) const -> decltype(auto)
	        {
	            return accessor()(mapToMemory(mapping(), ai, rc, storageBlobs));
	        }

	        template<std::size_t... Coords>
	        LLAMA_FN_HOST_ACC_INLINE auto access(ArrayIndex ai, RecordCoord<Coords...> rc = {}) -> decltype(auto)
	        {
	            return accessor()(mapToMemory(mapping(), ai, rc, storageBlobs));
	        }
	    };

	    template<typename View>
	    inline constexpr auto isView = false;

	    template<typename Mapping, typename BlobType, typename Accessor>
	    inline constexpr auto isView<View<Mapping, BlobType, Accessor>> = true;

	    namespace internal
	    {
	        template<typename Blobs, typename TransformBlobFunc, std::size_t... Is>
	        LLAMA_FN_HOST_ACC_INLINE auto makeTransformedBlobArray(
	            Blobs& storageBlobs,
	            const TransformBlobFunc& transformBlob,
	            std::integer_sequence<std::size_t, Is...>)
	        {
	            return llama::Array{transformBlob(storageBlobs[Is])...};
	        }
	    } // namespace internal

	    /// Applies the given transformation to the blobs of a view and creates a new view with the transformed blobs and
	    /// the same mapping and accessor as the old view.
	    template<typename View, typename TransformBlobFunc, typename = std::enable_if_t<isView<std::decay_t<View>>>>
	    LLAMA_FN_HOST_ACC_INLINE auto transformBlobs(View& view, const TransformBlobFunc& transformBlob)
	    {
	        constexpr auto blobCount = std::decay_t<View>::Mapping::blobCount;
	        auto blobs = internal::makeTransformedBlobArray(
	            view.storageBlobs,
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
	    template<
	        typename View,
	        typename NewBlobType = CopyConst<View, std::byte>*,
	        typename = std::enable_if_t<isView<std::decay_t<View>>>>
	    LLAMA_FN_HOST_ACC_INLINE auto shallowCopy(View& view)
	    {
	        if constexpr(std::is_same_v<typename std::decay_t<View>::BlobType, NewBlobType>)
	            return view;
	        else
	            return transformBlobs(
	                view,
	                [](auto& blob)
	                {
	                    LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	                    return static_cast<NewBlobType>(&blob[0]);
	                    LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	                });
	    }

	    // Creates a new view from an existing view with the given accessor.
	    // \param view A view which's mapping and blobs are copied into a new view with the different accessor. If you no
	    // longer need the old view, consider moving it into the argument of this function.
	    template<typename NewAccessor, typename Mapping, typename BlobType, typename OldAccessor>
	    LLAMA_FN_HOST_ACC_INLINE auto withAccessor(View<Mapping, BlobType, OldAccessor> view, NewAccessor newAccessor = {})
	    {
	        return View<Mapping, BlobType, NewAccessor>{
	            std::move(view.mapping()),
	            std::move(view.storageBlobs),
	            std::move(newAccessor)};
	    }

	    // Creates a new view from an existing view with the given mapping.
	    // \param view A view which's accessor and blobs are copied into a new view with the different mapping. If you no
	    // longer need the old view, consider moving it into the argument of this function.
	    template<typename NewMapping, typename Mapping, typename BlobType, typename Accessor>
	    LLAMA_FN_HOST_ACC_INLINE auto withMapping(View<Mapping, BlobType, Accessor> view, NewMapping newMapping = {})
	    {
	        static_assert(Mapping::blobCount == NewMapping::blobCount);
	        for(std::size_t i = 0; i < Mapping::blobCount; i++)
	        {
	            assert(view.mapping().blobSize(i) == newMapping.blobSize(i));
	        }

	        return View<NewMapping, BlobType, Accessor>{
	            std::move(newMapping),
	            std::move(view.storageBlobs),
	            std::move(view.accessor())};
	    }

	    /// Like a \ref View, but array indices are shifted.
	    /// @tparam TStoredParentView Type of the underlying view. May be cv qualified and/or a reference type.
	    template<typename TStoredParentView>
	    struct SubView
	    {
	        using StoredParentView = TStoredParentView;
	        using ParentView = std::remove_const_t<std::remove_reference_t<StoredParentView>>; ///< type of the parent view
	        using Mapping = typename ParentView::Mapping; ///< mapping of the parent view
	        using ArrayExtents = typename Mapping::ArrayExtents; ///< array extents of the parent view
	        using ArrayIndex = typename Mapping::ArrayIndex; ///< array index of the parent view

	        using size_type = typename ArrayExtents::value_type;

	        /// Creates a SubView given a parent \ref View and offset.
	        template<typename StoredParentViewFwd>
	        LLAMA_FN_HOST_ACC_INLINE SubView(StoredParentViewFwd&& parentView, ArrayIndex offset)
	            : parentView(std::forward<StoredParentViewFwd>(parentView))
	            , offset(offset)
	        {
	        }

	        /// Same as \ref View::operator()(ArrayIndex), but shifted by the offset of this \ref SubView.
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) const -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
	            return parentView(ArrayIndex{ai + offset});
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
	        {
	            LLAMA_FORCE_INLINE_RECURSIVE
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
	            LLAMA_FORCE_INLINE_RECURSIVE
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
	            LLAMA_FORCE_INLINE_RECURSIVE
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

	        StoredParentView parentView;
	        const ArrayIndex offset; ///< offset by which this view's \ref ArrayIndex indices are shifted when passed to
	                                 ///< the parent view.
	    };

	    /// SubView vview(view); will store a reference to view.
	    /// SubView vview(std::move(view)); will store the view.
	    template<typename TStoredParentView>
	    SubView(TStoredParentView&&, typename std::remove_reference_t<TStoredParentView>::Mapping::ArrayIndex)
	        -> SubView<TStoredParentView>;
	} // namespace llama
	// ==
	// == ./View.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/AoSoA.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	// #include <limits>    // amalgamate: file already included

	namespace llama::mapping
	{
	    /// The maximum number of vector lanes that can be used to fetch each leaf type in the record dimension into a
	    /// vector register of the given size in bits.
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
	    }
	    ();

	    /// Array of struct of arrays mapping. Used to create a \ref View via \ref allocView.
	    /// \tparam Lanes The size of the inner arrays of this array of struct of arrays.
	    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
	    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
	    /// \ref FlattenRecordDimMinimizePadding.
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        typename TArrayExtents::value_type Lanes,
	        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
	        template<typename> typename FlattenRecordDim = FlattenRecordDimInOrder>
	    struct AoSoA : MappingBase<TArrayExtents, TRecordDim>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using size_type = typename Base::size_type;

	    public:
	        inline static constexpr typename TArrayExtents::value_type lanes = Lanes;
	        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
	        using Flattener = FlattenRecordDim<TRecordDim>;
	        inline static constexpr std::size_t blobCount = 1;

	        using Base::Base;

	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
	        {
	            const auto rs = static_cast<size_type>(sizeOf<TRecordDim>);
	            return roundUpToMultiple(LinearizeArrayDimsFunctor{}.size(Base::extents()) * rs, Lanes * rs);
	        }

	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            typename Base::ArrayIndex ai,
	            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
	        {
	            constexpr std::size_t flatFieldIndex =
	#ifdef __NVCC__
	                *& // mess with nvcc compiler state to workaround bug
	#endif
	                 Flattener::template flatIndex<RecordCoords...>;
	            const auto flatArrayIndex = LinearizeArrayDimsFunctor{}(ai, Base::extents());
	            const auto blockIndex = flatArrayIndex / Lanes;
	            const auto laneIndex = flatArrayIndex % Lanes;
	            const auto offset = static_cast<size_type>(sizeOf<TRecordDim> * Lanes) * blockIndex
	                + static_cast<size_type>(flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, false>)
	                    * Lanes
	                + static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>)) * laneIndex;
	            return {0, offset};
	        }
	    };

	    /// Binds parameters to an \ref AoSoA mapping except for array and record dimension, producing a quoted meta
	    /// function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    template<std::size_t Lanes, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct BindAoSoA
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = AoSoA<ArrayExtents, RecordDim, Lanes, LinearizeArrayDimsFunctor>;
	    };

	    template<typename Mapping>
	    inline constexpr bool isAoSoA = false;

	    template<typename AD, typename RD, typename AD::value_type L>
	    inline constexpr bool isAoSoA<AoSoA<AD, RD, L>> = true;

	} // namespace llama::mapping
	// ==
	// == ./mapping/AoSoA.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/SoA.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	// #include <limits>    // amalgamate: file already included

	namespace llama::mapping
	{
	    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
	    /// \tparam SeparateBuffers If true, every element of the record dimension is mapped to its own buffer.
	    /// \tparam AlignSubArrays Only relevant when SeparateBuffers == false. If true, aligns the sub arrays created
	    /// within the single blob by inserting padding.
	    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    /// \tparam FlattenRecordDimSingleBlob Defines how the record dimension's fields should be flattened if
	    /// SeparateBuffers is false. See \ref FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref
	    /// FlattenRecordDimDecreasingAlignment and \ref FlattenRecordDimMinimizePadding.
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        bool SeparateBuffers = true,
	        bool AlignSubArrays = false,
	        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
	        template<typename> typename FlattenRecordDimSingleBlob = FlattenRecordDimInOrder>
	    struct SoA : MappingBase<TArrayExtents, TRecordDim>
	    {
	    private:
	        using Base = MappingBase<TArrayExtents, TRecordDim>;
	        using size_type = typename TArrayExtents::value_type;

	    public:
	        inline static constexpr bool separateBuffers = SeparateBuffers;
	        inline static constexpr bool alignSubArrays = AlignSubArrays;
	        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
	        using Flattener = FlattenRecordDimSingleBlob<TRecordDim>;
	        inline static constexpr std::size_t blobCount
	            = SeparateBuffers ? boost::mp11::mp_size<FlatRecordDim<TRecordDim>>::value : 1;

	        using Base::Base;

	        LLAMA_FN_HOST_ACC_INLINE
	        constexpr auto blobSize([[maybe_unused]] size_type blobIndex) const -> size_type
	        {
	            const auto flatSize = LinearizeArrayDimsFunctor{}.size(Base::extents());
	            if constexpr(SeparateBuffers)
	            {
	                constexpr auto typeSizes = []() constexpr
	                {
	                    Array<size_type, blobCount> r{};
	                    forEachLeafCoord<TRecordDim>([&r, i = 0](auto rc) mutable constexpr {
	                        r[i++] = sizeof(GetType<TRecordDim, decltype(rc)>);
	                    });
	                    return r;
	                }
	                ();
	                return flatSize * typeSizes[blobIndex];
	            }
	            else if constexpr(AlignSubArrays)
	            {
	                size_type size = 0;
	                using namespace boost::mp11;
	                using FRD = typename Flattener::FlatRecordDim;
	                mp_for_each<mp_transform<mp_identity, FRD>>(
	                    [&](auto ti)
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

	        template<std::size_t... RecordCoords>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
	            typename Base::ArrayIndex ad,
	            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
	        {
	            if constexpr(SeparateBuffers)
	            {
	                constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
	                const auto offset = LinearizeArrayDimsFunctor{}(ad, Base::extents())
	                    * static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>));
	                return {blob, offset};
	            }
	            else
	            {
	                const auto subArrayOffset = LinearizeArrayDimsFunctor{}(ad, Base::extents())
	                    * static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>));
	                constexpr std::size_t flatFieldIndex =
	#ifdef __NVCC__
	                    *& // mess with nvcc compiler state to workaround bug
	#endif
	                     Flattener::template flatIndex<RecordCoords...>;
	                const auto flatSize = LinearizeArrayDimsFunctor{}.size(Base::extents());
	                using FRD = typename Flattener::FlatRecordDim;
	                if constexpr(AlignSubArrays)
	                {
	                    // TODO(bgruber): we can take a shortcut here if we know that flatSize is a multiple of all type's
	                    // alignment. We can also precompute a table of sub array starts (and maybe store it), or rely on
	                    // the compiler pulling it out of loops.
	                    using namespace boost::mp11;
	                    size_type offset = 0;
	                    mp_for_each<mp_transform<mp_identity, mp_take_c<FRD, flatFieldIndex>>>(
	                        [&](auto ti)
	                        {
	                            using FieldType = typename decltype(ti)::type;
	                            offset = roundUpToMultiple(offset, static_cast<size_type>(alignof(FieldType)));
	                            offset += static_cast<size_type>(sizeof(FieldType)) * flatSize;
	                        });
	                    offset = roundUpToMultiple(offset, static_cast<size_type>(alignof(mp_at_c<FRD, flatFieldIndex>)));
	                    offset += subArrayOffset;
	                    return {0, offset};
	                }
	                else
	                {
	                    const auto offset
	                        = subArrayOffset + static_cast<size_type>(flatOffsetOf<FRD, flatFieldIndex, false>) * flatSize;
	                    return {0, offset};
	                }
	            }
	        }
	    };

	    // we can drop this when inherited ctors also inherit deduction guides
	    template<typename TArrayExtents, typename TRecordDim>
	    SoA(TArrayExtents, TRecordDim) -> SoA<TArrayExtents, TRecordDim>;

	    /// Struct of array mapping storing the entire layout in a single blob. The starts of the sub arrays are aligned by
	    /// inserting padding. \see SoA
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    using AlignedSingleBlobSoA = SoA<ArrayExtents, RecordDim, false, true, LinearizeArrayDimsFunctor>;

	    /// Struct of array mapping storing the entire layout in a single blob. The sub arrays are tightly packed,
	    /// violating the type's alignment requirements. \see SoA
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    using PackedSingleBlobSoA = SoA<ArrayExtents, RecordDim, false, false, LinearizeArrayDimsFunctor>;

	    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
	    /// \see SoA
	    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    using MultiBlobSoA = SoA<ArrayExtents, RecordDim, true, false, LinearizeArrayDimsFunctor>;

	    /// Binds parameters to an \ref SoA mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    template<
	        bool SeparateBuffers = true,
	        bool AlignSubArrays = false,
	        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
	    struct BindSoA
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = SoA<ArrayExtents, RecordDim, SeparateBuffers, AlignSubArrays, LinearizeArrayDimsFunctor>;
	    };

	    template<typename Mapping>
	    inline constexpr bool isSoA = false;

	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        bool SeparateBuffers,
	        bool AlignSubArrays,
	        typename LinearizeArrayDimsFunctor>
	    inline constexpr bool
	        isSoA<SoA<ArrayExtents, RecordDim, SeparateBuffers, AlignSubArrays, LinearizeArrayDimsFunctor>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/SoA.hpp ==
	// ============================================================================


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

        using memcopyFunc = void* (*) (void*, const void*, std::size_t);

        inline void parallelMemcpy(
            std::byte* dst,
            const std::byte* src,
            std::size_t size,
            std::size_t threadId = 0,
            std::size_t threadCount = 1,
            memcopyFunc singleThreadMemcpy = std::memcpy)
        {
            const auto sizePerThread = size / threadCount;
            const auto sizeLastThread = sizePerThread + size % threadCount;
            const auto sizeThisThread = threadId == threadCount - 1 ? sizeLastThread : sizePerThread;
            singleThreadMemcpy(dst + threadId * sizePerThread, src + threadId * sizePerThread, sizeThisThread);
        }
    } // namespace internal

    /// Direct memcpy from source view blobs to destination view blobs. Both views need to have the same mappings with
    /// the same array dimensions.
    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
    template<typename Mapping, typename SrcBlob, typename DstBlob>
    void blobMemcpy(
        const View<Mapping, SrcBlob>& srcView,
        View<Mapping, DstBlob>& dstView,
        std::size_t threadId = 0,
        std::size_t threadCount = 1)
    {
        internal::assertTrivialCopyable<typename Mapping::RecordDim>();

        // TODO(bgruber): we do not verify if the mappings have other runtime state than the array dimensions
        if(srcView.mapping().extents() != dstView.mapping().extents())
            throw std::runtime_error{"Array dimensions sizes are different"};

        // TODO(bgruber): this is maybe not the best parallel copying strategy
        for(std::size_t i = 0; i < Mapping::blobCount; i++)
            internal::parallelMemcpy(
                &dstView.storageBlobs[i][0],
                &srcView.storageBlobs[i][0],
                dstView.mapping().blobSize(i),
                threadId,
                threadCount);
    }

    /// Field-wise copy from source to destination view. Both views need to have the same array and record dimensions.
    /// @param threadId Optional. Thread id in case of multi-threaded copy.
    /// @param threadCount Optional. Thread count in case of multi-threaded copy.
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

        if(srcView.mapping().extents() != dstView.mapping().extents())
            throw std::runtime_error{"Array dimensions sizes are different"};

        auto copyOne = [&](auto ai) LLAMA_LAMBDA_INLINE
        {
            forEachLeafCoord<typename DstMapping::RecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE
                                                             { dstView(ai)(rc) = srcView(ai)(rc); });
        };

        constexpr auto dims = SrcMapping::ArrayExtents::rank;
        const auto extents = srcView.mapping().extents().toArray();
        const auto workPerThread = (extents[0] + threadCount - 1) / threadCount;
        const auto start = threadId * workPerThread;
        const auto end = std::min((threadId + 1) * workPerThread, static_cast<std::size_t>(extents[0]));
        for(auto i = start; i < end; i++)
        {
            using SrcSizeType = typename SrcMapping::ArrayExtents::value_type;
            if constexpr(dims > 1)
                forEachADCoord(
                    ArrayIndex<SrcSizeType, dims - 1>{popFront(extents)},
                    copyOne,
                    static_cast<SrcSizeType>(i));
            else
                copyOne(ArrayIndex<SrcSizeType, dims>{static_cast<std::size_t>(i)});
        }
    }

    namespace internal
    {
        template<typename Mapping>
        inline constexpr std::size_t aosoaLanes = 0;

        template<
            typename ArrayExtents,
            typename RecordDim,
            bool SeparateBuffers,
            bool AlignSubArrays,
            typename LinearizeArrayDimsFunctor>
        inline constexpr std::size_t aosoaLanes<
            mapping::SoA<ArrayExtents, RecordDim, SeparateBuffers, AlignSubArrays, LinearizeArrayDimsFunctor>> = std::
            numeric_limits<std::size_t>::max();

        template<typename ArrayExtents, typename RecordDim, std::size_t Lanes, typename LinearizeArrayDimsFunctor>
        inline constexpr std::size_t
            aosoaLanes<mapping::AoSoA<ArrayExtents, RecordDim, Lanes, LinearizeArrayDimsFunctor>> = Lanes;
    } // namespace internal

    /// AoSoA copy strategy which transfers data in common blocks. SoA mappings are also allowed for at most 1
    /// argument.
    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
    template<typename SrcMapping, typename SrcBlob, typename DstMapping, typename DstBlob>
    void aosoaCommonBlockCopy(
        const View<SrcMapping, SrcBlob>& srcView,
        View<DstMapping, DstBlob>& dstView,
        bool readOpt,
        std::size_t threadId = 0,
        std::size_t threadCount = 1)
    {
        // TODO(bgruber): think if we can remove this restriction
        static_assert(
            std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>,
            "The source and destination record dimensions must be the same");
        static_assert(
            std::is_same_v<
                typename SrcMapping::LinearizeArrayDimsFunctor,
                typename DstMapping::LinearizeArrayDimsFunctor>,
            "Source and destination mapping need to use the same array dimensions linearizer");
        using RecordDim = typename SrcMapping::RecordDim;
        internal::assertTrivialCopyable<RecordDim>();

        [[maybe_unused]] static constexpr bool isSrcMB = SrcMapping::blobCount > 1;
        [[maybe_unused]] static constexpr bool isDstMB = DstMapping::blobCount > 1;
        static constexpr auto lanesSrc = internal::aosoaLanes<SrcMapping>;
        static constexpr auto lanesDst = internal::aosoaLanes<DstMapping>;

        if(srcView.mapping().extents() != dstView.mapping().extents())
            throw std::runtime_error{"Array dimensions sizes are different"};

        static constexpr auto srcIsAoSoA = lanesSrc != std::numeric_limits<std::size_t>::max();
        static constexpr auto dstIsAoSoA = lanesDst != std::numeric_limits<std::size_t>::max();

        static_assert(srcIsAoSoA || dstIsAoSoA, "At least one of the mappings must be an AoSoA mapping");
        static_assert(
            !srcIsAoSoA || std::tuple_size_v<decltype(srcView.storageBlobs)> == 1,
            "Implementation assumes AoSoA with single blob");
        static_assert(
            !dstIsAoSoA || std::tuple_size_v<decltype(dstView.storageBlobs)> == 1,
            "Implementation assumes AoSoA with single blob");

        const auto flatSize = product(dstView.mapping().extents());

        // TODO(bgruber): implement the following by adding additional copy loops for the remaining elements
        if(!srcIsAoSoA && flatSize % lanesDst != 0)
            throw std::runtime_error{"Source SoA mapping's total array elements must be evenly divisible by the "
                                     "destination AoSoA Lane count."};
        if(!dstIsAoSoA && flatSize % lanesSrc != 0)
            throw std::runtime_error{"Destination SoA mapping's total array elements must be evenly divisible by the "
                                     "source AoSoA Lane count."};

        // the same as AoSoA::blobNrAndOffset but takes a flat array index
        auto mapAoSoA = [](std::size_t flatArrayIndex, auto rc, std::size_t Lanes) LLAMA_LAMBDA_INLINE
        {
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = (sizeOf<RecordDim> * Lanes) * blockIndex + offsetOf<RecordDim, decltype(rc)> * Lanes
                + sizeof(GetType<RecordDim, decltype(rc)>) * laneIndex;
            return offset;
        };
        // the same as SoA::blobNrAndOffset but takes a flat array index
        auto mapSoA = [&](std::size_t flatArrayIndex, auto rc, bool mb) LLAMA_LAMBDA_INLINE
        {
            const auto blob = mb * flatRecordCoord<RecordDim, decltype(rc)>;
            const auto offset = !mb * offsetOf<RecordDim, decltype(rc)> * flatSize
                + sizeof(GetType<RecordDim, decltype(rc)>) * flatArrayIndex;
            return NrAndOffset{blob, offset};
        };

        auto mapSrc = [&](std::size_t flatArrayIndex, auto rc) LLAMA_LAMBDA_INLINE
        {
            if constexpr(srcIsAoSoA)
                return &srcView.storageBlobs[0][0] + mapAoSoA(flatArrayIndex, rc, lanesSrc);
            else
            {
                const auto [blob, off] = mapSoA(flatArrayIndex, rc, isSrcMB);
                return &srcView.storageBlobs[blob][off];
            }
        };
        auto mapDst = [&](std::size_t flatArrayIndex, auto rc) LLAMA_LAMBDA_INLINE
        {
            if constexpr(dstIsAoSoA)
                return &dstView.storageBlobs[0][0] + mapAoSoA(flatArrayIndex, rc, lanesDst);
            else
            {
                const auto [blob, off] = mapSoA(flatArrayIndex, rc, isDstMB);
                return &dstView.storageBlobs[blob][off];
            }
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
    template<typename SrcMapping, typename DstMapping, typename SFINAE = void>
    struct Copy
    {
        template<typename SrcView, typename DstView>
        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
        {
            fieldWiseCopy(srcView, dstView, threadId, threadCount);
        }
    };

    template<typename Mapping>
    struct Copy<Mapping, Mapping>
    {
        template<typename SrcView, typename DstView>
        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
        {
            blobMemcpy(srcView, dstView, threadId, threadCount);
        }
    };

    template<
        typename ArrayExtents,
        typename RecordDim,
        typename LinearizeArrayDims,
        std::size_t LanesSrc,
        std::size_t LanesDst>
    struct Copy<
        mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayDims>,
        mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayDims>,
        std::enable_if_t<LanesSrc != LanesDst>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayDims>, SrcBlob>& srcView,
            View<mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayDims>, DstBlob>& dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    template<
        typename ArrayExtents,
        typename RecordDim,
        typename LinearizeArrayDims,
        std::size_t LanesSrc,
        bool DstSeparateBuffers,
        bool DstAlignSubArrays>
    struct Copy<
        mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayDims>,
        mapping::SoA<ArrayExtents, RecordDim, DstSeparateBuffers, DstAlignSubArrays, LinearizeArrayDims>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayDims>, SrcBlob>& srcView,
            View<
                mapping::SoA<ArrayExtents, RecordDim, DstSeparateBuffers, DstAlignSubArrays, LinearizeArrayDims>,
                DstBlob>& dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    template<
        typename ArrayExtents,
        typename RecordDim,
        typename LinearizeArrayDims,
        std::size_t LanesDst,
        bool SrcSeparateBuffers,
        bool SrcAlignSubArrays>
    struct Copy<
        mapping::SoA<ArrayExtents, RecordDim, SrcSeparateBuffers, SrcAlignSubArrays, LinearizeArrayDims>,
        mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayDims>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<
                mapping::SoA<ArrayExtents, RecordDim, SrcSeparateBuffers, SrcAlignSubArrays, LinearizeArrayDims>,
                SrcBlob>& srcView,
            View<mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayDims>, DstBlob>& dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    /// Copy data from source view to destination view. Both views need to have the same array and record
    /// dimensions. Delegates to \ref Copy to choose an implementation.
    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
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
// == ./Copy.hpp ==
// ============================================================================

// ============================================================================
// == ./Proofs.hpp ==
// ==
// SPDX-License-Identifier: GPL-3.0-or-later

// #pragma once
// #include "ArrayIndexRange.hpp"    // amalgamate: file already expanded
// #include "Core.hpp"    // amalgamate: file already expanded

namespace llama
{
    namespace internal
    {
        constexpr auto divRoundUp(std::size_t dividend, std::size_t divisor) -> std::size_t
        {
            return (dividend + divisor - 1) / divisor;
        }
    } // namespace internal

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
    template<typename Mapping>
    constexpr auto mapsNonOverlappingly(const Mapping& m) -> bool
    {
        internal::DynArray<internal::DynArray<std::uint64_t>> blobByteMapped(m.blobCount);
        for(std::size_t i = 0; i < m.blobCount; i++)
            blobByteMapped.data[i].resize(internal::divRoundUp(m.blobSize(i), 64));

        auto testAndSet = [&](auto blob, auto offset) constexpr
        {
            const auto bit = std::uint64_t{1} << (offset % 64);
            if(blobByteMapped.data[blob].data[offset / 64] & bit)
                return true;
            blobByteMapped.data[blob].data[offset / 64] |= bit;
            return false;
        };

        bool collision = false;
        forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc) constexpr {
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
    template<std::size_t PieceLength, typename Mapping>
    constexpr auto mapsPiecewiseContiguous(const Mapping& m) -> bool
    {
        bool collision = false;
        forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc) constexpr {
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
// == ./Proofs.hpp ==
// ============================================================================

// ============================================================================
// == ./llama.hpp ==
// ==
// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

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
/// LLAMA is licensed under the LGPL3+.

// NOLINTNEXTLINE(modernize-macro-to-enum)
#define LLAMA_VERSION_MAJOR 0
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define LLAMA_VERSION_MINOR 4
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define LLAMA_VERSION_PATCH 0

// suppress warnings on missing return statements. we get a lot of these because nvcc/nvc++ have some troubles with if
// constexpr.
#ifdef __CUDACC__
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

// #include "ArrayExtents.hpp"    // amalgamate: file already expanded
// #include "ArrayIndexRange.hpp"    // amalgamate: file already expanded
// #include "BlobAllocators.hpp"    // amalgamate: file already expanded
// #include "Copy.hpp"    // amalgamate: file already expanded
// #include "Core.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./DumpMapping.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	#if __has_include(<fmt/format.h>)
	// #    include "ArrayIndexRange.hpp"    // amalgamate: file already expanded
	// #    include "Core.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./StructName.hpp ==
		// ==
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "Core.hpp"    // amalgamate: file already expanded

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

		                auto size1 = removeAllOccurences(nameArray, nameArray.size(), std::string_view{"struct "});
		                auto size2 = removeAllOccurences(nameArray, size1, std::string_view{"class "});
		#else
		                auto size2 = nameArray.size();
		#endif

		                auto size3Func = [&](auto& nameArray) constexpr
		                {
		                    // remove spaces between closing template angle brackets and after commas
		                    auto e = nameArray.begin() + size2;
		                    for(auto b = nameArray.begin(); b < e - 2; b++)
		                    {
		                        if((b[0] == '>' && b[1] == ' ' && b[2] == '>') || (b[0] == ',' && b[1] == ' '))
		                        {
		                            constexprCopy(b + 2, e, b + 1);
		                            e--;
		                        }
		                    }
		                    return e - nameArray.begin();
		                };
		                auto size3 = size3Func(nameArray);

		                return std::pair{nameArray, size3};
		            }
		            ();

		            Array<char, arrAndSize.second> a{};
		            constexprCopy(arrAndSize.first.begin(), arrAndSize.first.begin() + arrAndSize.second, a.begin());
		            return a;
		        }

		        template<typename T>
		        inline constexpr auto typeNameStorage = typeNameAsArray<T>();
		    } // namespace internal

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
		                auto e = s.end();
		                auto b = s.begin();
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
		            }
		            ();

		            Array<char, arrAndSize.second> a{};
		            constexprCopy(arrAndSize.first.begin(), arrAndSize.first.begin() + arrAndSize.second, a.begin());
		            return a;
		        }
		        ();
		    } // namespace internal

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
		            for(auto n = s; n != 0; n /= 10)
		                len++;
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
		                boost::mp11::mp_for_each<Tags>(
		                    [&](auto tag)
		                    {
		                        if(s != 0)
		                            s++; // for the '.'s
		                        using Tag = decltype(tag);
		                        if constexpr(isRecordCoord<Tag>)
		                            s += intToStrSize(s);
		                        else
		                            s += structName(tag).size();
		                    });
		                return s;
		            }
		            ();
		            llama::Array<char, size> a{};
		            for(auto& c : a)
		                c = '?';
		            auto w = a.begin();

		            boost::mp11::mp_for_each<Tags>([&](auto tag) constexpr {
		                if(w != a.begin())
		                {
		                    *w = '.';
		                    w++;
		                }
		                using Tag = decltype(tag);
		                if constexpr(isRecordCoord<Tag>)
		                {
		                    // handle array indices
		                    static_assert(Tag::size == 1);
		                    // convert to string
		                    auto n = Tag::front;
		                    w += intToStrSize(n) - 1;
		                    do
		                    {
		                        *w = '0' + n % 10;
		                        w--;
		                        n /= 10;
		                    } while(n != 0);
		                }
		                else
		                {
		                    constexpr auto sn = structName(tag);
		                    constexprCopy(sn.begin(), sn.end(), w);
		                    w += sn.size();
		                }
		            });
		            return a;
		        }
		        ();
		    } // namespace internal

		    /// Returns the tags interspersed by '.' represented by the given record coord in the given record dimension.
		    template<typename RecordDim, std::size_t... Coords>
		    constexpr auto recordCoordTags(RecordCoord<Coords...> = {}) -> std::string_view
		    {
		        constexpr auto& value = internal::recordCoordTagsStorage<RecordDim, Coords...>;
		        return std::string_view{&value[0], value.size()};
		    }

		    template<typename RecordDim>
		    constexpr auto recordCoordTags(RecordCoord<>) -> std::string_view
		    {
		        return {};
		    }
		} // namespace llama
		// ==
		// == ./StructName.hpp ==
		// ============================================================================

	// #    include "View.hpp"    // amalgamate: file already expanded

	#    include <boost/functional/hash.hpp>
	#    include <fmt/format.h>
	#    include <optional>
	// #    include <string>    // amalgamate: file already included
	// #    include <vector>    // amalgamate: file already included

	namespace llama
	{
	    namespace internal
	    {
	        template<typename Mapping>
	        constexpr auto hasAnyComputedField() -> bool
	        {
	            bool computed = false;
	            forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc)
	                                                          { computed |= llama::isComputed<Mapping, decltype(rc)>; });
	            return computed;
	        }

	        template<std::size_t... Coords>
	        auto toVec(RecordCoord<Coords...>) -> std::vector<std::size_t>
	        {
	            return {Coords...};
	        }

	        inline auto color(const std::vector<std::size_t>& recordCoord) -> std::size_t
	        {
	            auto c = boost::hash<std::vector<std::size_t>>{}(recordCoord) &std::size_t{0xFFFFFF};
	            c |= std::size_t{0x404040}; // ensure color per channel is at least 0x40.
	            return c;
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
	            std::vector<std::size_t> recordCoord;
	            std::string_view recordTags;
	            NrAndOffset<std::size_t> nrAndOffset;
	            std::size_t size;
	        };

	        template<typename View>
	        void fillBlobsWithPattern(View& view, uint8_t pattern)
	        {
	            const auto& mapping = view.mapping();
	            for(std::size_t i = 0; i < View::Mapping::blobCount; i++)
	                std::memset(&view.storageBlobs[i][0], pattern, mapping.blobSize(i));
	        }

	        template<typename View, typename RecordCoord>
	        void boxesFromComputedField(
	            View& view,
	            typename View::Mapping::ArrayIndex ai,
	            RecordCoord rc,
	            std::vector<FieldBox<typename View::Mapping::ArrayIndex>>& infos)
	        {
	            using Mapping = typename View::Mapping;
	            using RecordDim = typename Mapping::RecordDim;

	            auto emitInfo = [&](auto nrAndOffset, std::size_t size) {
	                infos.push_back({ai, internal::toVec(rc), recordCoordTags<RecordDim>(rc), nrAndOffset, size});
	            };

	            using Type = GetType<RecordDim, decltype(rc)>;
	            // computed values can come from anywhere, so we can only apply heuristics
	            auto& blobs = view.storageBlobs;
	            auto&& ref = view.mapping().compute(ai, rc, blobs);

	            // try to find the mapped address in one of the blobs
	            if constexpr(std::is_reference_v<decltype(ref)>)
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
	                const auto pattern = std::is_same_v<Type, bool> ? std::uint8_t{0xFF} : std::uint8_t{0xAA};
	                fillBlobsWithPattern(view, pattern);
	                ref = Type{}; // a broad range of types is default constructible and should write
	                              // zero bytes
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
	        auto boxesFromMapping(const Mapping& mapping) -> std::vector<FieldBox<typename Mapping::ArrayIndex>>
	        {
	            std::vector<FieldBox<typename Mapping::ArrayIndex>> infos;

	            std::optional<decltype(allocView(mapping))> view;
	            if constexpr(hasAnyComputedField<Mapping>())
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
	                                 internal::toVec(rc),
	                                 recordCoordTags<RecordDim>(rc),
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

	        constexpr auto hasAnyComputedField = internal::hasAnyComputedField<Mapping>();
	        std::array<int, Mapping::blobCount + hasAnyComputedField + 1> blobYOffset{};
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
	            const auto fill = internal::color(info.recordCoord);
	            const auto width = byteSizeInPixel * info.size;

	            const auto nextOffset = [&]
	            {
	                if(&info == &infos.back())
	                    return std::numeric_limits<std::size_t>::max();
	                const auto& nextInfo = (&info)[1];
	                if(nextInfo.nrAndOffset.nr < Mapping::blobCount)
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
	                info.recordTags);
	            if(cropBoxes)
	                svg += R"(</svg>
	)";
	        }

	        if(hasAnyComputedField)
	        {
	            writeBlobHeader(Mapping::blobCount, computedSizeSoFar, "Comp.");

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
	                    internal::cssClass(std::string{recordCoordTags<RecordDim>(rc)}),
	                    byteSizeInPixel * size,
	                    internal::color(internal::toVec(rc)));
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
	                internal::cssClass(std::string{info.recordTags}),
	                internal::formatArrayIndex(info.arrayIndex),
	                info.recordTags);
	        }
	        html += R"(</body>
	</html>)";
	        return html;
	    }
	} // namespace llama

	#endif
	// ==
	// == ./DumpMapping.hpp ==
	// ============================================================================

// #include "HasRanges.hpp"    // amalgamate: file already expanded
// #include "Meta.hpp"    // amalgamate: file already expanded
// #include "ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./RecordRef.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "Concepts.hpp"    // amalgamate: file already expanded
	// #include "HasRanges.hpp"    // amalgamate: file already expanded
	// #include "ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
	// #include "StructName.hpp"    // amalgamate: file already expanded
	// #include "View.hpp"    // amalgamate: file already expanded

	#include <iosfwd>
	// #include <type_traits>    // amalgamate: file already included

	namespace llama
	{
	    template<typename View, typename BoundRecordCoord, bool OwnView>
	    struct RecordRef;

	    template<typename View>
	    inline constexpr auto isRecordRef = false;

	    template<typename View, typename BoundRecordCoord, bool OwnView>
	    inline constexpr auto isRecordRef<RecordRef<View, BoundRecordCoord, OwnView>> = true;

	    /// Returns a \ref One with the same record dimension as the given record ref, with values copyied from rr.
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
	            if constexpr(std::is_same_v<
	                             typename LeftRecord::AccessibleRecordDim,
	                             typename RightRecord::AccessibleRecordDim>)
	            {
	                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE
	                                                                           { Functor{}(left(rc), right(rc)); });
	            }
	            else
	            {
	                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
	                    [&](auto leftRC) LLAMA_LAMBDA_INLINE
	                    {
	                        using LeftInnerCoord = decltype(leftRC);
	                        forEachLeafCoord<typename RightRecord::AccessibleRecordDim>(
	                            [&](auto rightRC) LLAMA_LAMBDA_INLINE
	                            {
	                                using RightInnerCoord = decltype(rightRC);
	                                if constexpr(hasSameTags<
	                                                 typename LeftRecord::AccessibleRecordDim,
	                                                 LeftInnerCoord,
	                                                 typename RightRecord::AccessibleRecordDim,
	                                                 RightInnerCoord>)
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
	            if constexpr(std::is_same_v<
	                             typename LeftRecord::AccessibleRecordDim,
	                             typename RightRecord::AccessibleRecordDim>)
	            {
	                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
	                    [&](auto rc) LLAMA_LAMBDA_INLINE { result &= Functor{}(left(rc), right(rc)); });
	            }
	            else
	            {
	                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
	                    [&](auto leftRC) LLAMA_LAMBDA_INLINE
	                    {
	                        using LeftInnerCoord = decltype(leftRC);
	                        forEachLeafCoord<typename RightRecord::AccessibleRecordDim>(
	                            [&](auto rightRC) LLAMA_LAMBDA_INLINE
	                            {
	                                using RightInnerCoord = decltype(rightRC);
	                                if constexpr(hasSameTags<
	                                                 typename LeftRecord::AccessibleRecordDim,
	                                                 LeftInnerCoord,
	                                                 typename RightRecord::AccessibleRecordDim,
	                                                 RightInnerCoord>)
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
	        inline constexpr auto
	            isTupleLike<T, std::void_t<decltype(get<0>(std::declval<T>())), std::tuple_size<T>>> = true;

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
	            isDirectListInitializableImpl<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...> = true;

	        template<typename T, typename... Args>
	        inline constexpr auto isDirectListInitializable = isDirectListInitializableImpl<T, void, Args...>;

	        template<typename T, typename Tuple>
	        inline constexpr auto isDirectListInitializableFromTuple = false;

	        template<typename T, template<typename...> typename Tuple, typename... Args>
	        inline constexpr auto
	            isDirectListInitializableFromTuple<T, Tuple<Args...>> = isDirectListInitializable<T, Args...>;

	        template<typename T, typename Simd, typename RecordCoord>
	        LLAMA_FN_HOST_ACC_INLINE void loadSimdRecord(const T& srcRef, Simd& dstSimd, RecordCoord rc);

	        template<typename Simd, typename T, typename RecordCoord>
	        LLAMA_FN_HOST_ACC_INLINE void storeSimdRecord(const Simd& srcSimd, T&& dstRef, RecordCoord rc);
	    } // namespace internal

	    /// Record reference type returned by \ref View after resolving an array dimensions coordinate or partially
	    /// resolving a \ref RecordCoord. A record reference does not hold data itself, it just binds enough information
	    /// (array dimensions coord and partial record coord) to retrieve it later from a \ref View. Records references
	    /// should not be created by the user. They are returned from various access functions in \ref View and RecordRef
	    /// itself.
	    template<typename TView, typename TBoundRecordCoord, bool OwnView>
	    struct RecordRef : private TView::Mapping::ArrayIndex
	    {
	        using View = TView; ///< View this record reference points into.
	        using BoundRecordCoord
	            = TBoundRecordCoord; ///< Record coords into View::RecordDim which are already bound by this RecordRef.

	    private:
	        using ArrayIndex = typename View::Mapping::ArrayIndex;
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
	            , view{allocViewStack<0, RecordDim>()}
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

	        // NOLINTNEXTLINE(cert-oop54-cpp)
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
	            if constexpr(isRecord<AccessedType> || internal::IsBoundedArray<AccessedType>::value)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return RecordRef<const View, AbsolutCoord>{arrayIndex(), this->view};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return this->view.access(arrayIndex(), AbsolutCoord{});
	            }
	        }

	        // FIXME(bgruber): remove redundancy
	        template<std::size_t... Coord>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>) -> decltype(auto)
	        {
	            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
	            using AccessedType = GetType<RecordDim, AbsolutCoord>;
	            if constexpr(isRecord<AccessedType> || internal::IsBoundedArray<AccessedType>::value)
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return RecordRef<View, AbsolutCoord>{arrayIndex(), this->view};
	            }
	            else
	            {
	                LLAMA_FORCE_INLINE_RECURSIVE
	                return this->view.access(arrayIndex(), AbsolutCoord{});
	            }
	        }

	        /// Access a record in the record dimension underneath the current record reference using a series of tags. If
	        /// the access resolves to a leaf, an l-value reference to a variable inside the \ref View storage is returned,
	        /// otherwise another RecordRef.
	        template<typename... Tags>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
	        {
	            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, Tags...>;

	            LLAMA_FORCE_INLINE_RECURSIVE
	            return operator()(RecordCoord{});
	        }

	        // FIXME(bgruber): remove redundancy
	        template<typename... Tags>
	        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
	        {
	            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, Tags...>;

	            LLAMA_FORCE_INLINE_RECURSIVE
	            return operator()(RecordCoord{});
	        }

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
	            return internal::recordRefRelOperator<std::not_equal_to<>>(vd, t);
	        }

	        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
	        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const RecordRef& vd) -> bool
	        {
	            return vd != t;
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
	                    swap(a(rc), b(rc));
	                });
	        }

	        // FIXME(bgruber): the SIMD load/store functions need to navigate back from a record ref to the contained view
	        // to find subsequent elements. This is not a great design for now and the SIMD load/store functions should
	        // probably take iterators to records.
	        template<typename T, typename Simd, typename RecordCoord>
	        friend void internal::loadSimdRecord(const T& srcRef, Simd& dstSimd, RecordCoord rc);
	        template<typename Simd, typename T, typename RecordCoord>
	        friend void internal::storeSimdRecord(const Simd& srcSimd, T&& dstRef, RecordCoord rc);
	    };

	    // swap for heterogeneous RecordRef
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

	    template<typename View, typename BoundRecordCoord, bool OwnView>
	    auto operator<<(std::ostream& os, const RecordRef<View, BoundRecordCoord, OwnView>& vr) -> std::ostream&
	    {
	        using RecordDim = typename RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim;
	        os << "{";
	        if constexpr(std::is_array_v<RecordDim>)
	        {
	            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<std::extent_v<RecordDim>>>(
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
	            boost::mp11::mp_for_each<boost::mp11::mp_iota<boost::mp11::mp_size<RecordDim>>>(
	                [&](auto ic)
	                {
	                    constexpr std::size_t i = decltype(ic)::value;
	                    if(i > 0)
	                        os << ", ";
	                    using Field = boost::mp11::mp_at_c<RecordDim, i>;
	                    os << structName<GetFieldTag<Field>>() << ": " << vr(RecordCoord<i>{});
	                });
	        }
	        os << "}";
	        return os;
	    }

	    template<typename RecordRefFwd, typename Functor>
	    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaf(RecordRefFwd&& vr, Functor&& functor)
	    {
	        using RecordRef = std::remove_reference_t<RecordRefFwd>;
	        LLAMA_FORCE_INLINE_RECURSIVE
	        forEachLeafCoord<typename RecordRef::AccessibleRecordDim>(
	            [functor = std::forward<Functor>(functor), &vr = vr](auto rc)
	                LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr mutable) { std::forward<Functor>(functor)(vr(rc)); });
	    }

	    namespace internal
	    {
	        // gets the value type for a given T, where T models a reference type. T is either an l-value reference, a
	        // proxy reference or a RecordRef
	        template<typename T, typename = void>
	        struct ValueOf
	        {
	            static_assert(sizeof(T) == 0, "T does not model a reference");
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
	            using type = T;
	        };
	    } // namespace internal

	    /// Scope guard type. ScopedUpdate takes a copy of a value through a reference and stores it internally during
	    /// construction. The stored value is written back when ScopedUpdate is destroyed. ScopedUpdate tries to act like
	    /// the stored value as much as possible, exposing member functions of the stored value and acting like a proxy
	    /// reference if the stored value is a primitive type.
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

	    template<typename T>
	    ScopedUpdate(T) -> ScopedUpdate<typename internal::ReferenceTo<std::remove_reference_t<T>>::type>;
	} // namespace llama

	template<typename View, typename BoundRecordCoord, bool OwnView>
	struct std::tuple_size<llama::RecordRef<View, BoundRecordCoord, OwnView>> // NOLINT(cert-dcl58-cpp)
	    : boost::mp11::mp_size<typename llama::RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim>
	{
	};

	template<std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
	struct std::tuple_element<I, llama::RecordRef<View, BoundRecordCoord, OwnView>> // NOLINT(cert-dcl58-cpp)
	{
	    using type = decltype(std::declval<llama::RecordRef<View, BoundRecordCoord, OwnView>>().template get<I>());
	};

	template<std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
	struct std::tuple_element<I, const llama::RecordRef<View, BoundRecordCoord, OwnView>> // NOLINT(cert-dcl58-cpp)
	{
	    using type = decltype(std::declval<const llama::RecordRef<View, BoundRecordCoord, OwnView>>().template get<I>());
	};

	#if CAN_USE_RANGES
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
	// == ./RecordRef.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./Simd.hpp ==
	// ==
	// #pragma once
	// #include "Core.hpp"    // amalgamate: file already expanded
	// #include "RecordRef.hpp"    // amalgamate: file already expanded
	// #include "macros.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/AoS.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "Common.hpp"    // amalgamate: file already expanded

		namespace llama::mapping
		{
		    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
		    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
		    /// If false, struct members are tightly packed.
		    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
		    /// how big the linear domain gets.
		    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
		    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
		    /// \ref FlattenRecordDimMinimizePadding.
		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        bool AlignAndPad = true,
		        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
		        template<typename> typename FlattenRecordDim = FlattenRecordDimInOrder>
		    struct AoS : MappingBase<TArrayExtents, TRecordDim>
		    {
		    private:
		        using Base = MappingBase<TArrayExtents, TRecordDim>;
		        using size_type = typename Base::size_type;

		    public:
		        inline static constexpr bool alignAndPad = AlignAndPad;
		        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
		        using Flattener = FlattenRecordDim<TRecordDim>;
		        inline static constexpr std::size_t blobCount = 1;

		        using Base::Base;

		        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
		        {
		            return LinearizeArrayDimsFunctor{}.size(Base::extents())
		                * flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad>;
		        }

		        template<std::size_t... RecordCoords>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
		            typename Base::ArrayIndex ai,
		            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
		        {
		            constexpr std::size_t flatFieldIndex =
		#ifdef __NVCC__
		                *& // mess with nvcc compiler state to workaround bug
		#endif
		                 Flattener::template flatIndex<RecordCoords...>;
		            const auto offset = LinearizeArrayDimsFunctor{}(ai, Base::extents())
		                    * static_cast<size_type>(flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad>)
		                + static_cast<size_type>(flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, AlignAndPad>);
		            return {size_type{0}, offset};
		        }
		    };

		    // we can drop this when inherited ctors also inherit deduction guides
		    template<typename TArrayExtents, typename TRecordDim>
		    AoS(TArrayExtents, TRecordDim) -> AoS<TArrayExtents, TRecordDim>;

		    /// Array of struct mapping preserving the alignment of the field types by inserting padding.
		    /// \see AoS
		    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
		    using AlignedAoS = AoS<ArrayExtents, RecordDim, true, LinearizeArrayDimsFunctor>;

		    /// Array of struct mapping preserving the alignment of the field types by inserting padding and permuting the
		    /// field order to minimize this padding. \see AoS
		    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
		    using MinAlignedAoS
		        = AoS<ArrayExtents, RecordDim, true, LinearizeArrayDimsFunctor, FlattenRecordDimMinimizePadding>;

		    /// Array of struct mapping packing the field types tightly, violating the type's alignment requirements.
		    /// \see AoS
		    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
		    using PackedAoS = AoS<ArrayExtents, RecordDim, false, LinearizeArrayDimsFunctor>;

		    /// Binds parameters to an \ref AoS mapping except for array and record dimension, producing a quoted meta
		    /// function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
		    template<bool AlignAndPad = true, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
		    struct BindAoS
		    {
		        template<typename ArrayExtents, typename RecordDim>
		        using fn = AoS<ArrayExtents, RecordDim, AlignAndPad, LinearizeArrayDimsFunctor>;
		    };

		    template<typename Mapping>
		    inline constexpr bool isAoS = false;

		    template<
		        typename ArrayExtents,
		        typename RecordDim,
		        bool AlignAndPad,
		        typename LinearizeArrayDimsFunctor,
		        template<typename>
		        typename FlattenRecordDim>
		    inline constexpr bool
		        isAoS<AoS<ArrayExtents, RecordDim, AlignAndPad, LinearizeArrayDimsFunctor, FlattenRecordDim>> = true;
		} // namespace llama::mapping
		// ==
		// == ./mapping/AoS.hpp ==
		// ============================================================================

	// #include "mapping/SoA.hpp"    // amalgamate: file already expanded

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
	    template<typename Simd, typename SFINAE = void>
	    struct SimdTraits
	    {
	        static_assert(sizeof(Simd) == 0, "Please specialize SimdTraits for the type Simd");
	    };

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
	    };

	    /// The number of SIMD simdLanes the given SIMD vector or \ref Simd<T> has. If Simd is not a structural \ref Simd
	    /// or \ref SimdN, this is a shortcut for SimdTraits<Simd>::lanes.
	    template<typename Simd, typename SFINAE = void>
	    inline constexpr auto simdLanes = SimdTraits<Simd>::lanes;

	    /// Chooses the number of SIMD lanes for the given record dimension by mapping each field type to a SIMD type and
	    /// then reducing their sizes.
	    /// @tparam MakeSimd Type function creating a SIMD type given a field type from the record dimension.
	    /// @param reduce Binary reduction function to reduce the SIMD lanes.
	    template<typename RecordDim, template<typename> typename MakeSimd, typename BinaryReductionFunction>
	    constexpr auto chooseSimdLanes(BinaryReductionFunction reduce) -> std::size_t
	    {
	        using FRD = FlatRecordDim<RecordDim>;
	        std::size_t lanes = simdLanes<MakeSimd<boost::mp11::mp_first<FRD>>>;
	        boost::mp11::mp_for_each<boost::mp11::mp_transform<std::add_pointer_t, boost::mp11::mp_drop_c<FRD, 1>>>(
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
	    template<typename RecordDim, template<typename> typename MakeSimd>
	    inline constexpr std::size_t simdLanesWithFullVectorsFor
	        = chooseSimdLanes<RecordDim, MakeSimd>([](auto a, auto b) { return std::max(a, b); });

	    /// Determines the number of simd lanes suitable to process all types occurring in the given record dimension. The
	    /// algorithm ensures that the smallest number of SIMD registers is needed and may thus only partially fill
	    /// registers for some data types.
	    /// @tparam RecordDim The record dimension to simdize
	    /// @tparam MakeSimd Type function creating a SIMD type given a field type from the record dimension.
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
	    template<typename RecordDim, std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
	    using SimdizeN = typename internal::SimdizeNImpl<RecordDim, N, MakeSizedSimd>::type;

	    /// Transforms the given record dimension into a SIMD version of it. Each leaf field type will be replaced by a
	    /// SIMD vector, as determined by MakeSimd.
	    template<typename RecordDim, template<typename> typename MakeSimd>
	    using Simdize = TransformLeaves<RecordDim, MakeSimd>;

	    /// Creates a SIMD version of the given type. Of T is a record dimension, creates a \ref One where each field is a
	    /// SIMD type of the original field type. The SIMD vectors have length N. If N is 1, an ordinary \ref One of the
	    /// record dimension T is created. If T is not a record dimension, a SIMD vector with value T and length N is
	    /// created. If N is 1 (and T is not a record dimension), then T is produced.
	    template<typename T, std::size_t N, template<typename, /* std::integral */ auto> typename MakeSizedSimd>
	    using SimdN = typename std::conditional_t<
	        isRecord<T> || internal::IsBoundedArray<T>::value,
	        std::conditional_t<
	            N == 1,
	            boost::mp11::mp_identity<One<T>>,
	            boost::mp11::mp_identity<One<SimdizeN<T, N, MakeSizedSimd>>>>,
	        std::conditional_t<
	            N == 1,
	            boost::mp11::mp_identity<T>,
	            boost::mp11::mp_identity<SimdizeN<T, N, MakeSizedSimd>>>>::type;

	    /// Creates a SIMD version of the given type. Of T is a record dimension, creates a \ref One where each field is a
	    /// SIMD type of the original field type.
	    template<typename T, template<typename> typename MakeSimd>
	    using Simd = typename std::conditional_t<
	        isRecord<T> || internal::IsBoundedArray<T>::value,
	        boost::mp11::mp_identity<One<Simdize<T, MakeSimd>>>,
	        boost::mp11::mp_identity<Simdize<T, MakeSimd>>>::type;

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
	    template<typename Simd>
	    inline constexpr std::size_t simdLanes<Simd, std::enable_if_t<isRecordRef<Simd>>> = []
	    {
	        using FRD = FlatRecordDim<typename Simd::AccessibleRecordDim>;
	        using FirstFieldType = boost::mp11::mp_first<FRD>;
	        static_assert(boost::mp11::mp_all_of_q<FRD, internal::SizeEqualTo<simdLanes<FirstFieldType>>>::value);
	        return simdLanes<FirstFieldType>;
	    }();

	    namespace internal
	    {
	        template<typename T, typename Simd, typename RecordCoord>
	        LLAMA_FN_HOST_ACC_INLINE void loadSimdRecord(const T& srcRef, Simd& dstSimd, RecordCoord rc)
	        {
	            using RecordDim = typename T::AccessibleRecordDim;
	            using FieldType = GetType<RecordDim, decltype(rc)>;
	            using ElementSimd = std::decay_t<decltype(dstSimd(rc))>;
	            using Traits = SimdTraits<ElementSimd>;

	            // TODO(bgruber): can we generalize the logic whether we can load a dstSimd from that mapping?
	            using Mapping = typename T::View::Mapping;
	            if constexpr(mapping::isSoA<Mapping>)
	            {
	                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	                dstSimd(rc) = Traits::loadUnaligned(&srcRef(rc)); // SIMD load
	                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	            }
	            // else if constexpr(mapping::isAoSoA<typename T::View::Mapping>)
	            //{
	            //    // it turns out we do not need the specialization, because clang already fuses the scalar
	            //    loads
	            //    // into a vector load :D
	            //    assert(srcRef.arrayDimsCoord()[0] % SIMD_WIDTH == 0);
	            //    // if(srcRef.arrayDimsCoord()[0] % SIMD_WIDTH != 0)
	            //    //    __builtin_unreachable(); // this also helps nothing
	            //    //__builtin_assume(srcRef.arrayDimsCoord()[0] % SIMD_WIDTH == 0);  // this also helps nothing
	            //    dstSimd(rc) = Traits::load_from(&srcRef(rc)); // SIMD load
	            //}
	            else if constexpr(mapping::isAoS<Mapping>)
	            {
	                static_assert(mapping::isAoS<Mapping>);
	                static constexpr auto srcStride
	                    = flatSizeOf<typename Mapping::Flattener::FlatRecordDim, Mapping::alignAndPad>;
	                const auto* srcBaseAddr = reinterpret_cast<const std::byte*>(&srcRef(rc));
	                ElementSimd elemSimd; // g++-12 really needs the intermediate elemSimd and memcpy
	                for(auto i = 0; i < Traits::lanes; i++)
	                    reinterpret_cast<FieldType*>(&elemSimd)[i]
	                        = *reinterpret_cast<const FieldType*>(srcBaseAddr + i * srcStride);
	                std::memcpy(&dstSimd(rc), &elemSimd, sizeof(elemSimd));
	            }
	            else
	            {
	                auto b = ArrayIndexIterator{srcRef.view.mapping().extents(), srcRef.arrayIndex()};
	                ElementSimd elemSimd; // g++-12 really needs the intermediate elemSimd and memcpy
	                for(auto i = 0; i < Traits::lanes; i++)
	                    reinterpret_cast<FieldType*>(&elemSimd)[i]
	                        = srcRef.view(*b++)(cat(typename T::BoundRecordCoord{}, rc)); // scalar loads
	                std::memcpy(&dstSimd(rc), &elemSimd, sizeof(elemSimd));
	            }
	        }

	        template<typename Simd, typename TFwd, typename RecordCoord>
	        LLAMA_FN_HOST_ACC_INLINE void storeSimdRecord(const Simd& srcSimd, TFwd&& dstRef, RecordCoord rc)
	        {
	            using T = std::remove_reference_t<TFwd>;
	            using RecordDim = typename T::AccessibleRecordDim;
	            using FieldType = GetType<RecordDim, decltype(rc)>;
	            using ElementSimd = std::decay_t<decltype(srcSimd(rc))>;
	            using Traits = SimdTraits<ElementSimd>;

	            // TODO(bgruber): can we generalize the logic whether we can store a srcSimd to that mapping?
	            using Mapping = typename std::remove_reference_t<T>::View::Mapping;
	            if constexpr(mapping::isSoA<Mapping>)
	            {
	                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	                Traits::storeUnaligned(srcSimd(rc), &dstRef(rc)); // SIMD store
	                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	            }
	            else if constexpr(mapping::isAoS<Mapping>)
	            {
	                static constexpr auto stride
	                    = flatSizeOf<typename Mapping::Flattener::FlatRecordDim, Mapping::alignAndPad>;
	                auto* dstBaseAddr = reinterpret_cast<std::byte*>(&dstRef(rc));
	                const ElementSimd elemSimd = srcSimd(rc);
	                for(auto i = 0; i < Traits::lanes; i++)
	                    *reinterpret_cast<FieldType*>(dstBaseAddr + i * stride)
	                        = reinterpret_cast<const FieldType*>(&elemSimd)[i];
	            }
	            else
	            {
	                // TODO(bgruber): how does this generalize conceptually to 2D and higher dimensions? in which
	                // direction should we collect SIMD values?
	                const ElementSimd elemSimd = srcSimd(rc);
	                auto b = ArrayIndexIterator{dstRef.view.mapping().extents(), dstRef.arrayIndex()};
	                for(auto i = 0; i < Traits::lanes; i++)
	                    dstRef.view (*b++)(cat(typename T::BoundRecordCoord{}, rc))
	                        = reinterpret_cast<const FieldType*>(&elemSimd)[i]; // scalar store
	            }
	        }
	    } // namespace internal

	    /// Loads SIMD vectors of data starting from the given record reference to dstSimd. Only field tags occurring in
	    /// RecordRef are loaded. If Simd contains multiple fields of SIMD types, a SIMD vector will be fetched for each of
	    /// the fields. The number of elements fetched per SIMD vector depends on the SIMD width of the vector. Simd is
	    /// allowed to have different vector lengths per element.
	    template<typename T, typename Simd>
	    LLAMA_FN_HOST_ACC_INLINE void loadSimd(const T& srcRef, Simd& dstSimd)
	    {
	        // structured dstSimd type and record reference
	        if constexpr(isRecordRef<Simd> && isRecordRef<T>)
	        {
	            forEachLeafCoord<typename T::AccessibleRecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE
	                                                              { internal::loadSimdRecord(srcRef, dstSimd, rc); });
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
	    template<typename Simd, typename T>
	    LLAMA_FN_HOST_ACC_INLINE void storeSimd(const Simd& srcSimd, T&& dstRef)
	    {
	        // structured Simd type and record reference
	        if constexpr(isRecordRef<Simd> && isRecordRef<T>)
	        {
	            forEachLeafCoord<typename T::AccessibleRecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE
	                                                              { internal::storeSimdRecord(srcSimd, dstRef, rc); });
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
	// == ./Simd.hpp ==
	// ============================================================================

// #include "StructName.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./Vector.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "RecordRef.hpp"    // amalgamate: file already expanded
	// #include "View.hpp"    // amalgamate: file already expanded

	// #include <algorithm>    // amalgamate: file already included
	#include <stdexcept>
	// #include <string>    // amalgamate: file already included

	namespace llama
	{
	    // TODO(bgruber): expose blob allocator
	    /// An equivalent of std::vector<T> backed by a \ref View. Elements are never value initialized though. No strong
	    /// exception guarantee.
	    /// WARNING: This class is experimental.
	    /// @tparam Mapping The mapping to be used for the underlying view. Needs to have 1 array dimension.
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

	        LLAMA_FN_HOST_ACC_INLINE auto at(size_type i) -> decltype(auto)
	        {
	            if(i >= m_size)
	                throw std::out_of_range{
	                    "Index " + std::to_string(i) + "out of range [0:" + std::to_string(m_size) + "["};
	            return m_view(i);
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto at(size_type i) const -> decltype(auto)
	        {
	            if(i >= m_size)
	                throw std::out_of_range{
	                    "Index " + std::to_string(i) + "out of range [0:" + std::to_string(m_size) + "["};
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
	            return m_view.mapping().extents()[0];
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
	// == ./Vector.hpp ==
	// ============================================================================

// #include "View.hpp"    // amalgamate: file already expanded
// #include "macros.hpp"    // amalgamate: file already expanded
// #include "mapping/AoS.hpp"    // amalgamate: file already expanded
// #include "mapping/AoSoA.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./mapping/BitPackedFloatSoA.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/BitPackedIntSoA.hpp ==
		// ==
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "../Core.hpp"    // amalgamate: file already expanded
		// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
		// #include "Common.hpp"    // amalgamate: file already expanded

		// #include <climits>    // amalgamate: file already included
		// #include <type_traits>    // amalgamate: file already included

		namespace llama::mapping
		{
		    namespace internal
		    {
		        /// A proxy type representing a reference to a reduced precision integral value, stored in a buffer at a
		        /// specified bit offset.
		        /// @tparam Integral Integral data type which can be loaded and store through this reference.
		        /// @tparam StoredIntegralPointer Pointer to integral type used for storing the bits.
		        template<typename Integral, typename StoredIntegralPointer, typename VHBits, typename SizeType>
		        struct BitPackedIntRef
		            : private VHBits
		            , ProxyRefOpMixin<BitPackedIntRef<Integral, StoredIntegralPointer, VHBits, SizeType>, Integral>
		        {
		        private:
		            using StoredIntegral = std::remove_const_t<std::remove_pointer_t<StoredIntegralPointer>>;

		            static_assert(std::is_integral_v<StoredIntegral>);
		            static_assert(std::is_unsigned_v<StoredIntegral>);
		            static_assert(
		                sizeof(StoredIntegral) >= sizeof(Integral),
		                "The integral type used for the storage must be at least as big as the type of the values to "
		                "retrieve");

		            StoredIntegralPointer ptr;
		            SizeType bitOffset;
		#ifndef NDEBUG
		            StoredIntegralPointer endPtr;
		#endif

		            // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
		            static constexpr auto bitsPerStoredIntegral = static_cast<SizeType>(sizeof(StoredIntegral) * CHAR_BIT);

		        public:
		            using value_type = Integral;

		            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedIntRef(
		                StoredIntegralPointer ptr,
		                SizeType bitOffset,
		                VHBits vhBits
		#ifndef NDEBUG
		                ,
		                StoredIntegralPointer endPtr
		#endif
		                )
		                : VHBits{vhBits}
		                , ptr{ptr}
		                , bitOffset{bitOffset}

		#ifndef NDEBUG
		                , endPtr{endPtr}
		#endif
		            {
		            }

		            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
		            LLAMA_FN_HOST_ACC_INLINE constexpr operator Integral() const
		            {
		                auto* p = ptr + bitOffset / bitsPerStoredIntegral;
		                const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;
		                assert(p < endPtr);
		                auto v = p[0] >> innerBitOffset;

		                const auto innerBitEndOffset = innerBitOffset + VHBits::value();
		                if(innerBitEndOffset <= bitsPerStoredIntegral)
		                {
		                    const auto mask = (StoredIntegral{1} << VHBits::value()) - 1u;
		                    v &= mask;
		                }
		                else
		                {
		                    const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
		                    const auto bitsLoaded = bitsPerStoredIntegral - innerBitOffset;
		                    const auto mask = (StoredIntegral{1} << excessBits) - 1u;
		                    assert(p + 1 < endPtr);
		                    v |= (p[1] & mask) << bitsLoaded;
		                }
		                if constexpr(std::is_signed_v<Integral>)
		                {
		                    if(v & (StoredIntegral{1} << (VHBits::value() - 1)))
		                        v |= ~StoredIntegral{0} << VHBits::value(); // sign extend
		                }
		                return static_cast<Integral>(v);
		            }

		            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Integral value) -> BitPackedIntRef&
		            {
		                // NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
		                const auto unsignedValue = static_cast<StoredIntegral>(value);
		                const auto mask = (StoredIntegral{1} << VHBits::value()) - 1u;
		                StoredIntegral valueBits;
		                if constexpr(!std::is_signed_v<Integral>)
		                    valueBits = unsignedValue & mask;
		                else
		                {
		                    const auto magnitudeMask = (StoredIntegral{1} << (VHBits::value() - 1)) - 1u;
		                    const auto isSigned = value < 0;
		                    valueBits = (StoredIntegral{isSigned} << (VHBits::value() - 1)) | (unsignedValue & magnitudeMask);
		                }

		                auto* p = ptr + bitOffset / bitsPerStoredIntegral;
		                const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;
		                const auto clearMask = ~(mask << innerBitOffset);
		                assert(p < endPtr);
		                auto mem = p[0] & clearMask; // clear previous bits
		                mem |= valueBits << innerBitOffset; // write new bits
		                p[0] = mem;

		                const auto innerBitEndOffset = innerBitOffset + VHBits::value();
		                if(innerBitEndOffset > bitsPerStoredIntegral)
		                {
		                    const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
		                    const auto bitsWritten = bitsPerStoredIntegral - innerBitOffset;
		                    const auto clearMask = ~((StoredIntegral{1} << excessBits) - 1u);
		                    assert(p + 1 < endPtr);
		                    auto mem = p[1] & clearMask; // clear previous bits
		                    mem |= valueBits >> bitsWritten; // write new bits
		                    p[1] = mem;
		                }

		                return *this;
		            }
		        };

		        template<typename A, typename B>
		        using HasLargerSize = boost::mp11::mp_bool<sizeof(A) < sizeof(B)>;

		        template<typename RecordDim>
		        using LargestIntegral = boost::mp11::mp_max_element<FlatRecordDim<RecordDim>, HasLargerSize>;

		        template<typename T, typename SFINAE = void>
		        struct MakeUnsigned : std::make_unsigned<T>
		        {
		        };

		        template<>
		        struct MakeUnsigned<bool>
		        {
		            using type = std::uint8_t;
		        };

		        template<typename T>
		        struct MakeUnsigned<T, std::enable_if_t<std::is_enum_v<T>>> : std::make_unsigned<std::underlying_type_t<T>>
		        {
		        };

		        template<typename RecordDim>
		        using StoredUnsignedFor = typename MakeUnsigned<LargestIntegral<RecordDim>>::type;
		    } // namespace internal

		    /// Struct of array mapping using bit packing to reduce size/precision of integral data types. If your record
		    /// dimension contains non-integral types, split them off using the \ref Split mapping first.
		    /// \tparam Bits If Bits is llama::Constant<N>, the compile-time N specifies the number of bits to use. If Bits is
		    /// an integral type T, the number of bits is specified at runtime, passed to the constructor and stored as type T.
		    /// Must not be zero.
		    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
		    /// how big the linear domain gets. \tparam TStoredIntegral Integral type used as storage of reduced precision
		    /// integers.
		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        typename Bits = typename TArrayExtents::value_type,
		        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
		        typename TStoredIntegral = internal::StoredUnsignedFor<TRecordDim>>
		    struct BitPackedIntSoA
		        : MappingBase<TArrayExtents, TRecordDim>
		        , private llama::internal::BoxedValue<Bits>
		    {
		    private:
		        using Base = MappingBase<TArrayExtents, TRecordDim>;
		        using VHBits = llama::internal::BoxedValue<Bits>;
		        using size_type = typename TArrayExtents::value_type;

		        template<typename T>
		        using IsAllowedFieldType = boost::mp11::mp_or<std::is_integral<T>, std::is_enum<T>>;

		        static_assert(
		            boost::mp11::mp_all_of<FlatRecordDim<TRecordDim>, IsAllowedFieldType>::value,
		            "All record dimension field types must be integral");

		    public:
		        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
		        using StoredIntegral = TStoredIntegral;
		        static constexpr std::size_t blobCount = boost::mp11::mp_size<FlatRecordDim<TRecordDim>>::value;

		        LLAMA_FN_HOST_ACC_INLINE
		        constexpr auto bits() const -> size_type
		        {
		            return static_cast<size_type>(VHBits::value());
		        }

		        template<typename B = Bits, std::enable_if_t<isConstant<B>, int> = 0>
		        LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntSoA(
		            TArrayExtents extents = {},
		            Bits bits = {},
		            TRecordDim = {})
		            : Base(extents)
		            , VHBits{bits}
		        {
		            static_assert(VHBits::value() > 0);
		        }

		        template<typename B = Bits, std::enable_if_t<!isConstant<B>, int> = 0>
		        LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntSoA(TArrayExtents extents, Bits bits, TRecordDim = {})
		            : Base(extents)
		            , VHBits{bits}
		        {
		            assert(this->bits() > 0);
		        }

		        LLAMA_FN_HOST_ACC_INLINE
		        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
		        {
		            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(StoredIntegral) * CHAR_BIT);
		            const auto bitsNeeded = LinearizeArrayDimsFunctor{}.size(Base::extents()) * VHBits::value();
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
		            constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
		            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, Base::extents()) * VHBits::value();

		            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
		            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
		            return internal::BitPackedIntRef<DstType, QualifiedStoredIntegral*, VHBits, size_type>{
		                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
		                bitOffset,
		                static_cast<const VHBits&>(*this)
		#ifndef NDEBUG
		                    ,
		                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0] + blobSize(blob))
		#endif
		            };
		        }
		    };

		    /// Binds parameters to a \ref BitPackedIntSoA mapping except for array and record dimension, producing a quoted
		    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
		    template<
		        typename Bits = unsigned,
		        typename LinearizeArrayDimsFunctor = mapping::LinearizeArrayDimsCpp,
		        typename StoredIntegral = void>
		    struct BindBitPackedIntSoA
		    {
		        template<typename ArrayExtents, typename RecordDim>
		        using fn = BitPackedIntSoA<
		            ArrayExtents,
		            RecordDim,
		            Bits,
		            LinearizeArrayDimsFunctor,
		            std::conditional_t<
		                !std::is_void_v<StoredIntegral>,
		                StoredIntegral,
		                internal::StoredUnsignedFor<RecordDim>>>;
		    };

		    template<typename Mapping>
		    inline constexpr bool isBitPackedIntSoA = false;

		    template<typename... Ts>
		    inline constexpr bool isBitPackedIntSoA<BitPackedIntSoA<Ts...>> = true;
		} // namespace llama::mapping
		// ==
		// == ./mapping/BitPackedIntSoA.hpp ==
		// ============================================================================

	// #include "Common.hpp"    // amalgamate: file already expanded

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
	        /// @tparam StoredIntegralPointer Pointer to integral type used for storing the bits.
	        template<typename Float, typename StoredIntegralPointer, typename VHExp, typename VHMan, typename SizeType>
	        struct LLAMA_DECLSPEC_EMPTY_BASES BitPackedFloatRef
	            : private VHExp
	            , private VHMan
	            , ProxyRefOpMixin<BitPackedFloatRef<Float, StoredIntegralPointer, VHExp, VHMan, SizeType>, Float>
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
	                StoredIntegralPointer,
	                decltype(integBits(std::declval<VHExp>(), std::declval<VHMan>())),
	                SizeType>
	                intref;

	        public:
	            using value_type = Float;

	            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedFloatRef(
	                StoredIntegralPointer p,
	                SizeType bitOffset,
	                VHExp vhExp,
	                VHMan vhMan
	#ifndef NDEBUG
	                ,
	                StoredIntegralPointer endPtr
	#endif
	                )
	                : VHExp{vhExp}
	                , VHMan{vhMan}
	                , intref{
	                      p,
	                      bitOffset,
	                      integBits(vhExp, vhMan),
	#ifndef NDEBUG
	                      endPtr
	#endif
	                  }
	            {
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
	        using StoredIntegralFor = std::conditional_t<
	            boost::mp11::mp_contains<FlatRecordDim<RecordDim>, double>::value,
	            std::uint64_t,
	            std::uint32_t>;
	    } // namespace internal

	    /// Struct of array mapping using bit packing to reduce size/precision of floating-point data types. The bit layout
	    /// is [1 sign bit, exponentBits bits from the exponent, mantissaBits bits from the mantissa]+ and tries to follow
	    /// IEEE 754. Infinity and NAN are supported. If the packed exponent bits are not big enough to hold a number, it
	    /// will be set to infinity (preserving the sign). If your record dimension contains non-floating-point types,
	    /// split them off using the \ref Split mapping first.
	    /// \tparam ExponentBits If ExponentBits is llama::Constant<N>, the compile-time N specifies the number of bits to
	    /// use to store the exponent. If ExponentBits is llama::Value<T>, the number of bits is specified at runtime,
	    /// passed to the constructor and stored as type T. Must not be zero.
	    /// \tparam MantissaBits Like ExponentBits but for the mantissa bits. May be zero.
	    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
	    /// how big the linear domain gets.
	    /// \tparam TStoredIntegral Integral type used as storage of reduced precision floating-point values.
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        typename ExponentBits = typename TArrayExtents::value_type,
	        typename MantissaBits = ExponentBits,
	        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
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
	        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
	        using StoredIntegral = TStoredIntegral;
	        static constexpr std::size_t blobCount = boost::mp11::mp_size<FlatRecordDim<TRecordDim>>::value;

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
	                = LinearizeArrayDimsFunctor{}.size(Base::extents()) * (exponentBits() + mantissaBits() + 1);
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
	                = LinearizeArrayDimsFunctor{}(ai, Base::extents()) * (exponentBits() + mantissaBits() + 1);

	            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
	            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
	            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
	            return internal::BitPackedFloatRef<DstType, QualifiedStoredIntegral*, VHExp, VHMan, size_type>{
	                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
	                bitOffset,
	                static_cast<const VHExp&>(*this),
	                static_cast<const VHMan&>(*this)
	#ifndef NDEBUG
	                    ,
	                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0] + blobSize(blob))
	#endif
	            };
	            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
	        }
	    };

	    /// Binds parameters to a \ref BitPackedFloatSoA mapping except for array and record dimension, producing a quoted
	    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
	    template<
	        typename ExponentBits = unsigned,
	        typename MantissaBits = ExponentBits,
	        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
	        typename StoredIntegral = void>
	    struct BindBitPackedFloatSoA
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = BitPackedFloatSoA<
	            ArrayExtents,
	            RecordDim,
	            ExponentBits,
	            MantissaBits,
	            LinearizeArrayDimsFunctor,
	            std::conditional_t<
	                !std::is_void_v<StoredIntegral>,
	                StoredIntegral,
	                internal::StoredIntegralFor<RecordDim>>>;
	    };

	    template<typename Mapping>
	    inline constexpr bool isBitPackedFloatSoA = false;

	    template<typename... Ts>
	    inline constexpr bool isBitPackedFloatSoA<BitPackedFloatSoA<Ts...>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/BitPackedFloatSoA.hpp ==
	// ============================================================================

// #include "mapping/BitPackedIntSoA.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./mapping/Bytesplit.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
	// #include "Common.hpp"    // amalgamate: file already expanded

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
	    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
	    struct Bytesplit : private InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>
	    {
	        using Inner = InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>;

	        using ArrayExtents = typename Inner::ArrayExtents;
	        using ArrayIndex = typename Inner::ArrayIndex;
	        using RecordDim = TRecordDim; // hide Inner::RecordDim
	        using Inner::blobCount;

	        using Inner::blobSize;
	        using Inner::extents;

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
	        struct Reference : ProxyRefOpMixin<Reference<RC, BlobArray>, GetType<TRecordDim, RC>>
	        {
	        private:
	            const Inner& inner;
	            ArrayIndex ai;
	            BlobArray& blobs;

	        public:
	            using value_type = GetType<TRecordDim, RC>;

	            LLAMA_FN_HOST_ACC_INLINE Reference(const Inner& innerMapping, ArrayIndex ai, BlobArray& blobs)
	                : inner(innerMapping)
	                , ai(ai)
	                , blobs(blobs)
	            {
	            }

	            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
	            LLAMA_FN_HOST_ACC_INLINE operator value_type() const
	            {
	                value_type v;
	                auto* p = reinterpret_cast<std::byte*>(&v);
	                boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(value_type)>>(
	                    [&](auto ic)
	                    {
	                        constexpr auto i = decltype(ic)::value;
	                        auto&& ref = mapToMemory(inner, ai, Cat<RC, RecordCoord<i>>{}, blobs);
	                        p[i] = ref;
	                    });
	                return v;
	            }

	            LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type v) -> Reference&
	            {
	                auto* p = reinterpret_cast<std::byte*>(&v);
	                boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(value_type)>>(
	                    [&](auto ic)
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
	    template<template<typename, typename> typename InnerMapping>
	    struct BindBytesplit
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = Bytesplit<ArrayExtents, RecordDim, InnerMapping>;
	    };

	    template<typename Mapping>
	    inline constexpr bool isBytesplit = false;

	    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
	    inline constexpr bool isBytesplit<Bytesplit<TArrayExtents, TRecordDim, InnerMapping>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/Bytesplit.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/Byteswap.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "../Core.hpp"    // amalgamate: file already expanded
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
	// #include "Common.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/Projection.hpp ==
		// ==
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
		// #include "../View.hpp"    // amalgamate: file already expanded
		// #include "Common.hpp"    // amalgamate: file already expanded

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
		            using namespace boost::mp11;
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
		                    return boost::mp11::mp_identity<RecordDimType>{};
		                else
		                {
		                    using LoadFunc = UnaryFunctionTraits<decltype(&Projection::load)>;
		                    using StoreFunc = UnaryFunctionTraits<decltype(&Projection::store)>;

		                    static_assert(std::is_same_v<typename LoadFunc::ReturnType, RecordDimType>);
		                    static_assert(std::is_same_v<typename StoreFunc::ArgumentType, RecordDimType>);
		                    static_assert(std::is_same_v<typename LoadFunc::ArgumentType, typename StoreFunc::ReturnType>);

		                    return boost::mp11::mp_identity<typename StoreFunc::ReturnType>{};
		                }
		            }

		            template<typename Coord, typename RecordDimType>
		            using fn = typename decltype(replacedTypeProj<Coord, RecordDimType>())::type;
		        };

		        template<typename RecordDim, typename ProjectionMap>
		        using ReplaceTypesByProjectionResults
		            = TransformLeavesWithCoord<RecordDim, MakeReplacerProj<ProjectionMap>::template fn>;

		        template<typename Reference, typename Projection>
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
		        using ArrayIndex = typename Inner::ArrayIndex;
		        using RecordDim = TRecordDim; // hide Inner::RecordDim
		        using Inner::blobCount;
		        using Inner::blobSize;
		        using Inner::extents;
		        using Inner::Inner;

		        template<typename RecordCoord>
		        LLAMA_FN_HOST_ACC_INLINE static constexpr auto isComputed(RecordCoord) -> bool
		        {
		            return !std::is_void_v<
		                internal::ProjectionOrVoid<ProjectionMap, RecordCoord, GetType<RecordDim, RecordCoord>>>;
		        }

		        template<std::size_t... RecordCoords, typename BlobArray>
		        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
		            typename Inner::ArrayIndex ai,
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
		    template<template<typename, typename> typename InnerMapping, typename ProjectionMap>
		    struct BindProjection
		    {
		        template<typename ArrayExtents, typename RecordDim>
		        using fn = Projection<ArrayExtents, RecordDim, InnerMapping, ProjectionMap>;
		    };

		    template<typename Mapping>
		    inline constexpr bool isProjection = false;

		    template<
		        typename TArrayExtents,
		        typename TRecordDim,
		        template<typename, typename>
		        typename InnerMapping,
		        typename ReplacementMap>
		    inline constexpr bool isProjection<Projection<TArrayExtents, TRecordDim, InnerMapping, ReplacementMap>> = true;
		} // namespace llama::mapping
		// ==
		// == ./mapping/Projection.hpp ==
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
	        using MakeByteswapProjectionPair = boost::mp11::mp_list<T, ByteswapProjection<T>>;

	        template<typename RecordDim>
	        using MakeByteswapProjectionMap
	            = boost::mp11::mp_transform<MakeByteswapProjectionPair, boost::mp11::mp_unique<FlatRecordDim<RecordDim>>>;
	    } // namespace internal

	    /// Mapping that swaps the byte order of all values when loading/storing.
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
	    template<template<typename, typename> typename InnerMapping>
	    struct BindByteswap
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = Byteswap<ArrayExtents, RecordDim, InnerMapping>;
	    };

	    template<typename Mapping>
	    inline constexpr bool isByteswap = false;

	    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
	    inline constexpr bool isByteswap<Byteswap<TArrayExtents, TRecordDim, InnerMapping>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/Byteswap.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/ChangeType.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already expanded
	// #include "Common.hpp"    // amalgamate: file already expanded
	// #include "Projection.hpp"    // amalgamate: file already expanded

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template<typename UserT, typename StoredT>
	        struct ChangeTypeProjection
	        {
	            static auto load(StoredT v) -> UserT
	            {
	                return static_cast<UserT>(v); // we could allow stronger casts here
	            }

	            static auto store(UserT v) -> StoredT
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
	                    return boost::mp11::mp_identity<GetType<RecordDim, Key>>{};
	                else
	                    return boost::mp11::mp_identity<Key>{};
	            }

	            template<
	                typename Pair,
	                typename Key = boost::mp11::mp_first<Pair>,
	                typename StoredT = boost::mp11::mp_second<Pair>>
	            using fn = boost::mp11::
	                mp_list<Key, ChangeTypeProjection<typename decltype(recordDimType<Key>())::type, StoredT>>;
	        };

	        template<typename RecordDim, typename ReplacementMap>
	        using MakeProjectionMap
	            = boost::mp11::mp_transform<MakeProjectionPair<RecordDim>::template fn, ReplacementMap>;
	    } // namespace internal

	    /// Mapping that changes the type in the record domain for a different one in storage. Conversions happen during
	    /// load and store.
	    /// @tparam ReplacementMap A type list of binary type lists (a map) specifiying which type or the type at a \ref
	    /// RecordCoord (map key) to replace by which other type (mapped value).
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
	    template<template<typename, typename> typename InnerMapping, typename ReplacementMap>
	    struct BindChangeType
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = ChangeType<ArrayExtents, RecordDim, InnerMapping, ReplacementMap>;
	    };

	    template<typename Mapping>
	    inline constexpr bool isChangeType = false;

	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        template<typename, typename>
	        typename InnerMapping,
	        typename ReplacementMap>
	    inline constexpr bool isChangeType<ChangeType<TArrayExtents, TRecordDim, InnerMapping, ReplacementMap>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/ChangeType.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/Heatmap.hpp ==
	// ==
	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	// #include <array>    // amalgamate: file already included
	// #include <atomic>    // amalgamate: file already included
	#include <sstream>
	// #include <vector>    // amalgamate: file already included

	namespace llama::mapping
	{
	    /// Forwards all calls to the inner mapping. Counts all accesses made to blocks inside the blobs, allowing to
	    /// extract a heatmap.
	    /// @tparam Mapping The type of the inner mapping.
	    /// @tparam Granularity The granularity in bytes on which to could accesses. A value of 1 counts every byte.
	    /// individually. A value of e.g. 64, counts accesses per 64 byte block.
	    /// @tparam TCountType Data type used to count the number of accesses. Atomic increments must be supported for this
	    /// type.
	    template<
	        typename Mapping,
	        typename Mapping::ArrayExtents::value_type Granularity = 1,
	        typename TCountType = std::size_t>
	    struct Heatmap : private Mapping
	    {
	    private:
	        using size_type = typename Mapping::ArrayExtents::value_type;

	    public:
	        using Inner = Mapping;
	        inline static constexpr std::size_t granularity = Granularity;
	        using CountType = TCountType;
	        using typename Mapping::ArrayExtents;
	        using typename Mapping::ArrayIndex;
	        using typename Mapping::RecordDim;

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
	            return blockHitsSize(blobIndex) * sizeof(CountType);
	        }

	        template<std::size_t... RecordCoords>
	        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
	        {
	            return true;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto compute(
	            typename Mapping::ArrayIndex ai,
	            RecordCoord<RecordCoords...> rc,
	            Blobs& blobs) const -> decltype(auto)
	        {
	            static_assert(
	                !std::is_const_v<Blobs>,
	                "Cannot access (even just reading) data through Heatmap from const blobs/view, since we need to write "
	                "the access counts");

	            const auto [nr, offset] = Mapping::blobNrAndOffset(ai, rc);
	            using Type = GetType<typename Mapping::RecordDim, RecordCoord<RecordCoords...>>;

	            auto* hits = blockHits(nr, blobs);
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
	            return divCeil(Mapping::blobSize(forBlobI), Granularity);
	        }
	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING

	        template<typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto blockHits(size_type forBlobI, const Blobs& blobs) const -> const CountType*
	        {
	            return reinterpret_cast<const CountType*>(&blobs[size_type{Mapping::blobCount} + forBlobI][0]);
	        }

	        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
	        template<typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE auto blockHits(size_type forBlobI, Blobs& blobs) const -> CountType*
	        {
	            return reinterpret_cast<CountType*>(&blobs[size_type{Mapping::blobCount} + forBlobI][0]);
	        }

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
	                auto* bh = blockHits(i, blobs);
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
	                auto* bh = blockHits(i, blobs);
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

	    template<typename Mapping>
	    inline constexpr bool isHeatmap = false;

	    template<typename Mapping, typename Mapping::ArrayExtents::value_type Granularity, typename CountType>
	    inline constexpr bool isHeatmap<Heatmap<Mapping, Granularity, CountType>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/Heatmap.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/Null.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "../ProxyRefOpMixin.hpp"    // amalgamate: file already expanded

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

	    template<typename Mapping>
	    inline constexpr bool isNull = false;

	    template<typename ArrayExtents, typename RecordDim>
	    inline constexpr bool isNull<Null<ArrayExtents, RecordDim>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/Null.hpp ==
	// ============================================================================

// #include "mapping/One.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./mapping/PermuteArrayIndex.hpp ==
	// ==
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "Common.hpp"    // amalgamate: file already expanded

	namespace llama::mapping
	{
	    /// Meta mapping permuting the array indices before forwarding to another mapping. The array extents are not
	    /// changed.
	    /// @tparam Permutation The pack of integrals describing the permutation of the array indices. The inner mapping
	    /// will be called with an ArrayIndex{ai[Permutation]...}.
	    template<typename Mapping, std::size_t... Permutation>
	    struct PermuteArrayIndex : Mapping
	    {
	    private:
	        using size_type = typename Mapping::ArrayExtents::value_type;

	    public:
	        using Inner = Mapping;
	        using ArrayIndex = typename Inner::ArrayIndex;

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

	    template<typename Mapping>
	    PermuteArrayIndex(Mapping) -> PermuteArrayIndex<Mapping>;

	    template<typename Mapping>
	    inline constexpr bool isPermuteArrayIndex = false;

	    template<typename Mapping, std::size_t... Permutation>
	    inline constexpr bool isPermuteArrayIndex<PermuteArrayIndex<Mapping, Permutation...>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/PermuteArrayIndex.hpp ==
	// ============================================================================

// #include "mapping/Projection.hpp"    // amalgamate: file already expanded
// #include "mapping/SoA.hpp"    // amalgamate: file already expanded
	// ============================================================================
	// == ./mapping/Split.hpp ==
	// ==
	// #pragma once
	// #include "../View.hpp"    // amalgamate: file already expanded
	// #include "Common.hpp"    // amalgamate: file already expanded

	namespace llama::mapping
	{
	    namespace internal
	    {
	        template<typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
	        auto partitionRecordDim(Record<Fields...>, RecordCoord<FirstCoord, Coords...>)
	        {
	            using namespace boost::mp11;
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
	            using Part1Before = boost::mp11::mp_first<Acc>;
	            using Part2Before = boost::mp11::mp_second<Acc>;
	            using R = decltype(partitionRecordDim(Part2Before{}, GetCoordFromTags<Part2Before, TagList>{}));
	            using Part1After = boost::mp11::mp_first<R>;
	            using Part2After = boost::mp11::mp_second<R>;

	            using type = boost::mp11::mp_list<MergedRecordDims<Part1Before, Part1After>, Part2After>;
	        };

	        template<typename Acc, typename TagList>
	        using PartitionFoldOp = typename PartitionFoldOpImpl<Acc, TagList>::type;

	        template<typename... Fields, typename... RCs>
	        auto partitionRecordDim(Record<Fields...>, boost::mp11::mp_list<RCs...>)
	        {
	            using namespace boost::mp11;
	            using Initial = mp_list<Record<>, Record<Fields...>>; // initially, nothing selected for mapping 1
	            return mp_fold<mp_list<GetTags<Record<Fields...>, RCs>...>, Initial, PartitionFoldOp>{};
	        }

	        // workaround for nvcc 11.3 and below: we cannot put the decltype() directly into the Split class
	        template<typename RecordDim, typename RecordCoordForMapping1>
	        struct PartionedRecordDim
	        {
	            using type = decltype(partitionRecordDim(RecordDim{}, RecordCoordForMapping1{}));
	        };

	        template<typename RC, typename RecordCoordForMapping1>
	        inline constexpr bool isSelected = recordCoordCommonPrefixIsSame<RecordCoordForMapping1, RC>;

	        template<typename RC>
	        struct IsSelectedPredicate
	        {
	            template<typename RecordCoordForMapping1>
	            using fn = boost::mp11::mp_bool<isSelected<RC, RecordCoordForMapping1>>;
	        };

	        template<typename RC, typename... RecordCoordsForMapping1>
	        inline constexpr bool isSelected<RC, boost::mp11::mp_list<RecordCoordsForMapping1...>> = boost::mp11::
	            mp_any_of_q<boost::mp11::mp_list<RecordCoordsForMapping1...>, IsSelectedPredicate<RC>>::value;
	    } // namespace internal

	    /// Mapping which splits off a part of the record dimension and maps it differently then the rest.
	    /// \tparam TRecordCoordForMapping1 A \ref RecordCoord or a list of RecordCoords selecting the part of the record
	    /// dimension to be mapped differently.
	    /// \tparam MappingTemplate1 The mapping used for the selected part of the record dimension.
	    /// \tparam MappingTemplate2 The mapping used for the not selected part of the record dimension.
	    /// \tparam SeparateBlobs If true, both pieces of the record dimension are mapped to separate blobs.
	    template<
	        typename TArrayExtents,
	        typename TRecordDim,
	        typename TRecordCoordForMapping1,
	        template<typename...>
	        typename MappingTemplate1,
	        template<typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs = false>
	    struct Split
	    {
	        using ArrayExtents = TArrayExtents;
	        using ArrayIndex = typename ArrayExtents::Index;
	        using RecordDim = TRecordDim;

	        using RecordCoordForMapping1 = TRecordCoordForMapping1;
	        using RecordDimPartitions = typename internal::PartionedRecordDim<RecordDim, RecordCoordForMapping1>::type;
	        using RecordDim1 = boost::mp11::mp_first<RecordDimPartitions>;
	        using RecordDim2 = boost::mp11::mp_second<RecordDimPartitions>;

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

	            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, RecordCoordForMapping1>)
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
	            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, RecordCoordForMapping1>)
	                return llama::isComputed<Mapping1, GetCoordFromTags<RecordDim1, Tags>>;
	            else
	                return llama::isComputed<Mapping2, GetCoordFromTags<RecordDim2, Tags>>;
	        }

	        template<std::size_t... RecordCoords, typename Blobs>
	        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(ArrayIndex ai, RecordCoord<RecordCoords...>, Blobs& blobs)
	            const
	        {
	            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;
	            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, RecordCoordForMapping1>)
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
	    template<
	        typename RecordCoordsForMapping1,
	        template<typename...>
	        typename MappingTemplate1,
	        template<typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs = false>
	    struct BindSplit
	    {
	        template<typename ArrayExtents, typename RecordDim>
	        using fn = Split<
	            ArrayExtents,
	            RecordDim,
	            RecordCoordsForMapping1,
	            MappingTemplate1,
	            MappingTemplate2,
	            SeparateBlobs>;
	    };

	    template<typename Mapping>
	    inline constexpr bool isSplit = false;

	    template<
	        typename ArrayExtents,
	        typename RecordDim,
	        typename RecordCoordsForMapping1,
	        template<typename...>
	        typename MappingTemplate1,
	        template<typename...>
	        typename MappingTemplate2,
	        bool SeparateBlobs>
	    inline constexpr bool isSplit<Split<
	        ArrayExtents,
	        RecordDim,
	        RecordCoordsForMapping1,
	        MappingTemplate1,
	        MappingTemplate2,
	        SeparateBlobs>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/Split.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/Trace.hpp ==
	// ==
	// #pragma once
	// #include "../StructName.hpp"    // amalgamate: file already expanded
	// #include "Common.hpp"    // amalgamate: file already expanded

	#include <cstdio>
	#include <iomanip>
	// #include <iostream>    // amalgamate: file already included

	namespace llama::mapping
	{
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
	        struct TraceReference : ProxyRefOpMixin<TraceReference<Value, Ref, Count>, Value>
	        {
	            using value_type = Value;

	            template<typename RefFwd>
	            LLAMA_FN_HOST_ACC_INLINE TraceReference(RefFwd&& r, AccessCounts<Count>* hits)
	                : r(std::forward<RefFwd>(r))
	                , hits(hits)
	            {
	                static_assert(std::is_same_v<std::remove_reference_t<Ref>, std::remove_reference_t<RefFwd>>);
	            }

	            TraceReference(const TraceReference&) = default;
	            TraceReference(TraceReference&&) noexcept = default;
	            auto operator=(TraceReference&& ref) noexcept -> TraceReference& = default;
	            ~TraceReference() = default;

	            LLAMA_FN_HOST_ACC_INLINE auto operator=(const TraceReference& ref) -> TraceReference&
	            {
	                if(&ref != this)
	                {
	                    internal::atomicInc(hits->writes);
	                    r = static_cast<value_type>(ref);
	                }
	                return *this;
	            }

	            LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type value) -> TraceReference&
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

	    /// Forwards all calls to the inner mapping. Traces all accesses made through this mapping and allows printing a
	    /// summary.
	    /// @tparam Mapping The type of the inner mapping.
	    /// @tparam TCountType The type used for counting the number of accesses.
	    /// @tparam MyCodeHandlesProxyReferences If false, Trace will avoid proxy references but can then only count
	    /// the number of address computations
	    template<typename Mapping, typename TCountType = std::size_t, bool MyCodeHandlesProxyReferences = true>
	    struct Trace : Mapping
	    {
	    private:
	        using size_type = typename Mapping::ArrayExtents::value_type;

	    public:
	        using RecordDim = typename Mapping::RecordDim;
	        using CountType = TCountType;
	        inline static constexpr bool myCodeHandlesProxyReferences = MyCodeHandlesProxyReferences;
	        using FieldHitsArray = Array<AccessCounts<CountType>, flatFieldCount<RecordDim>>;

	        inline static constexpr auto blobCount = Mapping::blobCount + 1;

	        constexpr Trace() = default;

	        LLAMA_FN_HOST_ACC_INLINE
	        explicit Trace(Mapping mapping) : Mapping(std::move(mapping))
	        {
	        }

	        template<typename... Args>
	        LLAMA_FN_HOST_ACC_INLINE explicit Trace(Args&&... innerArgs) : Mapping(std::forward<Args>(innerArgs)...)
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
	            typename Mapping::ArrayIndex ai,
	            RecordCoord<RecordCoords...> rc,
	            Blobs& blobs) const -> decltype(auto)
	        {
	            static_assert(
	                !std::is_const_v<Blobs>,
	                "Cannot access (even just reading) data through Trace from const blobs/view, since we need to write "
	                "the access counts");

	            auto& hits = fieldHits(blobs)[+flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>];
	            decltype(auto) ref = mapToMemory(inner(), ai, rc, blobs); // T& or proxy reference (value)
	            if constexpr(MyCodeHandlesProxyReferences)
	            {
	                using Value = GetType<RecordDim, decltype(rc)>;
	                using Ref = decltype(ref);
	                return internal::TraceReference<Value, Ref, CountType>{std::forward<Ref>(ref), &hits};
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

	        void printFieldHitsHost(const FieldHitsArray& hits) const
	        {
	            if constexpr(MyCodeHandlesProxyReferences)
	                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
	                          << std::setw(columnWidth) << "Reads" << ' ' << std::right << std::setw(columnWidth)
	                          << "Writes" << '\n';
	            else
	                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
	                          << std::setw(columnWidth) << "Mlocs comp" << '\n';
	            forEachLeafCoord<RecordDim>(
	                [&](auto rc)
	                {
	                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
	                    if constexpr(MyCodeHandlesProxyReferences)
	                        std::cout << std::left << std::setw(columnWidth) << recordCoordTags<RecordDim>(rc) << ' '
	                                  << std::right << std::setw(columnWidth) << hits[i].reads << ' ' << std::right
	                                  << std::setw(columnWidth) << hits[i].writes << '\n';
	                    else
	                        std::cout << std::left << std::setw(columnWidth) << recordCoordTags<RecordDim>(rc) << ' '
	                                  << std::right << std::setw(columnWidth) << hits[i].memLocsComputed << '\n';
	                });
	            std::cout << std::internal;
	        }

	        LLAMA_ACC void printFieldHitsDevice(const FieldHitsArray& hits) const
	        {
	            if constexpr(MyCodeHandlesProxyReferences)
	            {
	                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                printf("%*s %*s %*s\n", columnWidth, "Field", columnWidth, "Reads", columnWidth, "Writes");
	            }
	            else
	            {
	                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                printf("%*s %*s\n", columnWidth, "Field", columnWidth, "Mlocs comp");
	            }
	            forEachLeafCoord<RecordDim>(
	                [&](auto rc)
	                {
	                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
	                    constexpr auto fieldName = recordCoordTags<RecordDim>(rc);
	                    char fieldNameZT[fieldName.size() + 1]{}; // nvcc does not handle the %*.*s parameter correctly
	                    llama::internal::constexprCopy(fieldName.begin(), fieldName.end(), fieldNameZT);
	                    if constexpr(MyCodeHandlesProxyReferences)
	                    {
	                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
	                        printf(
	                            "%*.s %*lu %*lu\n",
	                            columnWidth,
	                            fieldNameZT,
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
	                            columnWidth,
	                            static_cast<unsigned long>(hits[i].memLocsComputed));
	                    }
	                });
	        }

	        LLAMA_FN_HOST_ACC_INLINE auto inner() const -> const Mapping&
	        {
	            return static_cast<const Mapping&>(*this);
	        }
	    };

	    template<typename Mapping>
	    inline constexpr bool isTrace = false;

	    template<typename Mapping, typename CountType, bool MyCodeHandlesProxyReferences>
	    inline constexpr bool isTrace<Trace<Mapping, CountType, MyCodeHandlesProxyReferences>> = true;
	} // namespace llama::mapping
	// ==
	// == ./mapping/Trace.hpp ==
	// ============================================================================

	// ============================================================================
	// == ./mapping/tree/Mapping.hpp ==
	// ==
	// Copyright 2018 Alexander Matthes
	// SPDX-License-Identifier: GPL-3.0-or-later

	// #pragma once
	// #include "../Common.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/tree/Functors.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
			// ============================================================================
			// == ./mapping/tree/TreeFromDimensions.hpp ==
			// ==
			// Copyright 2018 Alexander Matthes
			// SPDX-License-Identifier: GPL-3.0-or-later
			// #pragma once
			// #include "../../Core.hpp"    // amalgamate: file already expanded
			// #include "../../Tuple.hpp"    // amalgamate: file already expanded

			// #include <cstddef>    // amalgamate: file already included
			// #include <string>    // amalgamate: file already included
			// #include <type_traits>    // amalgamate: file already included

			namespace llama::mapping::tree
			{
			    template<typename T>
			    inline constexpr auto one = 1;

			    template<>
			    inline constexpr auto one<boost::mp11::mp_size_t<1>> = boost::mp11::mp_size_t<1>{};

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
			        static constexpr boost::mp11::mp_size_t<ChildIndex> childIndex = {};
			        const ArrayIndexType arrayIndex = {};
			    };

			    template<std::size_t... Coords>
			    using TreeCoord = Tuple<TreeCoordElement<Coords, boost::mp11::mp_size_t<0>>...>;

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
			            using type = Leaf<Tag, RecordDim, boost::mp11::mp_size_t<1>>;
			        };

			        template<typename Tag, typename... Fields, typename CountType>
			        struct CreateTreeElement<Tag, Record<Fields...>, CountType>
			        {
			            using type = Node<
			                Tag,
			                Tuple<
			                    typename CreateTreeElement<GetFieldTag<Fields>, GetFieldType<Fields>, boost::mp11::mp_size_t<1>>::
			                        type...>,
			                CountType>;
			        };

			        template<typename Tag, typename ChildType, std::size_t Count, typename CountType>
			        struct CreateTreeElement<Tag, ChildType[Count], CountType>
			        {
			            template<std::size_t... Is>
			            static auto createChildren(std::index_sequence<Is...>)
			            {
			                return Tuple<
			                    typename CreateTreeElement<RecordCoord<Is>, ChildType, boost::mp11::mp_size_t<1>>::type...>{};
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
			                TreeCoordElement<RecordCoords, boost::mp11::mp_size_t<0>>{}...,
			                TreeCoordElement<0, boost::mp11::mp_size_t<0>>{}};
			        }
			    } // namespace internal

			    template<typename RecordCoord, typename ArrayIndex>
			    LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(const ArrayIndex& ai)
			    {
			        return internal::createTreeCoord(ai, std::make_index_sequence<ArrayIndex::rank>{}, RecordCoord{});
			    }
			} // namespace llama::mapping::tree
			// ==
			// == ./mapping/tree/TreeFromDimensions.hpp ==
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
		            return Node<Identifier, decltype(children), boost::mp11::mp_size_t<1>>{{}, children};
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
		                auto first = TreeCoordElement<BasicCoord::FirstElement::childIndex, boost::mp11::mp_size_t<0>>{};

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
		    using MoveRTDownFixed = MoveRTDown<TreeCoord, boost::mp11::mp_size_t<Amount>>;
		} // namespace llama::mapping::tree::functor
		// ==
		// == ./mapping/tree/Functors.hpp ==
		// ============================================================================

	// #include "TreeFromDimensions.hpp"    // amalgamate: file already expanded
		// ============================================================================
		// == ./mapping/tree/toString.hpp ==
		// ==
		// Copyright 2018 Alexander Matthes
		// SPDX-License-Identifier: GPL-3.0-or-later

		// #pragma once
		// #include "TreeFromDimensions.hpp"    // amalgamate: file already expanded

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
		// == ./mapping/tree/toString.hpp ==
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
	            boost::mp11::mp_first<Tuple<Operations...>> operation = {};
	            using ResultTree = decltype(operation.basicToResult(Tree()));
	            ResultTree treeAfterOp;
	            MergeFunctors<ResultTree, boost::mp11::mp_drop_c<Tuple<Operations...>, 1>> next = {};

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
	            constexpr std::size_t childCount = boost::mp11::mp_size<std::decay_t<decltype(node.childs)>>::value;
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
	// == ./mapping/tree/Mapping.hpp ==
	// ============================================================================


#if defined(__CUDACC__) || defined(__NVCOMPILER)
#    ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#        pragma nv_diag_default 940
#    else
#        pragma diag_default 940
#    endif
#endif
// ==
// == ./llama.hpp ==
// ============================================================================


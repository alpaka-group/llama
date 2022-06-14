// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "macros.hpp"

#include <ostream>
#include <tuple>

namespace llama
{
    /// Array class like `std::array` but suitable for use with offloading devices like GPUs.
    /// \tparam T type if array elements.
    /// \tparam N rank of the array.
    template<typename T, std::size_t N>
    struct Array
    {
        using value_type = T;
        T element[N > 0 ? N : 1];

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
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&& idx) const -> T const&
        {
            return element[idx];
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
    LLAMA_FN_HOST_ACC_INLINE constexpr auto push_front([[maybe_unused]] Array<T, N> a, T v) -> Array<T, N + 1>
    {
        Array<T, N + 1> r{};
        r[0] = v;
        if constexpr(N > 0)
            for(std::size_t i = 0; i < N; i++)
                r[i + 1] = a[i];
        return r;
    }

    template<typename T, std::size_t N>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto push_back([[maybe_unused]] Array<T, N> a, T v) -> Array<T, N + 1>
    {
        Array<T, N + 1> r{};
        if constexpr(N > 0)
            for(std::size_t i = 0; i < N; i++)
                r[i] = a[i];
        r[N] = v;
        return r;
    }

    template<typename T, std::size_t N>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto pop_back([[maybe_unused]] Array<T, N> a)
    {
        static_assert(N > 0);
        Array<T, N - 1> r{};
        if constexpr(N > 1)
            for(std::size_t i = 0; i < N - 1; i++)
                r[i] = a[i];
        return r;
    }

    template<typename T, std::size_t N>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto pop_front([[maybe_unused]] Array<T, N> a)
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
    struct tuple_size<llama::Array<T, N>> : integral_constant<size_t, N>
    {
    };

    template<size_t I, typename T, size_t N>
    struct tuple_element<I, llama::Array<T, N>>
    {
        using type = T;
    };
} // namespace std

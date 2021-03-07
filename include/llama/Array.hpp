// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "macros.hpp"

#include <tuple>

namespace llama
{
    /// Array class like `std::array` but suitable for use with offloading
    /// devices like GPUs.
    /// \tparam T type if array elements.
    /// \tparam N rank of the array.
    template <typename T, std::size_t N>
    struct Array
    {
        using value_type = T;
        static constexpr std::size_t rank
            = N; // FIXME this is right from the ArrayDims's POV, but wrong from the Array's POV
        T element[N > 0 ? N : 1];

        LLAMA_FN_HOST_ACC_INLINE constexpr T* begin()
        {
            return &element[0];
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr const T* begin() const
        {
            return &element[0];
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr T* end()
        {
            return &element[N];
        };

        LLAMA_FN_HOST_ACC_INLINE constexpr const T* end() const
        {
            return &element[N];
        };

        template <typename IndexType>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&& idx) -> T&
        {
            return element[idx];
        }

        template <typename IndexType>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](IndexType&& idx) const -> T const&
        {
            return element[idx];
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr friend auto operator==(const Array& a, const Array& b) -> bool
        {
            for (std::size_t i = 0; i < N; ++i)
                if (a.element[i] != b.element[i])
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
            for (std::size_t i = 0; i < N; ++i)
                temp[i] = a[i] + b[i];
            return temp;
        }

        template <std::size_t I>
        constexpr auto get() -> T&
        {
            return element[I];
        }

        template <std::size_t I>
        constexpr auto get() const -> const T&
        {
            return element[I];
        }
    };

    template <typename First, typename... Args>
    Array(First, Args... args) -> Array<First, sizeof...(Args) + 1>;

    template <typename T, std::size_t N>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto push_front(Array<T, N> a, T v) -> Array<T, N + 1>
    {
        Array<T, N + 1> r{};
        r[0] = v;
        for (auto i = 0; i < N; i++)
            r[i + 1] = a[i];
        return r;
    }

    template <typename T, std::size_t N>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto push_back(Array<T, N> a, T v) -> Array<T, N + 1>
    {
        Array<T, N + 1> r{};
        for (auto i = 0; i < N; i++)
            r[i] = a[i];
        r[N] = v;
        return r;
    }

    template <typename T, std::size_t N>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto pop_back(Array<T, N> a)
    {
        static_assert(N > 0);
        Array<T, N - 1> r{};
        for (auto i = 0; i < N - 1; i++)
            r[i] = a[i];
        return r;
    }

    template <typename T, std::size_t N>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto pop_front(Array<T, N> a)
    {
        static_assert(N > 0);
        Array<T, N - 1> r{};
        for (auto i = 0; i < N - 1; i++)
            r[i] = a[i + 1];
        return r;
    }
} // namespace llama

namespace std
{
    template <typename T, size_t N>
    struct tuple_size<llama::Array<T, N>> : integral_constant<size_t, N>
    {
    };

    template <size_t I, typename T, size_t N>
    struct tuple_element<I, llama::Array<T, N>>
    {
        using type = T;
    };
} // namespace std

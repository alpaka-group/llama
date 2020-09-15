/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

#include "macros.hpp"

namespace llama
{
    /// Array class like `std::array` but suitable for use with offloading
    /// devices like GPUs.
    /// \tparam T type if array elements.
    /// \tparam N rank of the array.
    template<typename T, std::size_t N>
    struct Array
    {
        static constexpr std::size_t rank = N;
        T element[N > 0 ? N : 1];

        LLAMA_FN_HOST_ACC_INLINE T * begin()
        {
            return &element[0];
        }

        LLAMA_FN_HOST_ACC_INLINE const T * begin() const
        {
            return &element[0];
        }

        LLAMA_FN_HOST_ACC_INLINE T * end()
        {
            return &element[N];
        };

        LLAMA_FN_HOST_ACC_INLINE const T * end() const
        {
            return &element[N];
        };

        template<typename IndexType>
        LLAMA_FN_HOST_ACC_INLINE auto operator[](IndexType && idx) -> T &
        {
            return element[idx];
        }

        template<typename IndexType>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto
        operator[](IndexType && idx) const -> T const &
        {
            return element[idx];
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto
        operator==(const Array<T, N> & a, const Array<T, N> & b) -> bool
        {
            for(std::size_t i = 0; i < N; ++i)
                if(a.element[i] != b.element[i])
                    return false;
            return true;
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto
        operator+(const Array<T, N> & a, const Array<T, N> & b) -> Array
        {
            Array temp;
            for(std::size_t i = 0; i < N; ++i) temp[i] = a[i] + b[i];
            return temp;
        }
    };

    template<typename First, typename... Args>
    Array(First, Args... args) -> Array<First, sizeof...(Args) + 1>;
}

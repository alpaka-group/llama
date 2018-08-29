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
#include "preprocessor/macros.hpp"

namespace llama
{

template<
    typename T,
    std::size_t T_dim
>
struct Array
{
    static constexpr std::size_t count = T_dim;

    Array() = default;
    Array( Array const & ) = default;
    Array( Array && ) = default;
    ~Array( ) = default;

    T element[count];

    LLAMA_FN_HOST_ACC_INLINE
    T* begin()
    {
        return &(element[0]);
    };

    LLAMA_FN_HOST_ACC_INLINE
    T* end()
    {
        return &(element[count]);
    };

    template< typename T_IndexType >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator[] ( T_IndexType && idx )
    -> T &
    {
        return element[ idx ];
    }

    template< typename T_IndexType >
    LLAMA_FN_HOST_ACC_INLINE
    constexpr
    auto
    operator[] ( T_IndexType && idx ) const
    -> T const &
    {
        return element[ idx ];
    }

    auto
    LLAMA_FN_HOST_ACC_INLINE
    pop_front() const
    -> Array<
        T,
        count-1
    >
    {
        Array<
            T,
            count - 1
        > result;
        for( std::size_t i = 0; i < count - 1; i++ )
            result.element[ i ] = element[ i + 1 ];
        return result;
    }

    auto
    LLAMA_FN_HOST_ACC_INLINE
    pop_back() const
    -> Array<
        T,
        count-1
    >
    {
        Array<
            T,
            count - 1
        > result;
        for( std::size_t i = 0; i < count - 1; i++ )
            result.element[ i ] = element[ i ];
        return result;
    }

    auto
    LLAMA_FN_HOST_ACC_INLINE
    push_front( T const new_element ) const
    -> Array<
        T,
        count+1
    >
    {
        Array<
            T,
            count+1
        > result;
        for( std::size_t i = 0; i < count - 1; i++ )
            result.element[ i + 1 ] = element[ i ];
        result.element[ 0 ] = new_element;
        return result;
    }

    auto
    LLAMA_FN_HOST_ACC_INLINE
    push_back( T const new_element ) const
    -> Array<
        T,
        count + 1
    >
    {
        Array<
            T,
            count + 1
        > result;
        for( std::size_t i = 0; i < count-1; i++ )
            result.element[ i ] = element[ i ];
        result.element[ count ] = new_element;
        return result;
    }

    template< typename T_Other >
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator==(const T_Other& other) const
    -> bool
    {
        if ( count != other.count )
            return false;
        for (std::size_t i = 0; i < count; ++i)
            if ( element[i] != other.element[i] )
                return false;
        return true;
    }

    template< typename T_Other >
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator+( const T_Other &second ) const
    -> Array
    {
        Array temp;
        for (std::size_t i = 0; i < count; ++i)
            temp.element[i] = element[i] + second[i];
        return temp;
    }
};

} // namespace llama

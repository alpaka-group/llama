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

/** Array class like std::array but suitable for use with offloading devices
 *  like GPUs and extended with some (for LLAMA) useful methods.
 * \tparam T type if array elements
 * \tparam T_dim number of elements in array
 * */
template<
    typename T,
    std::size_t T_dim
>
struct Array
{
    /// Number of elements in array
    static constexpr std::size_t count = T_dim;

    Array() = default;
    Array( Array const & ) = default;
    Array( Array && ) = default;
    ~Array( ) = default;

    /// Elements in the array, best to access with \ref operator[].
    T element[count];

    /** Returns an iterator to the first element. Basically just a pointer to
     *  the internal array of elements, which can be incremented.
     * \return pointer to first element
     * */
    LLAMA_FN_HOST_ACC_INLINE
    T* begin()
    {
        return &(element[0]);
    };

    /** Returns an iterator to the element after the last element.
     * \return pointer to element after the last element
     * */
    LLAMA_FN_HOST_ACC_INLINE
    T* end()
    {
        return &(element[count]);
    };

    /** Gives access to an element of the array *without* range check.
     * \tparam T_IndexType type of index
     * \param idx index of element
     * \return reference to element at index
     * */
    template< typename T_IndexType >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator[] ( T_IndexType && idx )
    -> T &
    {
        return element[ idx ];
    }

    /** Gives const access to an element of the array *without* range check.
     * \tparam T_IndexType type of index
     * \param idx index of element
     * \return const reference to element at index
     * */
    template< typename T_IndexType >
    LLAMA_FN_HOST_ACC_INLINE
    constexpr
    auto
    operator[] ( T_IndexType && idx ) const
    -> T const &
    {
        return element[ idx ];
    }

    /** Returns a copy of the array but with the first element removed.
     * \return Array with one element less
     * */
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

    /** Returns a copy of the array but with the last element removed.
     * \return Array with one element less
     * */
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

    /** Returns a copy of the array but with the one element added in front of
     *  the (former) first element.
     * \param new_element new element of type T to add at the beginning
     * \return Array with one element more
     * */
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

    /** Returns a copy of the array but with the one element added after the
     *  (former) last element.
     * \param new_element new element of type T to add at the end
     * \return Array with one element more
     * */
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

    /** Checks whether two arrays are elementwise the same. Returns false if at
     *  least one pair of elements with the same index in both arrays are not
     *  the same. Returns always false for arrays of different sizes.
     * \tparam T_Other type of other arrays. The type of the elements of the
     *  other array may differ and the operator still return true (e.g. for int
     *  and char).
     * \param other other array to compare with
     * \return true if the arrays are the same, otherwise false
     * */
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

    /** Adds an array to an existing array. May access invalid memory if the
     *  second array is smaller than the first!
     * \tparam T_Other type of the other array. The types of the elements of the
     *  arrays may differ.
     * \param second other array to add
     * \return a new array of the same type as the first array
     * */
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

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

#include "Array.hpp"

namespace llama
{

/** Helper class just holding some numbers at compile time. A similar sequence
 *  is part of the STL starting with C++14, but as LLAMA supports down to C++11
 *  an own implementation is needed
 * \tparam T_seg... variadic sequence of integers.
 */
template< std::size_t... T_seq >
struct IntegerSequence { };

namespace internal
{

template<
    std::size_t T_FirstVal,
    std::size_t... T_vals
>
struct IntegerSequenceHelper
{
    using type = typename IntegerSequenceHelper<
        T_FirstVal-1,
        T_FirstVal,
        T_vals...
    >::type;
};

template < std::size_t... T_vals >
struct IntegerSequenceHelper<
    0,
    T_vals...
>
{
    using type = IntegerSequence< 0, T_vals...>;
};

} // namespace internal

/** Gives the type of an \ref IntegerSequence with ascending elements, e.g.
 *  MakeIntegerSequence< 4 > = IntegerSequence< 0, 1, 2, 3 >.
 * \tparam T_N number of integers in the sequence
 * \return an \ref IntegerSequence type
 */
template< std::size_t T_N >
using MakeIntegerSequence = typename
internal::IntegerSequenceHelper< T_N - 1 >::type;

namespace internal
{

template<
    std::size_t T_Iter,
    std::size_t... T_vals
>
struct ZeroSequenceHelper
{
    using type = typename ZeroSequenceHelper<
        T_Iter-1,
        0,
        T_vals...
    >::type;
};

template < std::size_t... T_vals >
struct ZeroSequenceHelper<
    0,
    T_vals...
>
{
    using type = IntegerSequence< T_vals...>;
};

} // namespace internal


/** Gives the type of an \ref IntegerSequence of zeros, e.g.
 *  MakeIntegerSequence< 4 > = IntegerSequence< 0, 0, 0, 0 >.
 * \tparam T_N number of integers in the sequence
 * \return an \ref IntegerSequence type
 */
template< std::size_t T_N >
using MakeZeroSequence = typename
internal::ZeroSequenceHelper< T_N >::type;


} // namespace llama

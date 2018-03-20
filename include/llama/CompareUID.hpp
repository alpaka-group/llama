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

#include <type_traits>
#include "GetUID.hpp"

namespace llama
{

template<
    typename T_DDA,
    typename T_BaseA,
    typename T_LocalA,
    typename T_DDB,
    typename T_BaseB,
    typename T_LocalB,
    typename SFINAE = void
>
struct CompareUID
{
    static constexpr bool value =
        std::is_same<
            GetUIDFromName<
                T_DDA,
                typename T_BaseA::template PushBack< T_LocalA::front >
            >,
            GetUIDFromName<
                T_DDB,
                typename T_BaseB::template PushBack< T_LocalB::front >
            >
        >::value &&
        CompareUID<
            T_DDA,
            typename T_BaseA::template PushBack< T_LocalA::front >,
            typename T_LocalA::PopFront,
            T_DDB,
            typename T_BaseB::template PushBack< T_LocalB::front >,
            typename T_LocalB::PopFront
        >::value;
};

template<
    typename T_DDA,
    typename T_BaseA,
    typename T_LocalA,
    typename T_DDB,
    typename T_BaseB,
    typename T_LocalB
>
struct CompareUID
<
    T_DDA,
    T_BaseA,
    T_LocalA,
    T_DDB,
    T_BaseB,
    T_LocalB,
    typename std::enable_if< (T_LocalA::size != T_LocalB::size) >::type
>
{
    static constexpr bool value = false;
};

template<
    typename T_DDA,
    typename T_BaseA,
    typename T_LocalA,
    typename T_DDB,
    typename T_BaseB,
    typename T_LocalB
>
struct CompareUID
<
    T_DDA,
    T_BaseA,
    T_LocalA,
    T_DDB,
    T_BaseB,
    T_LocalB,
    typename std::enable_if<
        (T_LocalA::size == 0) &&
        (T_LocalB::size == 0)
    >::type
>
{
    static constexpr bool value = true;
};


} // namespace llama

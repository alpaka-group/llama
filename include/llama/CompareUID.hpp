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

/** Tells whether two coordinates in two datum domains have the same UID.
 * \tparam T_DDA first user domain
 * \tparam T_BaseA First part of the coordinate in the first user domain as
 *  \ref DatumCoord. This will be used for getting the UID, but not for the
 *  comparison.
 * \tparam T_LocalA Second part of the coordinate in the first user domain as
 *  \ref DatumCoord. This will be used for the comparison with the second
 *  datum domain.
 * \tparam T_DDB second user domain
 * \tparam T_BaseB First part of the coordinate in the second user domain as
 *  \ref DatumCoord. This will be used for getting the UID, but not for the
 *  comparison.
 * \tparam T_LocalB Second part of the coordinate in the second user domain as
 *  \ref DatumCoord. This will be used for the comparison with the first
 *  datum domain.
 */
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
    /// true if the two UIDs are exactly the same, otherwise false.
    static constexpr bool value LLAMA_IGNORE_LITERAL(;) =
        std::is_same<
            GetUID<
                T_DDA,
                typename T_BaseA::template PushBack< T_LocalA::front >
            >,
            GetUID<
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

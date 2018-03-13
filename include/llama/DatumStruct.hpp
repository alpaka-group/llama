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

#include "Types.hpp"
#include "internal/TreeHelper.hpp"

namespace llama
{

template< typename... T_Leaves >
struct DatumStruct
{
    static constexpr std::size_t size =
        internal::GetSizeOfDatumStructLeaves< T_Leaves... >::value;

    template< std::size_t T_coord >
    struct GetBranch
    {
        using type = typename internal::GetLeave<
            T_coord,
            T_Leaves...
        >::type;
    };

    template< std::size_t... T_coords >
    struct LinearBytePos
    {
        static constexpr std::size_t value =
            internal::GetSizeOfDatumStructLeavesWithCoord<
                DatumCoord< T_coords... >,
                DatumCoord< 0 >,
                T_Leaves...
            >::value;
    };
};

} // namespace llama

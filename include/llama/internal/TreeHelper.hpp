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

#include "../DatumCoord.hpp"
#include <tuple>
#include <type_traits>

namespace llama
{

namespace internal
{

template<
    std::size_t T_coord,
    typename... T_Leaves
>
struct GetLeave
{
    using type = typename std::tuple_element<
        T_coord,
        std::tuple< T_Leaves... >
    >::type;
};

template<
    typename T_Leave,
    typename... T_Leaves
>
struct GetSizeOfDatumStructLeaves;

template< typename T_Leave >
struct GetSizeOfDatumStructLeave
{
    static constexpr std::size_t value = sizeof( T_Leave );
};

template< typename... T_Leaves >
struct GetSizeOfDatumStructLeave< DatumStruct< T_Leaves... > >
{
    static constexpr std::size_t value =
        GetSizeOfDatumStructLeaves< T_Leaves... >::value;
};

template<
    typename T_Leave,
    typename... T_Leaves
>
struct GetSizeOfDatumStructLeaves
{
    static constexpr std::size_t value =
        GetSizeOfDatumStructLeave< T_Leave >::value +
        GetSizeOfDatumStructLeaves< T_Leaves... >::value;
};

template< typename T_Leave >
struct GetSizeOfDatumStructLeaves< T_Leave >
{
    static constexpr std::size_t value =
        GetSizeOfDatumStructLeave< T_Leave >::value;
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Leave,
    typename... T_Leaves
>
struct GetSizeOfDatumStructLeavesWithCoord;

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Leave
>
struct GetSizeOfDatumStructLeaveWithCoord
{
    static constexpr std::size_t value =
        sizeof( T_Leave ) * std::size_t( DatumCoordIsBigger<
            T_Coord,
            T_Pos
        >::value );
};

template<
    typename T_Coord,
    typename T_Pos,
    typename... T_Leaves
>
struct GetSizeOfDatumStructLeaveWithCoord<
    T_Coord,
    T_Pos,
    DatumStruct< T_Leaves... >
>
{
    static constexpr std::size_t value = GetSizeOfDatumStructLeavesWithCoord<
        T_Coord,
        typename T_Pos::template PushBack< 0 >,
        T_Leaves...
    >::value;
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Leave,
    typename... T_Leaves
>
struct GetSizeOfDatumStructLeavesWithCoord
{
    static constexpr std::size_t value = GetSizeOfDatumStructLeaveWithCoord<
            T_Coord,
            T_Pos,
            T_Leave
        >::value +
        GetSizeOfDatumStructLeavesWithCoord<
            T_Coord,
            typename T_Pos::IncBack,
            T_Leaves...
        >::value;
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Leave
>
struct GetSizeOfDatumStructLeavesWithCoord<
    T_Coord,
    T_Pos,
    T_Leave
>
{
    static constexpr std::size_t value = GetSizeOfDatumStructLeaveWithCoord<
        T_Coord,
        T_Pos,
        T_Leave
    >::value;
};

} // namespace internal

} // namespace llama

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

#include <tuple>
#include <type_traits>

namespace llama
{

    template<
        std::size_t T_coord,
        std::size_t... T_coords
    >
    struct DateCoord
    {
        static constexpr std::size_t front = T_coord;
        static constexpr std::size_t size = sizeof...( T_coords ) + 1;
        static constexpr std::size_t back = DateCoord< T_coords... >::back;
        using PopFront = DateCoord< T_coords... >;
        using IncBack = typename PopFront::IncBack::template PushFront< front >;
        template< std::size_t T_newCoord = 0 >
        using PushFront = DateCoord<
            T_newCoord,
            T_coord,
            T_coords...
        >;
        template< std::size_t T_newCoord = 0 >
        using PushBack = DateCoord<
            T_coord,
            T_coords...,
            T_newCoord
        >;
    };

    template< std::size_t T_coord >
    struct DateCoord < T_coord >
    {
        static constexpr std::size_t front = T_coord;
        static constexpr std::size_t back = T_coord;
        static constexpr std::size_t size = 1;
        using IncBack = DateCoord< T_coord + 1 >;
        template< std::size_t T_newCoord = 0 >
        using PushFront = DateCoord<
            T_newCoord,
            T_coord
        >;
        template< std::size_t T_newCoord = 0 >
        using PushBack = DateCoord<
            T_coord,
            T_newCoord
        >;
    };

} // namespace llama

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
		size_t coord,
		size_t... coords
	>
    struct DateCoord
    {
        static constexpr std::size_t front = coord;
        static constexpr std::size_t size = sizeof...( coords ) + 1;
        static constexpr std::size_t back = DateCoord< coords... >::back;
        using PopFront = DateCoord< coords... >;
        using IncBack = typename PopFront::IncBack::template PushFront< front >;
        template< std::size_t new_coord = 0 >
        using PushFront = DateCoord<
			new_coord,
			coord,
			coords...
		>;
        template< std::size_t new_coord = 0 >
        using PushBack = DateCoord<
			coord,
			coords...,
			new_coord
		>;
    };

    template< std::size_t coord >
    struct DateCoord < coord >
    {
        static constexpr std::size_t front = coord;
        static constexpr std::size_t back = coord;
        static constexpr std::size_t size = 1;
        using IncBack = DateCoord< coord + 1 >;
        template< std::size_t new_coord = 0 >
        using PushFront = DateCoord<
			new_coord,
			coord
		>;
        template< std::size_t new_coord = 0 >
        using PushBack = DateCoord<
			coord,
			new_coord
		>;
    };

} //namespace llama

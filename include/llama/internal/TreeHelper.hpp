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

#include "../DateCoord.hpp"
#include <tuple>
#include <type_traits>

namespace llama
{

namespace internal
{

    template<size_t coord,typename... Leaves>
    struct GetLeave
    {
        using type = typename std::tuple_element<
			coord,
			std::tuple< Leaves... >
		>::type;
    };

    template<
		typename Leave,
		typename... Leaves
	>
    struct GetSizeOfDateStructLeaves;

    template< typename Leave >
    struct GetSizeOfDateStructLeave
    {
        static constexpr std::size_t value = sizeof( Leave );
    };

    template< typename... Leaves >
    struct GetSizeOfDateStructLeave< DateStruct< Leaves... > >
    {
        static constexpr std::size_t value =
			GetSizeOfDateStructLeaves< Leaves... >::value;
    };

    template<
		typename Leave,
		typename... Leaves
	>
    struct GetSizeOfDateStructLeaves
    {
        static constexpr std::size_t value =
            GetSizeOfDateStructLeave< Leave >::value +
            GetSizeOfDateStructLeaves< Leaves... >::value;
    };

    template< typename Leave >
    struct GetSizeOfDateStructLeaves< Leave >
    {
        static constexpr std::size_t value =
			GetSizeOfDateStructLeave< Leave >::value;
    };

    template<
		typename First,
		typename Second,
		typename SFinae = void
	>
    struct CompareDateCoord;

    template<
		typename First,
		typename Second
	>
    struct CompareDateCoord<
		First,
		Second,
		typename std::enable_if<
			( First::size == 1 || Second::size == 1)
		>::type
	>
    {
        static constexpr bool isBigger = ( First::front > Second::front );
    };

    template<
		typename First,
		typename Second
	>
    struct CompareDateCoord<
		First,
		Second,
		typename std::enable_if<
			( First::size > 1 && Second::size > 1
			&& First::front == Second::front )
		>::type
	>
    {
        static constexpr bool isBigger = CompareDateCoord<
			typename First::PopFront,
			typename Second::PopFront
		>::isBigger;
    };

    template<
		typename First,
		typename Second
	>
    struct CompareDateCoord<
		First,
		Second,
		typename std::enable_if<
			( First::size > 1
			&& Second::size > 1
			&& First::front < Second::front )
		>::type
	>
    {
        static constexpr bool isBigger = false;
    };

    template<
		typename First,
		typename Second
	>
    struct CompareDateCoord<
		First,
		Second,
		typename std::enable_if<
			( First::size > 1
			&& Second::size > 1
			&& First::front > Second::front )
		>::type
	>
    {
        static constexpr bool isBigger = true;
    };

    template<
		typename Coord,
		typename Pos,
		typename Leave,
		typename... Leaves
	>
    struct GetSizeOfDateStructLeavesWithCoord;

    template<
		typename Coord,
		typename Pos,
		typename Leave
	>
    struct GetSizeOfDateStructLeaveWithCoord
    {
        static constexpr std::size_t value =
			sizeof( Leave ) * std::size_t( CompareDateCoord<
				Coord,
				Pos
			>::isBigger );
    };

    template<
		typename Coord,
		typename Pos,
		typename... Leaves
	>
    struct GetSizeOfDateStructLeaveWithCoord<
		Coord,
		Pos,
		DateStruct< Leaves... >
	>
    {
        static constexpr std::size_t value = GetSizeOfDateStructLeavesWithCoord<
            Coord,
            typename Pos::template PushBack< 0 >,
            Leaves...
        >::value;
    };

    template<
		typename Coord,
		typename Pos,
		typename Leave,
		typename... Leaves
	>
    struct GetSizeOfDateStructLeavesWithCoord
    {
        static constexpr std::size_t value = GetSizeOfDateStructLeaveWithCoord<
				Coord,
				Pos,
				Leave
			>::value +
            GetSizeOfDateStructLeavesWithCoord<
				Coord,
				typename Pos::IncBack,
				Leaves...
			>::value;
    };

    template<
		typename Coord,
		typename Pos,
		typename Leave
	>
    struct GetSizeOfDateStructLeavesWithCoord<
		Coord,
		Pos,
		Leave
	>
    {
        static constexpr std::size_t value = GetSizeOfDateStructLeaveWithCoord<
			Coord,
			Pos,
			Leave
		>::value;
    };

} //namespace internal

} //namespace llama

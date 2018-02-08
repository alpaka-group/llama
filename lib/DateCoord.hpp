#pragma once

#include <tuple>
#include <type_traits>

namespace llama
{

	template< size_t coord, size_t... coords >
	struct DateCoord
	{
		static constexpr size_t front = coord;
		static constexpr size_t size = sizeof...(coords) + 1;
		static constexpr size_t back = DateCoord<coords...>::back;
		using PopFront = DateCoord<coords...>;
		using IncBack = typename PopFront::IncBack::template PushFront<front>;
		template <size_t new_coord = 0>
		using PushFront = DateCoord<new_coord,coord,coords...>;
		template <size_t new_coord = 0>
		using PushBack = DateCoord<coord,coords...,new_coord>;
	};

	template< size_t coord >
	struct DateCoord < coord >
	{
		static constexpr size_t front = coord;
		static constexpr size_t back = coord;
		static constexpr size_t size = 1;
		using IncBack = DateCoord<coord+1>;
		template <size_t new_coord = 0>
		using PushFront = DateCoord<new_coord,coord>;
		template <size_t new_coord = 0>
		using PushBack = DateCoord<coord,new_coord>;
	};

} //namespace llama

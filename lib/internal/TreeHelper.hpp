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
		using type = typename std::tuple_element<coord, std::tuple<Leaves...>>::type;
	};

	template< typename Leave, typename... Leaves>
	struct GetSizeOfDateStructLeaves;

	template< typename Leave >
	struct GetSizeOfDateStructLeave
	{
		static constexpr size_t value = sizeof( Leave );
	};

	template< typename... Leaves >
	struct GetSizeOfDateStructLeave< DateStruct<Leaves...> >
	{
		static constexpr size_t value = GetSizeOfDateStructLeaves< Leaves... >::value;
	};

	template< typename Leave, typename... Leaves>
	struct GetSizeOfDateStructLeaves
	{
		static constexpr size_t value =
			GetSizeOfDateStructLeave< Leave >::value +
			GetSizeOfDateStructLeaves<Leaves...>::value;
	};

	template< typename Leave >
	struct GetSizeOfDateStructLeaves< Leave >
	{
		static constexpr size_t value = GetSizeOfDateStructLeave< Leave >::value;
	};

	template< typename First, typename Second, typename SFinae = void >
	struct CompareDateCoord;

	template< typename First, typename Second >
	struct CompareDateCoord<First,Second,typename std::enable_if<
		( First::size == 1 || Second::size == 1)
	>::type>
	{
		static constexpr bool isBigger = (First::front > Second::front);
	};

	//~ template< size_t first, size_t second >
	//~ struct CompareDateCoord<DateCoord<first>,DateCoord<second>, void >
	//~ {
		//~ static constexpr bool isBigger = (first > second);
	//~ };

	template< typename First, typename Second >
	struct CompareDateCoord<First,Second,typename std::enable_if<
		( First::size > 1 && Second::size > 1 && First::front == Second::front )
	>::type>
	{
		static constexpr bool isBigger =
			CompareDateCoord<typename First::PopFront,typename Second::PopFront>::isBigger;
	};

	template< typename First, typename Second >
	struct CompareDateCoord<First,Second,typename std::enable_if<
		( First::size > 1 && Second::size > 1 && First::front < Second::front )
	>::type>
	{
		static constexpr bool isBigger = false;
	};

	template< typename First, typename Second >
	struct CompareDateCoord<First,Second,typename std::enable_if<
		( First::size > 1 && Second::size > 1 && First::front > Second::front )
	>::type>
	{
		static constexpr bool isBigger = true;
	};

	template< typename Coord, typename Pos, typename Leave, typename... Leaves >
	struct GetSizeOfDateStructLeavesWithCoord;

	template< typename Coord, typename Pos, typename Leave >
	struct GetSizeOfDateStructLeaveWithCoord
	{
		static constexpr size_t value = sizeof( Leave ) * size_t(CompareDateCoord<Coord,Pos>::isBigger);
	};

	template< typename Coord, typename Pos, typename... Leaves >
	struct GetSizeOfDateStructLeaveWithCoord< Coord, Pos, DateStruct< Leaves...> >
	{
		static constexpr size_t value = GetSizeOfDateStructLeavesWithCoord<
			Coord,
			typename Pos::template PushBack<0>,
			Leaves...
		>::value;
	};

	template< typename Coord, typename Pos, typename Leave, typename... Leaves >
	struct GetSizeOfDateStructLeavesWithCoord
	{
		static constexpr size_t value =
			GetSizeOfDateStructLeaveWithCoord< Coord, Pos, Leave >::value +
			GetSizeOfDateStructLeavesWithCoord< Coord, typename Pos::IncBack, Leaves...>::value;
	};

	template< typename Coord, typename Pos, typename Leave >
	struct GetSizeOfDateStructLeavesWithCoord< Coord, Pos, Leave >
	{
		static constexpr size_t value = GetSizeOfDateStructLeaveWithCoord< Coord, Pos, Leave >::value;
	};

} //namespace internal

} //namespace llama

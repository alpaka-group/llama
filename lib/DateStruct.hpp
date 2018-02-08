#pragma once

#include "Types.hpp"
#include "internal/TreeHelper.hpp"

namespace llama
{

template <typename... Leaves>
struct DateStruct
{
	template<size_t coord>
	struct GetBranch
	{
		using type = typename internal::GetLeave<coord,Leaves...>::type;
	};
	static constexpr size_t size = internal::GetSizeOfDateStructLeaves<Leaves...>::value;
	template <size_t... coords>
	struct LinearBytePos
	{
		static constexpr size_t value =
			internal::GetSizeOfDateStructLeavesWithCoord<
				DateCoord<coords...>,
				DateCoord<0>,
				Leaves...
			>::value;
	};
};

} //namespace llama

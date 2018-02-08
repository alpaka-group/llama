#pragma once

#include "DateStruct.hpp"

namespace llama
{

namespace internal
{
	template< typename T, size_t count, typename... List >
	struct AddChildToStruct
	{
		using type = typename AddChildToStruct< T, count - 1, List..., T >::type;
	};
	template< typename T, typename... List >
	struct AddChildToStruct<T,0,List...>
	{
		using type = DateStruct< List... >;
	};

} //namespace internal

template< typename Child, size_t count >
using DateArray = typename internal::AddChildToStruct< Child, count >::type;

} //namespace llama

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

namespace llama
{

template< unsigned char... T_cs >
struct UniqueIdentifier
{
};

template<
	template< std::size_t > class T_Name,
	std::size_t T_i,
	std::size_t T_length,
	unsigned char... T_cs
>
struct MakeUniqueIdentifier
{
	using type = typename MakeUniqueIdentifier<
		T_Name,
		T_i + 1,
		T_length,
		T_Name< T_i >::value,
		T_cs...
	>::type;
};

template<
	template< std::size_t > class T_Name,
	std::size_t T_length,
	unsigned char... T_cs
>
struct MakeUniqueIdentifier
<
	T_Name,
	T_length,
	T_length,
	T_cs...
>
{
	using type = UniqueIdentifier< T_cs... >;
};

}; // namespace llama

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

#include <type_traits>
#include <boost/mp11.hpp>
#include "DatumStruct.hpp"

namespace llama
{

template<
    typename T_DatumDomain,
    typename T_DatumCoord,
    std::size_t T_pos,
    typename T_SFinae,
    typename... T_UID
>
struct GetCoordFromUIDImpl
{
	static_assert(
		boost::mp11::mp_size< T_DatumDomain >::value != 0,
		"UID combination is not valid"
	);
};

template<
    typename T_DatumDomain,
    typename T_DatumCoord,
    std::size_t T_pos,
    typename T_FirstUID,
    typename... T_UID
>
struct GetCoordFromUIDImpl<
	T_DatumDomain,
	T_DatumCoord,
	T_pos,
	typename std::enable_if<
		std::is_same<
			T_FirstUID,
			GetDatumElementUID< boost::mp11::mp_first< T_DatumDomain > >
		>::value
	>::type,
	T_FirstUID,
	T_UID...
>
{
	using type = typename GetCoordFromUIDImpl<
		GetDatumElementType< boost::mp11::mp_first< T_DatumDomain > >,
		typename T_DatumCoord::template PushBack< T_pos >,
		0,
		void,
		T_UID...
	>::type;
};

template<
    typename T_DatumDomain,
    typename T_DatumCoord,
    std::size_t T_pos,
    typename T_FirstUID,
    typename... T_UID
>
struct GetCoordFromUIDImpl<
	T_DatumDomain,
	T_DatumCoord,
	T_pos,
	typename std::enable_if<
		!std::is_same<
			T_FirstUID,
			GetDatumElementUID< boost::mp11::mp_first< T_DatumDomain > >
		>::value
	>::type,
	T_FirstUID,
	T_UID...
>
{
	using type = typename GetCoordFromUIDImpl<
		boost::mp11::mp_pop_front< T_DatumDomain >,
		T_DatumCoord,
		T_pos + 1,
		void,
		T_FirstUID,
		T_UID...
	>::type;
};

template<
    typename T_DatumDomain,
    typename T_DatumCoord,
    std::size_t T_pos
>
struct GetCoordFromUIDImpl<
	T_DatumDomain,
	T_DatumCoord,
	T_pos,
	void
>
{
	using type = T_DatumCoord;
};

template<
    typename T_DatumDomain,
    typename... T_UID
>
using GetCoordFromUID = typename GetCoordFromUIDImpl<
	T_DatumDomain,
    DatumCoord< >,
    0,
    void,
    T_UID...
>::type;

} // namespace llama

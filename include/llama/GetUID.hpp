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

#include <boost/mp11.hpp>
#include "DatumStruct.hpp"

namespace llama
{

namespace internal
{

template<
    typename T_DatumElement,
    std::size_t... T_datumDomainCoords
>
struct GetUIDImpl;

template<
    typename T_DatumElement,
    std::size_t T_firstDatumDomainCoord,
    std::size_t... T_datumDomainCoords
>
struct GetUIDImpl<
    T_DatumElement,
    T_firstDatumDomainCoord,
    T_datumDomainCoords...
>
{
    using _DateElement = boost::mp11::mp_at_c<
        GetDatumElementType< T_DatumElement >,
        T_firstDatumDomainCoord
    >;
    using type = typename GetUIDImpl<
        _DateElement,
        T_datumDomainCoords...
    >::type;
};

template<
    typename T_DatumElement
>
struct GetUIDImpl<
    T_DatumElement
>
{
    using type = GetDatumElementUID< T_DatumElement >;
};

template<
    typename T_DatumElement,
    typename T_DatumDomainCoord
>
struct GetUIDfromDatumCoord;

template<
    typename T_DatumElement,
    std::size_t... T_datumDomainCoords
>
struct GetUIDfromDatumCoord<
    T_DatumElement,
    DatumCoord< T_datumDomainCoords... >
>
{
	using type = typename GetUIDImpl<
		T_DatumElement,
		T_datumDomainCoords...
	>::type;
};

} // namespace internal

/** return the unique identifier of the \ref DatumElement at a \ref
 *  DatumCoord inside the datum domain tree.
 * \tparam T_DatumDomain the datum domain, probably of type \ref DatumStruct or
 *  \ref DatumArray
 * \tparam T_DatumCoord the datum coord, probably of type \ref DatumCoord
 * \return unique identifer type
 * */
template<
    typename T_DatumDomain,
    typename T_DatumCoord
>
using GetUID = typename internal::GetUIDfromDatumCoord<
    DatumElement<
		NoName,
		T_DatumDomain
	>,
    T_DatumCoord
>::type;

} // namespace llama

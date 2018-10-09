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
    typename T_DatumDomain,
    std::size_t... T_datumDomainCoords
>
struct GetTypeImpl;

template<
    typename T_DatumDomain,
    std::size_t T_firstDatumDomainCoord,
    std::size_t... T_datumDomainCoords
>
struct GetTypeImpl<
    T_DatumDomain,
    T_firstDatumDomainCoord,
    T_datumDomainCoords...
>
{
    using _DateElement = boost::mp11::mp_at_c<
        T_DatumDomain,
        T_firstDatumDomainCoord
    >;
    using type = typename GetTypeImpl<
        GetDatumElementType< _DateElement >,
        T_datumDomainCoords...
    >::type;
};

template<
    typename T_DatumDomain
>
struct GetTypeImpl<
    T_DatumDomain
>
{
    using type = T_DatumDomain;
};

} // namespace internal

/** Returns the type of a node in a datum domain tree for a coordinate given as
 *  tree index (like for \ref DatumCoord)
 * \tparam T_DatumDomain the datum domain (probably \ref DatumStruct)
 * \tparam T_datumDomainCoords... the coordinate
 * \return type at the specified node
 */
template<
    typename T_DatumDomain,
    std::size_t... T_datumDomainCoords
>
using GetType = typename internal::GetTypeImpl<
    T_DatumDomain,
    T_datumDomainCoords...
>::type;

namespace internal
{

template<
    typename T_DatumDomain,
    typename T_DatumCoord
>
struct GetTypeFromDatumCoordImpl;

template<
    typename T_DatumDomain,
    std::size_t... T_coords
>
struct GetTypeFromDatumCoordImpl<
    T_DatumDomain,
    DatumCoord< T_coords... >
>
{
    using type = GetType<
        T_DatumDomain,
        T_coords...
    >;
};

} // namespace internal

/** Returns the type of a node in a datum domain tree for a coordinate given as
 *  \ref DatumCoord
 * \tparam T_DatumDomain the datum domain (probably \ref DatumStruct)
 * \tparam T_DatumCoord the coordinate
 * \return type at the specified node
 */
template<
    typename T_DatumDomain,
    typename T_DatumCoord
>
using GetTypeFromDatumCoord = typename internal::GetTypeFromDatumCoordImpl<
    T_DatumDomain,
    T_DatumCoord
>::type;

} // namespace llama

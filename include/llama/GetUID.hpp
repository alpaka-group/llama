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

#include "DatumCoord.hpp"

namespace llama
{

template<
    typename T_DatumDomain,
    typename T_DatumCoord
>
struct GetUIDType;

template<
    typename T_DatumDomain,
    std::size_t... T_coords
>
struct GetUIDType
<
    T_DatumDomain,
    DatumCoord< T_coords... >
>
{
	using type = typename T_DatumDomain::Llama::template UID<
        void,
        T_coords...
    >::type;
};

template<
    typename T_DatumDomain,
    typename T_DatumCoordName
>
using GetUID = typename GetUIDType<
    T_DatumDomain,
    T_DatumCoordName
>::type;

template<
    typename T_DatumDomain,
    typename T_DatumCoordName
>
using GetUIDFromName = GetUID<
    T_DatumDomain,
    typename T_DatumCoordName::type
>;

} // namespace llama

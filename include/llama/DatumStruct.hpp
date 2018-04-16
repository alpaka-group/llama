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

#include "Types.hpp"
#include "DatumCoord.hpp"
#include <boost/mp11.hpp>
#include <type_traits>

namespace llama
{

template< typename T_DatumElement >
using GetDatumElementType = boost::mp11::mp_second< T_DatumElement >;

template< typename T_DatumElement >
using GetDatumElementUID = boost::mp11::mp_first< T_DatumElement >;

template<
    typename T_DatumDomain,
    typename T_DatumCoord,
    typename T_IterCoord
>
struct LinearBytePosImpl
{
    static constexpr std::size_t value = sizeof( T_DatumDomain )
        * std::size_t( DatumCoordIsBigger<
            T_DatumCoord,
            T_IterCoord
        >::value );
};

template<
    typename T_DatumCoord,
    typename T_IterCoord,
    typename T_FirstDatumElement,
    typename... T_DatumElements
>
struct LinearBytePosImpl<
    DatumStruct<
        T_FirstDatumElement,
        T_DatumElements...
    >,
    T_DatumCoord,
    T_IterCoord
>
{
    static constexpr std::size_t value =
        LinearBytePosImpl<
            GetDatumElementType< T_FirstDatumElement >,
            T_DatumCoord,
            typename T_IterCoord::template PushBack< 0 >
        >::value +
        LinearBytePosImpl<
            DatumStruct< T_DatumElements... >,
            T_DatumCoord,
            typename T_IterCoord::IncBack
        >::value;
};

template<
    typename T_DatumCoord,
    typename T_IterCoord
>
struct LinearBytePosImpl<
    DatumStruct< >,
    T_DatumCoord,
    T_IterCoord
>
{
    static constexpr std::size_t value = 0;

};

template<
    typename T_DatumDomain,
    std::size_t... T_coords
>
struct LinearBytePos
{
    static constexpr std::size_t value = LinearBytePosImpl<
        T_DatumDomain,
        DatumCoord< T_coords... >,
        DatumCoord< 0 >
    >::value;
};

template<
    typename T_DatumDomain,
    std::size_t T_coord
>
struct GetBranch
{
    using type = boost::mp11::mp_at_c<
        T_DatumDomain,
        T_coord
    >;
};

template< typename T_DatumDomain >
struct SizeOf
{
    static constexpr std::size_t value = sizeof( T_DatumDomain );
};

template<
    typename T_FirstDatumElement,
    typename... T_DatumElements
>
struct SizeOf<
    DatumStruct<
        T_FirstDatumElement,
        T_DatumElements...
    >
>
{
    static constexpr std::size_t value =
        SizeOf< GetDatumElementType< T_FirstDatumElement > >::value +
        SizeOf< DatumStruct< T_DatumElements... > >::value;
};

template< >
struct SizeOf< DatumStruct< > >
{
    static constexpr std::size_t value = 0;
};



} // namespace llama

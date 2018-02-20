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

#include <tuple>
#include <type_traits>

namespace llama
{

template < std::size_t... T_coords >
struct DateCoord;

namespace internal
{

template <
    std::size_t T_rest,
    std::size_t... T_coords
>
struct DateCoordFront;

template <
    std::size_t T_rest,
    std::size_t T_coord,
    std::size_t... T_coords
>
struct DateCoordFront<
    T_rest,
    T_coord,
    T_coords...
>
{
    using type = typename DateCoordFront<
        T_rest - 1,
        T_coords...
    >::type::template PushFront< T_coord >;
};

template <
    std::size_t T_coord,
    std::size_t... T_coords
>
struct DateCoordFront<
    0,
    T_coord,
    T_coords...
>
{
    using type = DateCoord< >;
};

template <
    std::size_t T_rest,
    std::size_t... T_coords
>
struct DateCoordBack;

template <
    std::size_t T_rest,
    std::size_t T_coord,
    std::size_t... T_coords
>
struct DateCoordBack<
    T_rest,
    T_coord,
    T_coords...
>
{
    using type = typename DateCoordBack<
        T_rest - 1,
        T_coords...
    >::type;
};

template <
    std::size_t T_coord,
    std::size_t... T_coords
>
struct DateCoordBack<
    0,
    T_coord,
    T_coords...
>
{
    using type = DateCoord< T_coord, T_coords... >;
};

template < >
struct DateCoordBack< 0 >
{
    using type = DateCoord< >;
};

} // namespace internal

template<
    std::size_t T_coord,
    std::size_t... T_coords
>
struct DateCoord< T_coord, T_coords... >
{
    static constexpr std::size_t front = T_coord;
    static constexpr std::size_t size = sizeof...( T_coords ) + 1;
    static constexpr std::size_t back = DateCoord< T_coords... >::back;
    using PopFront = DateCoord< T_coords... >;
    using IncBack = typename PopFront::IncBack::template PushFront< front >;
    template< std::size_t T_newCoord = 0 >
    using PushFront = DateCoord<
        T_newCoord,
        T_coord,
        T_coords...
    >;
    template< std::size_t T_newCoord = 0 >
    using PushBack = DateCoord<
        T_coord,
        T_coords...,
        T_newCoord
    >;
    template< std::size_t T_size >
    using Front = typename internal::DateCoordFront<
        T_size,
        T_coord,
        T_coords...
    >::type;
    template< std::size_t T_size >
    using Back = typename internal::DateCoordBack<
        size - T_size,
        T_coord,
        T_coords...
    >::type;
    template< typename T_Other >
    using Cat = typename DateCoord< T_coords... >::template
        Cat< T_Other >::template
            PushFront< T_coord >;
};

template< std::size_t T_coord >
struct DateCoord< T_coord >
{
    static constexpr std::size_t front = T_coord;
    static constexpr std::size_t back = T_coord;
    static constexpr std::size_t size = 1;
    using IncBack = DateCoord< T_coord + 1 >;
    template< std::size_t T_newCoord = 0 >
    using PushFront = DateCoord<
        T_newCoord,
        T_coord
    >;
    template< std::size_t T_newCoord = 0 >
    using PushBack = DateCoord<
        T_coord,
        T_newCoord
    >;
    template< std::size_t T_size >
    using Front = typename internal::DateCoordFront<
        T_size,
        T_coord
    >::type;
    template< std::size_t T_size >
    using Back = typename internal::DateCoordBack<
        size - T_size,
        T_coord
    >::type;
    template< typename T_Other >
    using Cat = typename T_Other::template PushFront< T_coord >;
};

template< >
struct DateCoord< >
{
    static constexpr std::size_t size = 0;
    using IncBack = DateCoord< 1 >;
    template< std::size_t T_newCoord = 0 >
    using PushFront = DateCoord< T_newCoord >;
    template< std::size_t T_newCoord = 0 >
    using PushBack = DateCoord< T_newCoord >;
    template< std::size_t T_size >
    using Front = DateCoord< >;
    template< std::size_t T_size >
    using Back = DateCoord< >;
    template< typename T_Other >
    using Cat = T_Other;
};

template<
    typename T_First,
    typename T_Second,
    typename T_SFinae = void
>
struct DateCoordIsBigger;

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsBigger<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size == 1 || T_Second::size == 1)
    >::type
>
{
    static constexpr bool value = ( T_First::front > T_Second::front );
};

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsBigger<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size > 1 && T_Second::size > 1
        && T_First::front == T_Second::front )
    >::type
>
{
    static constexpr bool value = DateCoordIsBigger<
        typename T_First::PopFront,
        typename T_Second::PopFront
    >::value;
};

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsBigger<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size > 1
        && T_Second::size > 1
        && T_First::front < T_Second::front )
    >::type
>
{
    static constexpr bool value = false;
};

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsBigger<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size > 1
        && T_Second::size > 1
        && T_First::front > T_Second::front )
    >::type
>
{
    static constexpr bool value = true;
};


template<
    typename T_First,
    typename T_Second,
    typename T_SFinae = void
>
struct DateCoordIsSame;

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsSame<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size < 1 || T_Second::size < 1)
    >::type
>
{
    static constexpr bool value = true;
};

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsSame<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size == 1 && T_Second::size >= 1) ||
        ( T_First::size >= 1 && T_Second::size == 1)
    >::type
>
{
    static constexpr bool value = ( T_First::front == T_Second::front );
};

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsSame<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size > 1 && T_Second::size > 1
        && T_First::front == T_Second::front )
    >::type
>
{
    static constexpr bool value = DateCoordIsSame<
        typename T_First::PopFront,
        typename T_Second::PopFront
    >::value;
};

template<
    typename T_First,
    typename T_Second
>
struct DateCoordIsSame<
    T_First,
    T_Second,
    typename std::enable_if<
        ( T_First::size > 1
        && T_Second::size > 1
        && T_First::front != T_Second::front )
    >::type
>
{
    static constexpr bool value = false;
};

} // namespace llama

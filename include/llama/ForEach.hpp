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

namespace llama
{

namespace internal
{

template<
    typename T_Pos,
    typename T_Coord,
    typename SFINAE = void
>
struct CoordIsIncluded;

template<
    typename T_Pos,
    typename T_Coord
>
struct CoordIsIncluded<
    T_Pos,
    T_Coord,
    typename std::enable_if< (T_Pos::size < T_Coord::size) >::type
>
{
    static constexpr bool value = false;
};

template<
    typename T_Pos,
    typename T_Coord
>
struct CoordIsIncluded<
    T_Pos,
    T_Coord,
    typename std::enable_if<
        (T_Pos::size >= T_Coord::size) &&
        DatumCoordIsSame<
            T_Pos,
            T_Coord
        >::value
    >::type
>
{
    static constexpr bool value = true;
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor,
    typename T_Leave,
    typename... T_Leaves
>
struct ApplyFunctorInDatumStructLeaves;


template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor,
    typename SFINAE = void
>
struct ApplyFunctorIfCoordIsIncluded
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor& functor )
    -> void
    { };
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor
>
struct ApplyFunctorIfCoordIsIncluded<
    T_Coord,
    T_Pos,
    T_Functor,
    typename std::enable_if<
        CoordIsIncluded<
            T_Pos,
            T_Coord
        >::value
    >::type
>
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor& functor )
    -> void
    {
        functor( T_Coord(), typename T_Pos::template Back< T_Pos::size - T_Coord::size >() );
    };
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor,
    typename T_Leave
>
struct ApplyFunctorInDatumStructLeave
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor&& functor )
    -> void
    {
        ApplyFunctorIfCoordIsIncluded<
            T_Coord,
            T_Pos,
            T_Functor
        >{}( std::forward<T_Functor>( functor ) );
    };
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor,
    typename... T_Leaves
>
struct ApplyFunctorInDatumStructLeave<
    T_Coord,
    T_Pos,
    T_Functor,
    DatumStruct< T_Leaves... >
>
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor&& functor )
    -> void
    {
        ApplyFunctorInDatumStructLeaves<
            T_Coord,
            typename T_Pos::template PushBack< 0 >,
            T_Functor,
            T_Leaves...
        >{}( std::forward<T_Functor>( functor ) );
    }
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor,
    typename T_Leave,
    typename... T_Leaves
>
struct ApplyFunctorInDatumStructLeaves
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor&& functor )
    -> void
    {
        ApplyFunctorInDatumStructLeave<
            T_Coord,
            T_Pos,
            T_Functor,
            T_Leave
        >{}( std::forward<T_Functor>( functor ) );
        ApplyFunctorInDatumStructLeaves<
            T_Coord,
            typename T_Pos::IncBack,
            T_Functor,
            T_Leaves...
        >{}( std::forward<T_Functor>( functor ) );
    }
};

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor,
    typename T_Leave
>
struct ApplyFunctorInDatumStructLeaves<
    T_Coord,
    T_Pos,
    T_Functor,
    T_Leave
>
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor&& functor )
    -> void
    {
        ApplyFunctorInDatumStructLeave<
            T_Coord,
            T_Pos,
            T_Functor,
            T_Leave
        >{}( std::forward<T_Functor>( functor ) );
    }
};

template<
    typename T_DatumCoord,
    typename T_Functor,
    typename... T_Leaves
>
LLAMA_FN_HOST_ACC_INLINE
void applyFunctorInDatumStruct(
    T_Functor&& functor,
    DatumStruct< T_Leaves... >
)
{
     ApplyFunctorInDatumStructLeaves<
        T_DatumCoord,
        DatumCoord< 0 >,
        T_Functor,
        T_Leaves...
    >{}( std::forward<T_Functor>( functor ) );
}

} // namespace internal

template<
    typename T_DatumDomain,
    typename T_DatumCoord = DatumCoord < >,
    typename T_Functor
>
LLAMA_FN_HOST_ACC_INLINE
void forEach( T_Functor&& functor )
{
    internal::applyFunctorInDatumStruct< T_DatumCoord >(
        std::forward<T_Functor>( functor ),
        typename T_DatumDomain::Llama::TypeTree()
    );
}

} // namespace llama


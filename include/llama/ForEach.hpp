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

#include "DatumCoord.hpp"
#include "GetCoordFromUID.hpp"

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
    typename... T_Leaves
>
struct ApplyFunctorForEachLeaveImpl;

template<
    typename T_Coord,
    typename T_Pos,
    typename T_Functor,
    typename T_Leave
>
struct ApplyFunctorForDatumDomainImpl
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
struct ApplyFunctorForDatumDomainImpl<
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
        ApplyFunctorForEachLeaveImpl<
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
struct ApplyFunctorForEachLeaveImpl<
    T_Coord,
    T_Pos,
    T_Functor,
    T_Leave,
    T_Leaves...
>
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor&& functor )
    -> void
    {
        ApplyFunctorForDatumDomainImpl<
            T_Coord,
            T_Pos,
            T_Functor,
            GetDatumElementType< T_Leave >
        >{}( std::forward<T_Functor>( functor ) );
        ApplyFunctorForEachLeaveImpl<
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
    typename T_Functor
>
struct ApplyFunctorForEachLeaveImpl<
    T_Coord,
    T_Pos,
    T_Functor
>
{
    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( T_Functor&& functor )
    -> void
    {}
};

template<
    typename T_DatumDomain,
    typename T_DatumCoord,
    typename T_Functor
>
struct ApplyFunctorForEachLeave;

template<
    typename T_DatumCoord,
    typename T_Functor,
    typename... T_Leaves
>
struct ApplyFunctorForEachLeave<
    DatumStruct< T_Leaves... >,
    T_DatumCoord,
    T_Functor
>
{
    LLAMA_FN_HOST_ACC_INLINE
    static void apply( T_Functor&& functor )
    {
         ApplyFunctorForEachLeaveImpl<
            T_DatumCoord,
            DatumCoord< 0 >,
            T_Functor,
            T_Leaves...
        >{}( std::forward<T_Functor>( functor ) );
    }
};

} // namespace internal

template<
    typename T_DatumDomain,
    typename T_DatumCoordOrFirstUID = DatumCoord < >,
    typename... T_RestUID
>
struct ForEach
{
    using T_DatumCoord = GetCoordFromUID<
        T_DatumDomain,
        T_DatumCoordOrFirstUID,
        T_RestUID...
    >;
    template< typename T_Functor >
    LLAMA_FN_HOST_ACC_INLINE
    static void apply( T_Functor&& functor )
    {
        internal::ApplyFunctorForEachLeave<
            T_DatumDomain,
            T_DatumCoord,
            T_Functor
        >::apply( std::forward<T_Functor>( functor ) );
    }
};

template<
    typename T_DatumDomain,
    std::size_t... T_coords
>
struct ForEach<
    T_DatumDomain,
    DatumCoord< T_coords... >
>
{
    template< typename T_Functor >
    LLAMA_FN_HOST_ACC_INLINE
    static void apply( T_Functor&& functor )
    {
        internal::ApplyFunctorForEachLeave<
            T_DatumDomain,
            DatumCoord< T_coords... >,
            T_Functor
        >::apply( std::forward<T_Functor>( functor ) );
    }
};


} // namespace llama


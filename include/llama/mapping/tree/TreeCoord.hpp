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

#include <cstddef>
#include <type_traits>
#include <string>

#include "TreeElement.hpp"
#include "../../Tuple.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

template<
    std::size_t T_compiletime = 0,
    typename T_RuntimeType = std::size_t
>
struct TreeCoordElement
{
    using CompileType = std::integral_constant< std::size_t, T_compiletime >;

    static constexpr CompileType compiletime = {};

    T_RuntimeType const runtime;

    LLAMA_FN_HOST_ACC_INLINE
    TreeCoordElement() : runtime( 0 ) {}

    LLAMA_FN_HOST_ACC_INLINE
    TreeCoordElement( T_RuntimeType const runtime ) : runtime( runtime ) {}
};

template<
    std::size_t T_compiletime,
    typename T_RuntimeType,
    T_RuntimeType T_runtime
>
struct TreeCoordElement<
    T_compiletime,
    std::integral_constant< T_RuntimeType, T_runtime >
>
{
    using RuntimeType = std::integral_constant< T_RuntimeType, T_runtime >;
    using CompileType = std::integral_constant< std::size_t, T_compiletime >;

    static constexpr CompileType compiletime = {};
    static constexpr RuntimeType runtime = {};

    LLAMA_FN_HOST_ACC_INLINE
    TreeCoordElement() = default;

    LLAMA_FN_HOST_ACC_INLINE
    TreeCoordElement( RuntimeType const ) {}
};

template<
    std::size_t T_compiletime = 0,
    std::size_t T_runtime = 0
>
using TreeCoordElementConst = TreeCoordElement<
    T_compiletime,
    std::integral_constant< std::size_t, T_runtime >
>;

namespace internal
{

template< std::size_t... T_coords >
struct TreeCoordFromCoords;

template<
    std::size_t T_firstCoord,
    std::size_t... T_coords
>
struct TreeCoordFromCoords<
    T_firstCoord,
    T_coords...
>
{
    using type = TupleCatType<
        Tuple< TreeCoordElementConst< T_firstCoord > >,
        typename TreeCoordFromCoords< T_coords...>::type
    >;
};

template< std::size_t T_lastCoord >
struct TreeCoordFromCoords< T_lastCoord >
{
    using type = Tuple< TreeCoordElementConst< T_lastCoord > >;
};

template< >
struct TreeCoordFromCoords< >
{
    using type = Tuple< >;
};

} // namespace internal

template< std::size_t... T_coords >
using TreeCoord = typename internal::TreeCoordFromCoords< T_coords... >::type;

namespace internal
{

template< typename T_TreeCoord >
struct TreeCoordToStringImpl
{
    auto
    operator()( const T_TreeCoord treeCoord )
    -> std::string
    {
        return
            std::to_string( treeCoord.first.runtime ) +
            std::string( ":" ) +
            std::to_string( treeCoord.first.compiletime ) +
            std::string( ", " ) +
            TreeCoordToStringImpl< typename T_TreeCoord::RestTuple >()
                ( treeCoord.rest );
    };
};

template< typename T_LastTreeElement >
struct TreeCoordToStringImpl< Tuple< T_LastTreeElement > >
{
    auto
    operator()( const Tuple< T_LastTreeElement > treeCoord )
    -> std::string
    {
        return
            std::to_string( treeCoord.first.runtime ) +
            std::string( ":" ) +
            std::to_string( treeCoord.first.compiletime );
    };
};

} // namespace internal;

template< typename T_TreeCoord >
auto
treeCoordToString( const T_TreeCoord treeCoord )
-> std::string
{
    return
        std::string( "[ " ) +
        internal::TreeCoordToStringImpl< T_TreeCoord >()( treeCoord ) +
        std::string( " ]" );
}


} // namespace tree

} // namespace mapping

} // namespace llama


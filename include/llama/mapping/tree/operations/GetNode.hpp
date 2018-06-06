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

#include "../TreeElement.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

namespace operations
{

namespace internal
{

template<
    typename T_Tree,
    typename T_TreeCoord
>
struct GetNode
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Tree const & tree ) const
    -> decltype(
        GetNode<
            GetTupleType<
                typename T_Tree::Type,
                decltype( T_TreeCoord::FirstElement::compiletime )::value
            >,
            typename T_TreeCoord::RestTuple
        >()(
            getTupleElement<
                decltype( T_TreeCoord::FirstElement::compiletime )::value
            > ( tree.childs )
        )
    )
    {
        return GetNode<
            GetTupleType<
                typename T_Tree::Type,
                decltype( T_TreeCoord::FirstElement::compiletime )::value
            >,
            typename T_TreeCoord::RestTuple
        >()(
            getTupleElement<
                decltype( T_TreeCoord::FirstElement::compiletime )::value
            > ( tree.childs )
        );
    }
};

template< typename T_Tree >
struct GetNode<
    T_Tree,
    Tuple< >
>
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Tree const & tree ) const
    -> T_Tree
    {
        return tree;
    }
};

} // namespace internal

template<
    typename T_TreeCoord,
    typename T_Tree
>
LLAMA_FN_HOST_ACC_INLINE
auto
getNode( T_Tree const & tree )
-> decltype(
    internal::GetNode<
        T_Tree,
        T_TreeCoord
    >()( tree )
)
{
    return internal::GetNode<
        T_Tree,
        T_TreeCoord
    >()( tree );
}

} // namespace functor

} // namespace tree

} // namespace mapping

} // namespace llama

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

#include "Reduce.hpp"
#include "Operations.hpp"
#include "GetBlobSize.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

namespace internal
{

template<
    typename T_Tree,
    std::size_t T_Pos
>
struct SummizeTreeSmallerPos
{
    using RestTree = TreeElement<
        typename T_Tree::Identifier,
        typename T_Tree::Type::RestTuple
    >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Tree const tree ) const
    -> std::size_t
    {
        return
            getTreeBlobSize( tree.childs.first ) +
            SummizeTreeSmallerPos<
                RestTree,
                T_Pos - 1
            >()(
                RestTree(
                    tree.count,
                    tree.childs.rest
                )
            );
    }
};

template< typename T_Tree >
struct SummizeTreeSmallerPos<
    T_Tree,
    0
>
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( const T_Tree tree ) const
    -> std::size_t
    {
        return 0;
    }
};


template<
    typename T_Tree,
    typename T_TreeCoord
>
struct GetTreeBlobByteImpl
{
    using SubTree = GetTupleType<
        typename T_Tree::Type,
        T_TreeCoord::FirstElement::compiletime
    >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tree const tree,
        T_TreeCoord const treeCoord
    ) const
    -> std::size_t
    {
        return
            getTreeBlobSize(
                TreeElement<
                    typename T_Tree::Identifier,
                    typename T_Tree::Type
                >(
                    treeCoord.first.runtime,
                    tree.childs
                )
            ) +
            SummizeTreeSmallerPos<
                T_Tree,
                T_TreeCoord::FirstElement::compiletime
            >()( tree ) +
            GetTreeBlobByteImpl<
                SubTree,
                typename T_TreeCoord::RestTuple
            >()(
                getTupleElement< T_TreeCoord::FirstElement::compiletime >(
                    tree.childs
                ),
                treeCoord.rest
            );
    }
};

template<
    typename T_Tree,
    typename T_LastCoord
>
struct GetTreeBlobByteImpl<
    T_Tree,
    Tuple< T_LastCoord >
>
{
    using TreeCoord = Tuple< T_LastCoord >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tree const tree,
        TreeCoord const treeCoord
    ) const
    -> std::size_t
    {
        return sizeof( typename T_Tree::Type ) * treeCoord.first.runtime;
    }
};

} // internal

template<
    typename T_Tree,
    typename T_TreeCoord
>
LLAMA_FN_HOST_ACC_INLINE
auto
getTreeBlobByte(
    T_Tree const tree,
    T_TreeCoord const treeCoord
)
-> std::size_t
{
    return internal::GetTreeBlobByteImpl<
        T_Tree,
        T_TreeCoord
    >()(
        tree,
        treeCoord
    );
}

} // namespace tree

} // namespace mapping

} // namespace llama


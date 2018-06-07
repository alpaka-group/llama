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
struct SummarizeTreeSmallerPos
{
    using RestTree = typename TreePopFrontChild< T_Tree >::ResultType;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        typename T_Tree::Type const & childs,
        decltype( T_Tree::count ) const & count
    ) const
    -> std::size_t
    {
        return
            getTreeBlobSize( childs.first ) +
            SummarizeTreeSmallerPos<
                RestTree,
                T_Pos - 1
            >()(
                childs.rest,
                count
            );
    }
};

template< typename T_Tree >
struct SummarizeTreeSmallerPos<
    T_Tree,
    0
>
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        typename T_Tree::Type const & childs,
        decltype( T_Tree::count ) const & count
    ) const
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
        T_Tree const & tree,
        T_TreeCoord const & treeCoord
    ) const
    -> std::size_t
    {
        return
            getTreeBlobSize(
                tree.childs,
                // cuda doesn't like references to static members of they are
                // not defined somewhere although only type informations
                // are used which is the case for runtime=std::integral_constant
                decltype(treeCoord.first.runtime)(treeCoord.first.runtime)
            ) +
            SummarizeTreeSmallerPos<
                T_Tree,
                T_TreeCoord::FirstElement::compiletime
            >()(
                tree.childs,
                tree.count
            ) +
            GetTreeBlobByteImpl<
                SubTree,
                typename T_TreeCoord::RestTuple
            >()(
                // For some reason I have to call the internal function by hand
                // for the cuda nvcc compiler
                //~ getTupleElementRef< T_TreeCoord::FirstElement::compiletime >(
                llama::internal::GetTupleElementImpl<
                    typename T_Tree::Type,
                    T_TreeCoord::FirstElement::compiletime
                >()(
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
        T_Tree const & tree,
        TreeCoord const & treeCoord
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
    T_Tree const & tree,
    T_TreeCoord const & treeCoord
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


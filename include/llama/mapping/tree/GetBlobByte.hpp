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

#include "GetBlobSize.hpp"
#include "Operations.hpp"

namespace llama::mapping::tree
{
    namespace internal
    {
        template<typename Tree, std::size_t Pos>
        LLAMA_FN_HOST_ACC_INLINE auto summarizeTreeSmallerPos(
            typename Tree::Type const & childs,
            decltype(Tree::count) const & count) -> std::size_t
        {
            if constexpr(Pos == 0)
                return 0;
            else
                return getTreeBlobSize(childs.first)
                    + summarizeTreeSmallerPos<
                           typename TreePopFrontChild<Tree>::ResultType,
                           Pos - 1>(childs.rest, count);
        }

        template<typename Tree, typename LastCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobByteImpl(
            const Tree & tree,
            const Tuple<LastCoord> & treeCoord) -> std::size_t
        {
            return sizeof(typename Tree::Type) * treeCoord.first.runtime;
        }

        template<typename Tree, typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobByteImpl(const Tree & tree, const TreeCoord & treeCoord)
            -> std::size_t
        {
            return getTreeBlobSize(
                       tree.childs, LLAMA_DEREFERENCE(treeCoord.first.runtime))
                + summarizeTreeSmallerPos<
                       Tree,
                       decltype(TreeCoord::FirstElement::compiletime)::value>(
                       tree.childs, LLAMA_DEREFERENCE(tree.count))
                + getTreeBlobByteImpl(
                       getTupleElementRef<TreeCoord::FirstElement::compiletime>(
                           tree.childs),
                       treeCoord.rest);
        }
    }

    template<typename Tree, typename TreeCoord>
    LLAMA_FN_HOST_ACC_INLINE auto
    getTreeBlobByte(const Tree & tree, const TreeCoord & treeCoord)
        -> std::size_t
    {
        return internal::getTreeBlobByteImpl(tree, treeCoord);
    }
}

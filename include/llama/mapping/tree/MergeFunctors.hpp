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

#include "TreeElement.hpp"

namespace llama::mapping::tree
{
    template<typename Tree, typename TreeOperationList>
    struct MergeFunctors
    {};

    template<
        typename Tree,
        typename FirstOperation,
        typename... RestOperations>
    struct MergeFunctors<Tree, Tuple<FirstOperation, RestOperations...>>
    {
        const FirstOperation operation = {};
        const decltype(operation.basicToResult(Tree())) treeAfterOp;
        const MergeFunctors<decltype(treeAfterOp), Tuple<RestOperations...>>
            subMergeFunctorsImpl = {};

        MergeFunctors() = default;

        LLAMA_FN_HOST_ACC_INLINE
        MergeFunctors(
            Tree const & tree,
            const Tuple<FirstOperation, RestOperations...> &
                treeOperationList) :
                operation(treeOperationList.first),
                treeAfterOp(operation.basicToResult(tree)),
                subMergeFunctorsImpl(treeAfterOp, treeOperationList.rest)
        {}

        LLAMA_FN_HOST_ACC_INLINE
        auto basicToResult(Tree const &) const
        {
            return subMergeFunctorsImpl.basicToResult(treeAfterOp);
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            TreeCoord const & basicCoord,
            Tree const & tree) const
        {
            return subMergeFunctorsImpl.basicCoordToResultCoord(
                operation.basicCoordToResultCoord(basicCoord, tree),
                treeAfterOp);
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            TreeCoord const & resultCoord,
            Tree const & tree) const
        {
            return subMergeFunctorsImpl.resultCoordToBasicCoord(
                operation.resultCoordToBasicCoord(resultCoord, tree),
                operation.basicToResult(tree));
        }
    };

    template<typename Tree, typename LastOperation>
    struct MergeFunctors<Tree, Tuple<LastOperation>>
    {
        const LastOperation operation = {};

        MergeFunctors() = default;

        LLAMA_FN_HOST_ACC_INLINE
        MergeFunctors(
            const Tree &,
            const Tuple<LastOperation> & treeOperationList) :
                operation(treeOperationList.first)
        {}

        LLAMA_FN_HOST_ACC_INLINE
        auto basicToResult(const Tree & tree) const
        {
            return operation.basicToResult(tree);
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            TreeCoord const & basicCoord,
            Tree const & tree) const
        {
            return operation.basicCoordToResultCoord(basicCoord, tree);
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            TreeCoord const & resultCoord,
            Tree const & tree) const
        {
            return operation.resultCoordToBasicCoord(resultCoord, tree);
        }
    };

    template<typename Tree>
    struct MergeFunctors<Tree, Tuple<>>
    {
        using TreeOperationList = Tuple<>;

        MergeFunctors() = default;

        LLAMA_FN_HOST_ACC_INLINE
        MergeFunctors(
            Tree const &,
            TreeOperationList const & treeOperationList)
        {}

        LLAMA_FN_HOST_ACC_INLINE
        auto basicToResult(const Tree & tree) const
        {
            return tree;
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            TreeCoord const & basicCoord,
            Tree const & tree) const -> TreeCoord
        {
            return basicCoord;
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            TreeCoord const & resultCoord,
            Tree const & tree) const -> TreeCoord
        {
            return resultCoord;
        }
    };
}

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
    namespace internal
    {
        template<typename T_Tree, typename T_TreeOperationList>
        struct MergeFunctorsImpl
        {};

        template<
            typename T_Tree,
            typename FirstOperation,
            typename... RestOperations>
        struct MergeFunctorsImpl<
            T_Tree,
            Tuple<FirstOperation, RestOperations...>>
        {
            using SubMergeFunctorsImpl = MergeFunctorsImpl<
                typename FirstOperation::template Result<T_Tree>,
                Tuple<RestOperations...>>;
            using Result = typename SubMergeFunctorsImpl::Result;

            const FirstOperation operation;
            const decltype(operation.basicToResult(T_Tree())) treeAfterOp;
            const SubMergeFunctorsImpl subMergeFunctorsImpl;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctorsImpl(
                T_Tree const & tree,
                const Tuple<FirstOperation, RestOperations...> &
                    treeOperationList) :
                    operation(treeOperationList.first),
                    treeAfterOp(operation.basicToResult(tree)),
                    subMergeFunctorsImpl(treeAfterOp, treeOperationList.rest)
            {}

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(T_Tree const &) const -> Result
            {
                return subMergeFunctorsImpl.basicToResult(treeAfterOp);
            }

            template<typename T_TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
                T_TreeCoord const & basicCoord,
                T_Tree const & tree) const
            {
                return subMergeFunctorsImpl.basicCoordToResultCoord(
                    operation.basicCoordToResultCoord(basicCoord, tree),
                    treeAfterOp);
            }

            template<typename T_TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
                T_TreeCoord const & resultCoord,
                T_Tree const & tree) const
            {
                return subMergeFunctorsImpl.resultCoordToBasicCoord(
                    operation.resultCoordToBasicCoord(resultCoord, tree),
                    operation.basicToResult(tree));
            }
        };

        template<typename T_Tree, typename T_LastOperation>
        struct MergeFunctorsImpl<T_Tree, Tuple<T_LastOperation>>
        {
            using LastOp = T_LastOperation;
            using Result = typename LastOp::template Result<T_Tree>;
            using TreeOperationList = Tuple<T_LastOperation>;

            const LastOp operation;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctorsImpl(
                T_Tree const &,
                TreeOperationList const & treeOperationList) :
                    operation(treeOperationList.first)
            {}

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const T_Tree & tree) const -> Result
            {
                return operation.basicToResult(tree);
            }

            template<typename T_TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
                T_TreeCoord const & basicCoord,
                T_Tree const & tree) const
            {
                return operation.basicCoordToResultCoord(basicCoord, tree);
            }

            template<typename T_TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
                T_TreeCoord const & resultCoord,
                T_Tree const & tree) const
            {
                return operation.resultCoordToBasicCoord(resultCoord, tree);
            }
        };

        template<typename T_Tree>
        struct MergeFunctorsImpl<T_Tree, Tuple<>>
        {
            using Result = T_Tree;
            using TreeOperationList = Tuple<>;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctorsImpl(
                T_Tree const &,
                TreeOperationList const & treeOperationList)
            {}

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const T_Tree & tree) const -> Result
            {
                return tree;
            }

            template<typename T_TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
                T_TreeCoord const & basicCoord,
                T_Tree const & tree) const -> T_TreeCoord
            {
                return basicCoord;
            }

            template<typename T_TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
                T_TreeCoord const & resultCoord,
                T_Tree const & tree) const -> T_TreeCoord
            {
                return resultCoord;
            }
        };
    }

    template<typename T_Tree, typename T_TreeOperationList>
    struct MergeFunctors
    {
        using MergeFunctorsImpl =
            typename internal::MergeFunctorsImpl<T_Tree, T_TreeOperationList>;
        using Result = typename MergeFunctorsImpl::Result;

        MergeFunctorsImpl const mergeFunctorsImpl;

        LLAMA_FN_HOST_ACC_INLINE
        MergeFunctors(
            T_Tree const & tree,
            const T_TreeOperationList & treeOperationList) :
                mergeFunctorsImpl(tree, treeOperationList)
        {}

        LLAMA_FN_HOST_ACC_INLINE
        auto basicToResult(T_Tree const & tree) const -> Result
        {
            return mergeFunctorsImpl.basicToResult(tree);
        }

        template<typename T_TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            T_TreeCoord const & basicCoord,
            T_Tree const & tree) const
        {
            return mergeFunctorsImpl.basicCoordToResultCoord(basicCoord, tree);
        }

        template<typename T_TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            T_TreeCoord const & resultCoord,
            T_Tree const & tree) const
        {
            return mergeFunctorsImpl.resultCoordToBasicCoord(resultCoord, tree);
        }
    };
}

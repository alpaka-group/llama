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

    template<typename... Args>
    auto rest(Tuple<Args...> t)
    {
        return t.rest;
    }

    template<typename T>
    auto rest(Tuple<T> t) -> Tuple<>
    {
        return {};
    }

    template<typename Tree, typename... Operations>
    struct MergeFunctors<Tree, Tuple<Operations...>>
    {
        boost::mp11::mp_first<Tuple<Operations...>> operation = {};
        using ResultTree = decltype(operation.basicToResult(Tree()));
        ResultTree treeAfterOp;
        MergeFunctors<
            ResultTree,
            boost::mp11::mp_drop_c<Tuple<Operations...>, 1>>
            next = {};

        MergeFunctors() = default;

        LLAMA_FN_HOST_ACC_INLINE
        MergeFunctors(
            const Tree & tree,
            const Tuple<Operations...> & treeOperationList) :
                operation(treeOperationList.first),
                treeAfterOp(operation.basicToResult(tree)),
                next(treeAfterOp, rest(treeOperationList))
        {}

        LLAMA_FN_HOST_ACC_INLINE
        auto basicToResult(const Tree & tree) const
        {
            if constexpr(sizeof...(Operations) > 1)
                return next.basicToResult(treeAfterOp);
            else if constexpr(sizeof...(Operations) == 1)
                return operation.basicToResult(tree);
            else
                return tree;
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            const TreeCoord & basicCoord,
            const Tree & tree) const
        {
            if constexpr(sizeof...(Operations) >= 1)
                return next.basicCoordToResultCoord(
                    operation.basicCoordToResultCoord(basicCoord, tree),
                    treeAfterOp);
            else
                return basicCoord;
        }

        template<typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            const TreeCoord & resultCoord,
            const Tree & tree) const
        {
            if constexpr(sizeof...(Operations) >= 1)
                return next.resultCoordToBasicCoord(
                    operation.resultCoordToBasicCoord(resultCoord, tree),
                    operation.basicToResult(tree));
            else
                return resultCoord;
        }
    };

    template<typename Tree>
    struct MergeFunctors<Tree, Tuple<>>
    {
        MergeFunctors() = default;

        LLAMA_FN_HOST_ACC_INLINE
        MergeFunctors(const Tree &, const Tuple<> & treeOperationList) {}

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

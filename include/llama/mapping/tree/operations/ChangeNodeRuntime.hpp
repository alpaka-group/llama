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

#include "../../../Tuple.hpp"
#include "../TreeElement.hpp"
#include "GetNode.hpp"

namespace llama::mapping::tree::operations
{
    template<typename T_TreeCoord, typename T_Tree>
    LLAMA_FN_HOST_ACC_INLINE auto
    changeNodeRuntime(const T_Tree & tree, std::size_t newValue)
    {
        if constexpr(std::is_same_v<T_TreeCoord, Tuple<>>)
        {
            if constexpr(HasChildren<T_Tree>::value)
                return TreeElement<
                    typename T_Tree::Identifier,
                    typename T_Tree::Type>{newValue, tree.childs};
            else
                return T_Tree{newValue};
        }
        else
        {
            auto current
                = getTupleElement<T_TreeCoord::FirstElement::compiletime>(
                    tree.childs);
            auto replacement
                = changeNodeRuntime<typename T_TreeCoord::RestTuple>(
                    current, newValue);
            auto children
                = tupleReplace<T_TreeCoord::FirstElement::compiletime>(
                    tree.childs, replacement);
            return TreeElement<typename T_Tree::Identifier, decltype(children)>(
                tree.count, children);
        }
    }

    namespace internal
    {
        template<template<typename, typename> typename T_Operation>
        struct ChangeNodeChildsRuntimeFunctor
        {
            const std::size_t newValue;

            template<typename T_Element>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(T_Element element) const
            {
                if constexpr(HasChildren<T_Element>::value)
                {
                    return TreeElement<
                        typename T_Element::Identifier,
                        typename T_Element::Type>(
                        T_Operation<decltype(element.count), std::size_t>::
                            apply(element.count, newValue),
                        element.childs);
                }
                else
                {
                    const auto newCount
                        = T_Operation<decltype(element.count), std::size_t>::
                            apply(element.count, newValue);
                    return TreeElement<
                        typename T_Element::Identifier,
                        typename T_Element::Type>{newCount};
                }
            }
        };
    }

    template<
        typename T_TreeCoord,
        template<typename, typename>
        typename T_Operation,
        typename T_Tree>
    LLAMA_FN_HOST_ACC_INLINE auto
    changeNodeChildsRuntime(T_Tree const & tree, std::size_t const newValue)
    {
        if constexpr(HasChildren<T_Tree>::value)
        {
            if constexpr(std::is_same_v<T_TreeCoord, Tuple<>>)
            {
                auto children = tupleTransform(
                    tree.childs,
                    internal::ChangeNodeChildsRuntimeFunctor<T_Operation>{
                        newValue});
                return TreeElement<
                    typename T_Tree::Identifier,
                    decltype(children)>(tree.count, children);
            }
            else
            {
                auto current
                    = getTupleElement<T_TreeCoord::FirstElement::compiletime>(
                        tree.childs);
                auto replacement = changeNodeChildsRuntime<
                    typename T_TreeCoord::RestTuple,
                    T_Operation>(current, newValue);
                auto children
                    = tupleReplace<T_TreeCoord::FirstElement::compiletime>(
                        tree.childs, replacement);
                return TreeElement<
                    typename T_Tree::Identifier,
                    decltype(children)>(tree.count, children);
            }
        }
        else
            return tree;
    }
}

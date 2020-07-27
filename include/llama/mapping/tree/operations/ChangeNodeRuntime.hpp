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
                        typename T_TreeCoord,
                        typename T_SFINAE = void>
                    struct ChangeNodeRuntime
                    {
                        using ResultChilds
                            = decltype(tupleReplace<
                                       T_TreeCoord::FirstElement::compiletime>(
                                T_Tree().childs,
                                ChangeNodeRuntime<
                                    GetTupleType<
                                        typename T_Tree::Type,
                                        T_TreeCoord::FirstElement::compiletime>,
                                    typename T_TreeCoord::RestTuple>()(
                                    getTupleElement<
                                        T_TreeCoord::FirstElement::compiletime>(
                                        T_Tree().childs),
                                    0)));
                        using ResultTree = TreeElement<
                            typename T_Tree::Identifier,
                            ResultChilds>;

                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_Tree const & tree,
                            std::size_t const newValue) const -> ResultTree
                        {
                            return ResultTree(
                                tree.count,
                                tupleReplace<
                                    T_TreeCoord::FirstElement::compiletime>(
                                    tree.childs,
                                    ChangeNodeRuntime<
                                        GetTupleType<
                                            typename T_Tree::Type,
                                            T_TreeCoord::FirstElement::
                                                compiletime>,
                                        typename T_TreeCoord::RestTuple>()(
                                        getTupleElement<
                                            T_TreeCoord::FirstElement::
                                                compiletime>(tree.childs),
                                        newValue)));
                        }
                    };

                    // Leaf case
                    template<typename T_Tree>
                    struct ChangeNodeRuntime<
                        T_Tree,
                        Tuple<>,
                        typename T_Tree::IsTreeElementWithoutChilds>
                    {
                        using ResultTree = TreeElement<
                            typename T_Tree::Identifier,
                            typename T_Tree::Type>;

                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            ResultTree const & tree,
                            std::size_t const newValue) const -> T_Tree
                        {
                            return ResultTree(newValue);
                        }
                    };

                    // Node case
                    template<typename T_Tree>
                    struct ChangeNodeRuntime<
                        T_Tree,
                        Tuple<>,
                        typename T_Tree::IsTreeElementWithChilds>
                    {
                        using ResultTree = TreeElement<
                            typename T_Tree::Identifier,
                            typename T_Tree::Type>;

                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_Tree const & tree,
                            std::size_t const newValue) const -> ResultTree
                        {
                            return ResultTree(newValue, tree.childs);
                        }
                    };

                } // namespace internal

                template<typename T_TreeCoord, typename T_Tree>
                LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
                    T_Tree const & tree,
                    std::size_t const newValue)
                    -> decltype(
                        internal::ChangeNodeRuntime<T_Tree, T_TreeCoord>()(
                            tree,
                            newValue))
                {
                    return internal::ChangeNodeRuntime<T_Tree, T_TreeCoord>()(
                        tree, newValue);
                }

                namespace internal
                {
                    template<
                        typename T_Tree,
                        template<class, class>
                        class T_Operation,
                        typename T_TreeCoord,
                        typename T_SFINAE = void>
                    struct ChangeNodeChildsRuntime;

                    template<
                        typename T_Tree,
                        template<class, class>
                        class T_Operation,
                        typename T_TreeCoord>
                    struct ChangeNodeChildsRuntime<
                        T_Tree,
                        T_Operation,
                        T_TreeCoord,
                        typename T_Tree::IsTreeElementWithChilds>
                    {
                        using ResultType = TreeElement<
                            typename T_Tree::Identifier,
                            decltype(tupleReplace<
                                     T_TreeCoord::FirstElement::compiletime>(
                                T_Tree().childs,
                                ChangeNodeChildsRuntime<
                                    GetTupleType<
                                        typename T_Tree::Type,
                                        T_TreeCoord::FirstElement::compiletime>,
                                    T_Operation,
                                    typename T_TreeCoord::RestTuple>()(
                                    getTupleElement<
                                        T_TreeCoord::FirstElement::compiletime>(
                                        T_Tree().childs),
                                    0)))>;

                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_Tree const & tree,
                            std::size_t const newValue) const -> ResultType
                        {
                            return ResultType(
                                tree.count,
                                tupleReplace<
                                    T_TreeCoord::FirstElement::compiletime>(
                                    tree.childs,
                                    ChangeNodeChildsRuntime<
                                        GetTupleType<
                                            typename T_Tree::Type,
                                            T_TreeCoord::FirstElement::
                                                compiletime>,
                                        T_Operation,
                                        typename T_TreeCoord::RestTuple>()(
                                        getTupleElement<
                                            T_TreeCoord::FirstElement::
                                                compiletime>(tree.childs),
                                        newValue)));
                        }
                    };

                    // Leaf case
                    template<
                        typename T_Tree,
                        template<class, class>
                        class T_Operation>
                    struct ChangeNodeChildsRuntime<
                        T_Tree,
                        T_Operation,
                        Tuple<>,
                        typename T_Tree::IsTreeElementWithoutChilds>
                    {
                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_Tree const & tree,
                            std::size_t const newValue) const -> T_Tree
                        {
                            return tree;
                        }
                    };

                    template<template<class, class> class T_Operation>
                    struct ChangeNodeChildsRuntimeFunctor
                    {
                        std::size_t const newValue;

                        template<typename T_Element, typename T_SFINAE = void>
                        struct OperatorSpecialization;

                        // Leaf case
                        template<typename T_Element>
                        struct OperatorSpecialization<
                            T_Element,
                            typename T_Element::IsTreeElementWithoutChilds>
                        {
                            using ResultElement = TreeElement<
                                typename T_Element::Identifier,
                                typename T_Element::Type>;

                            LLAMA_FN_HOST_ACC_INLINE
                            auto operator()(
                                std::size_t const newValue,
                                T_Element const element) const -> ResultElement
                            {
                                return ResultElement{
                                    T_Operation<
                                        decltype(element.count),
                                        std::size_t>::
                                        apply(element.count, newValue)};
                            }
                        };

                        // Node case
                        template<typename T_Element>
                        struct OperatorSpecialization<
                            T_Element,
                            typename T_Element::IsTreeElementWithChilds>
                        {
                            using ResultElement = TreeElement<
                                typename T_Element::Identifier,
                                typename T_Element::Type>;

                            LLAMA_FN_HOST_ACC_INLINE
                            auto operator()(
                                std::size_t const newValue,
                                T_Element const element) const -> ResultElement
                            {
                                return ResultElement(
                                    T_Operation<
                                        decltype(element.count),
                                        std::size_t>::
                                        apply(element.count, newValue),
                                    element.childs);
                            }
                        };

                        template<typename T_Element>
                        LLAMA_FN_HOST_ACC_INLINE auto
                        operator()(T_Element const element) const
                            -> decltype(OperatorSpecialization<T_Element>()(
                                newValue,
                                element))
                        {
                            return OperatorSpecialization<T_Element>()(
                                newValue, element);
                        }
                    };

                    // Node case
                    template<
                        typename T_Tree,
                        template<class, class>
                        class T_Operation>
                    struct ChangeNodeChildsRuntime<
                        T_Tree,
                        T_Operation,
                        Tuple<>,
                        typename T_Tree::IsTreeElementWithChilds>
                    {
                        using ResultType = TreeElement<
                            typename T_Tree::Identifier,
                            decltype(tupleTransform(
                                T_Tree().childs,
                                ChangeNodeChildsRuntimeFunctor<T_Operation>{
                                    0}))>;

                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_Tree const & tree,
                            std::size_t const newValue) const -> ResultType
                        {
                            ChangeNodeChildsRuntimeFunctor<T_Operation> const
                                functor{newValue};
                            return ResultType(
                                tree.count,
                                tupleTransform(tree.childs, functor));
                        }
                    };

                } // namespace internal

                template<
                    typename T_TreeCoord,
                    template<class, class>
                    class T_Operation,
                    typename T_Tree>
                LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
                    T_Tree const & tree,
                    std::size_t const newValue)
                    -> decltype(internal::ChangeNodeChildsRuntime<
                                T_Tree,
                                T_Operation,
                                T_TreeCoord>()(tree, newValue))
                {
                    return internal::ChangeNodeChildsRuntime<
                        T_Tree,
                        T_Operation,
                        T_TreeCoord>()(tree, newValue);
                }

            } // namespace functor

        } // namespace tree

    } // namespace mapping

} // namespace llama

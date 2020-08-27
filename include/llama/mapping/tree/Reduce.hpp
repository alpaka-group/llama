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
    template<
        typename Tree,
        class InnerOp,
        class OuterOp,
        template<class, class>
        class LeafFunctor,
        bool HC = HasChildren<Tree>::value>
    struct Reduce;

    namespace internal
    {
        // Leaf
        template<
            typename Tree,
            class InnerOp,
            class OuterOp,
            template<class, class>
            class LeafFunctor,
            typename SFINAE = void>
        struct ReduceElementType
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(const decltype(Tree::count) & count) const
                -> std::size_t
            {
                return LeafFunctor<
                    typename Tree::Type,
                    decltype(Tree::count)>()(count);
            }
        };

        // Node
        template<
            typename Tree,
            class InnerOp,
            class OuterOp,
            template<class, class>
            class LeafFunctor>
        struct ReduceElementType<
            Tree,
            InnerOp,
            OuterOp,
            LeafFunctor,
            std::enable_if_t<(SizeOfTuple<typename Tree::Type>::value > 1)>>
        {
            using IterTree = typename TreePopFrontChild<Tree>::ResultType;

            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(
                typename Tree::Type const & childs,
                decltype(Tree::count) const & count) const -> std::size_t
            {
                return InnerOp{}(
                    Reduce<
                        typename Tree::Type::FirstElement,
                        InnerOp,
                        OuterOp,
                        LeafFunctor>()(childs.first),
                    internal::ReduceElementType<
                        IterTree,
                        InnerOp,
                        OuterOp,
                        LeafFunctor>()(childs.rest, count));
            }
        };

        // Node with one (last) child
        template<
            typename Tree,
            class InnerOp,
            class OuterOp,
            template<class, class>
            class LeafFunctor>
        struct ReduceElementType<
            Tree,
            InnerOp,
            OuterOp,
            LeafFunctor,
            std::enable_if_t<SizeOfTuple<typename Tree::Type>::value == 1>>
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(
                typename Tree::Type const & childs,
                decltype(Tree::count) const & count) const -> std::size_t
            {
                return Reduce<
                    typename Tree::Type::FirstElement,
                    InnerOp,
                    OuterOp,
                    LeafFunctor>()(childs.first);
            }
        };
    }

    template<
        typename Tree,
        class InnerOp,
        class OuterOp,
        template<class, class>
        class LeafFunctor,
        bool HC>
    struct Reduce
    {
        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(
            typename Tree::Type const & childs,
            decltype(Tree::count) const & count) const -> std::size_t
        {
            return OuterOp{}(
                count,
                internal::
                    ReduceElementType<Tree, InnerOp, OuterOp, LeafFunctor>()(
                        childs, count));
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(Tree const & tree) const -> std::size_t
        {
            return operator()(tree.childs, LLAMA_DEREFERENCE(tree.count));
        }
    };

    template<
        typename Tree,
        class InnerOp,
        class OuterOp,
        template<class, class>
        class LeafFunctor>
    struct Reduce<Tree, InnerOp, OuterOp, LeafFunctor, false>
    {
        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(const decltype(Tree::count) & count) const
            -> std::size_t
        {
            return OuterOp{}(
                count,
                internal::
                    ReduceElementType<Tree, InnerOp, OuterOp, LeafFunctor>()(
                        count));
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(const Tree & tree) const -> std::size_t
        {
            return operator()(LLAMA_DEREFERENCE(tree.count));
        }
    };
}

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
        template<class, class>
        class T_InnerOp,
        template<class, class>
        class T_OuterOp,
        template<class, class>
        class T_LeafFunctor,
        bool HC = HasChildren<Tree>::value>
    struct Reduce;

    namespace internal
    {
        // Leaf
        template<
            typename Tree,
            template<class, class>
            class T_InnerOp,
            template<class, class>
            class T_OuterOp,
            template<class, class>
            class T_LeafFunctor,
            typename T_SFINAE = void>
        struct ReduceElementType
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(const decltype(Tree::count) & count) const
                -> std::size_t
            {
                return T_LeafFunctor<
                    typename Tree::Type,
                    decltype(Tree::count)>()(count);
            }
        };

        // Node
        template<
            typename Tree,
            template<class, class>
            class T_InnerOp,
            template<class, class>
            class T_OuterOp,
            template<class, class>
            class T_LeafFunctor>
        struct ReduceElementType<
            Tree,
            T_InnerOp,
            T_OuterOp,
            T_LeafFunctor,
            std::enable_if_t<(SizeOfTuple<typename Tree::Type>::value > 1)>>
        {
            using IterTree = typename TreePopFrontChild<Tree>::ResultType;

            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(
                typename Tree::Type const & childs,
                decltype(Tree::count) const & count) const -> std::size_t
            {
                return T_InnerOp<
                    decltype(Reduce<
                             typename Tree::Type::FirstElement,
                             T_InnerOp,
                             T_OuterOp,
                             T_LeafFunctor>()(childs.first)),
                    decltype(internal::ReduceElementType<
                             IterTree,
                             T_InnerOp,
                             T_OuterOp,
                             T_LeafFunctor>()(childs.rest, count))>::
                    apply(
                        Reduce<
                            typename Tree::Type::FirstElement,
                            T_InnerOp,
                            T_OuterOp,
                            T_LeafFunctor>()(childs.first),
                        internal::ReduceElementType<
                            IterTree,
                            T_InnerOp,
                            T_OuterOp,
                            T_LeafFunctor>()(childs.rest, count));
            }
        };

        // Node with one (last) child
        template<
            typename Tree,
            template<class, class>
            class T_InnerOp,
            template<class, class>
            class T_OuterOp,
            template<class, class>
            class T_LeafFunctor>
        struct ReduceElementType<
            Tree,
            T_InnerOp,
            T_OuterOp,
            T_LeafFunctor,
            std::enable_if_t<SizeOfTuple<typename Tree::Type>::value == 1>>
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(
                typename Tree::Type const & childs,
                decltype(Tree::count) const & count) const -> std::size_t
            {
                return Reduce<
                    typename Tree::Type::FirstElement,
                    T_InnerOp,
                    T_OuterOp,
                    T_LeafFunctor>()(childs.first);
            }
        };
    }

    template<
        typename Tree,
        template<class, class>
        class T_InnerOp,
        template<class, class>
        class T_OuterOp,
        template<class, class>
        class T_LeafFunctor,
        bool HC>
    struct Reduce
    {
        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(
            typename Tree::Type const & childs,
            decltype(Tree::count) const & count) const -> std::size_t
        {
            return T_OuterOp<
                decltype(Tree::count),
                decltype(internal::ReduceElementType<
                         Tree,
                         T_InnerOp,
                         T_OuterOp,
                         T_LeafFunctor>()(childs, count))>::
                apply(
                    count,
                    internal::ReduceElementType<
                        Tree,
                        T_InnerOp,
                        T_OuterOp,
                        T_LeafFunctor>()(childs, count));
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(Tree const & tree) const -> std::size_t
        {
            return operator()(tree.childs, LLAMA_DEREFERENCE(tree.count));
        }
    };

    template<
        typename Tree,
        template<class, class>
        class T_InnerOp,
        template<class, class>
        class T_OuterOp,
        template<class, class>
        class T_LeafFunctor>
    struct Reduce<Tree, T_InnerOp, T_OuterOp, T_LeafFunctor, false>
    {
        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(const decltype(Tree::count) & count) const
            -> std::size_t
        {
            return T_OuterOp<
                decltype(Tree::count),
                decltype(internal::ReduceElementType<
                         Tree,
                         T_InnerOp,
                         T_OuterOp,
                         T_LeafFunctor>()(count))>::
                apply(
                    count,
                    internal::ReduceElementType<
                        Tree,
                        T_InnerOp,
                        T_OuterOp,
                        T_LeafFunctor>()(count));
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(const Tree & tree) const -> std::size_t
        {
            return operator()(LLAMA_DEREFERENCE(tree.count));
        }
    };
}

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
    template<template<class> class LeafFunctor>
    struct Reduce
    {
        template<typename... Children, std::size_t... Is, typename Count>
        LLAMA_FN_HOST_ACC_INLINE auto reduceChildren(
            const Tuple<Children...> & childs,
            std::index_sequence<Is...>,
            const Count & count) const -> std::size_t
        {
            return (operator()(getTupleElement<Is>(childs)) + ...);
        }

        template<typename... Children, typename Count>
        LLAMA_FN_HOST_ACC_INLINE auto
        operator()(const Tuple<Children...> & childs, const Count & count) const
            -> std::size_t
        {
            return count
                * reduceChildren(
                       childs,
                       std::make_index_sequence<sizeof...(Children)>{},
                       count);
        }

        template<
            typename Tree,
            std::enable_if_t<HasChildren<Tree>::value, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(const Tree & tree) const
            -> std::size_t
        {
            return operator()(tree.childs, LLAMA_DEREFERENCE(tree.count));
        }

        template<
            typename Tree,
            std::enable_if_t<!HasChildren<Tree>::value, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(const Tree & tree) const
            -> std::size_t
        {
            return tree.count * LeafFunctor<typename Tree::Type>()(tree.count);
        }
    };
}

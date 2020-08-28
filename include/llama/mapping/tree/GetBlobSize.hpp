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

#include "Reduce.hpp"

namespace llama::mapping::tree
{
    template<typename Type, typename CountType>
    struct SizeOfFunctor
    {
        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(const CountType &) const -> std::size_t
        {
            return sizeof(Type);
        }
    };

    template<typename Childs, typename CountType>
    LLAMA_FN_HOST_ACC_INLINE auto
    getTreeBlobSize(const Childs & childs, const CountType & count)
        -> std::size_t
    {
        return Reduce<
            TreeElement<NoName, Childs, CountType>,
            std::plus<>,
            std::multiplies<>,
            SizeOfFunctor>()(childs, count);
    }

    template<typename Tree>
    LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Tree & tree)
        -> std::size_t
    {
        return Reduce<Tree, std::plus<>, std::multiplies<>, SizeOfFunctor>()(
            tree);
    }
}

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

#include "Operations.hpp"
#include "Reduce.hpp"

namespace llama::mapping::tree
{
    template<typename T_Type, typename T_CountType>
    struct SizeOfFunctor
    {
        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(T_CountType const &) const -> std::size_t
        {
            return sizeof(T_Type);
        }
    };

    template<typename Childs, typename CountType>
    struct TreeOptimalType
    {
        using type = TreeElement<NoName, Childs>;
    };

    template<typename Childs, std::size_t Count>
    struct TreeOptimalType<Childs, std::integral_constant<std::size_t, Count>>
    {
        using type = TreeElementConst<NoName, Childs, Count>;
    };

    template<typename T_Childs, typename T_CountType>
    LLAMA_FN_HOST_ACC_INLINE auto
    getTreeBlobSize(T_Childs const & childs, T_CountType const & count)
        -> std::size_t
    {
        return Reduce<
            typename TreeOptimalType<T_Childs, T_CountType>::type,
            Addition,
            Multiplication,
            SizeOfFunctor>()(childs, count);
    }

    template<typename T_Tree>
    LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(T_Tree const & tree)
        -> std::size_t
    {
        return Reduce<T_Tree, Addition, Multiplication, SizeOfFunctor>()(tree);
    }
}

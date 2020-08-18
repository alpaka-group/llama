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

#include "../TreeElement.hpp"

namespace llama::mapping::tree::operations
{
    template<typename T_TreeCoord, typename T_Tree>
    LLAMA_FN_HOST_ACC_INLINE auto getNode(const T_Tree & tree)
    {
        if constexpr(std::is_same_v<T_TreeCoord, Tuple<>>)
            return tree;
        else
            return getNode<typename T_TreeCoord::RestTuple>(
                getTupleElement<decltype(
                    T_TreeCoord::FirstElement::compiletime)::value>(
                    tree.childs));
    }
}

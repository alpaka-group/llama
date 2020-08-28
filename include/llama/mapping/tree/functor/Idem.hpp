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

namespace llama::mapping::tree::functor
{
    /// Functor for \ref tree::Mapping. Does nothing with the mapping tree at
    /// all (basically implemented for testing purposes). \see tree::Mapping
    struct Idem
    {
        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree & tree) const
            -> Tree
        {
            return tree;
        }

        template<typename Tree, typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            const TreeCoord & basicCoord,
            const Tree &) const -> TreeCoord
        {
            return basicCoord;
        }

        template<typename Tree, typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            const TreeCoord & resultCoord,
            const Tree &) const -> TreeCoord
        {
            return resultCoord;
        }
    };
}

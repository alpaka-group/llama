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

namespace llama
{

namespace mapping
{

namespace tree
{

namespace functor
{

struct Idem
{
    template< typename T_Tree >
    using Result = T_Tree;

    template< typename T_Tree >
    auto
    basicToResult( const T_Tree & tree ) const
    -> Result< T_Tree >
    {
        return tree;
    }

    template<
        typename T_Tree,
        typename T_TreeCoord
    >
    auto
    basicCoordToResultCoord(
        T_TreeCoord const basicCoord,
        T_Tree const
    ) const
    -> T_TreeCoord
    {
        return basicCoord;
    }

    template<
        typename T_Tree,
        typename T_TreeCoord
    >
    auto
    resultCoordToBasicCoord(
        T_TreeCoord const resultCoord,
        T_Tree const
    ) const
    -> T_TreeCoord
    {
        return resultCoord;
    }
};

} // namespace functor

} // namespace tree

} // namespace mapping

} // namespace llama

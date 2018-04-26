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
#include "Operations.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

template< typename T_Leave >
struct SizeOfFunctor
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Leave const leave ) const
    -> std::size_t
    {
        return sizeof( typename T_Leave::Type );
    }
};

template< typename T_Tree >
LLAMA_FN_HOST_ACC_INLINE
auto
getTreeBlobSize( T_Tree const tree )
-> std::size_t
{
    return Reduce<
        T_Tree,
        Addition,
        Multiplication,
        SizeOfFunctor
    >()( tree );
}

} // namespace tree

} // namespace mapping

} // namespace llama


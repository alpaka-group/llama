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

namespace llama
{

namespace mapping
{

namespace tree
{

struct Addition
{
    template<
        typename Parameter1,
        typename Parameter2
    >
    LLAMA_FN_HOST_ACC_INLINE
    static constexpr
    auto
    apply(
        Parameter1 const p1,
        Parameter2 const p2
    )
    -> decltype( p1 + p2 )
    {
        return p1 + p2;
    }
};

struct Multiplication
{
    template<
        typename Parameter1,
        typename Parameter2
    >
    LLAMA_FN_HOST_ACC_INLINE
    static constexpr
    auto
    apply(
        Parameter1 const p1,
        Parameter2 const p2
    )
    -> decltype( p1 * p2 )
    {
        return p1 * p2;
    }
};


} // namespace tree

} // namespace mapping

} // namespace llama


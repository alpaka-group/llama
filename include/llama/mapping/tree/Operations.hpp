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

template<
    typename Parameter1,
    typename Parameter2
>
struct Addition
{
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

template<
    typename T,
    T t1,
    T t2
>
struct Addition<
    std::integral_constant<T,t1>,
    std::integral_constant<T,t2>
>
{
    using Parameter1 = std::integral_constant<T,t1>;
    using Parameter2 = std::integral_constant<T,t2>;

    LLAMA_FN_HOST_ACC_INLINE
    static constexpr
    auto
    apply(
        Parameter1 const p1,
        Parameter2 const p2
    )
    -> std::integral_constant<T,t1 + t2>
    {
        return std::integral_constant<T,t1 + t2>();
    }
};

template<
    typename Parameter1,
    typename Parameter2
>
struct Multiplication
{
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

template<
    typename T,
    T t1,
    T t2
>
struct Multiplication<
    std::integral_constant<T,t1>,
    std::integral_constant<T,t2>
>
{
    using Parameter1 = std::integral_constant<T,t1>;
    using Parameter2 = std::integral_constant<T,t2>;

    LLAMA_FN_HOST_ACC_INLINE
    static constexpr
    auto
    apply(
        Parameter1 const p1,
        Parameter2 const p2
    )
    -> std::integral_constant<T,t1 * t2>
    {
        return std::integral_constant<T,t1 * t2>();
    }
};


} // namespace tree

} // namespace mapping

} // namespace llama


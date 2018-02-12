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

#include "DateStruct.hpp"

namespace llama
{

namespace internal
{
    template<
        typename T,
        std::size_t T_count,
        typename... T_List
    >
    struct AddChildToStruct
    {
        using type = typename AddChildToStruct<
            T,
            T_count - 1,
            T_List...,
            T
        >::type;
    };
    template<
        typename T,
        typename... T_List
    >
    struct AddChildToStruct<
        T,
        0,
        T_List...
    >
    {
        using type = DateStruct< T_List... >;
    };

} // namespace internal

template<
    typename T_Child,
    std::size_t T_count
>
using DateArray = typename internal::AddChildToStruct<
    T_Child,
    T_count
>::type;

} // namespace llama

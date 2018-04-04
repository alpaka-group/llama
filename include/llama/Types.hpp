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

#include "Array.hpp"
#include <boost/mp11.hpp>

namespace llama
{

template< std::size_t T_dim >
using UserDomain = Array<
    std::size_t,
    T_dim
>;

template<
    typename... T_Leaves
>
using DatumStruct = boost::mp11::mp_list<
    T_Leaves...
>;

template<
    typename... T_Leaves
>
using DS = DatumStruct<
    T_Leaves...
>;

template<
    typename T_Identifier,
    typename T_Type
>
using DatumElement = boost::mp11::mp_list<
    T_Identifier,
    T_Type
>;

template<
    typename T_Identifier,
    typename T_Type
>
using DE = DatumElement<
    T_Identifier,
    T_Type
>;

} // namespace llama

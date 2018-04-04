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

#include "DatumStruct.hpp"
#include <boost/mp11.hpp>

namespace llama
{

struct NoName {};

template<
    typename T_Child,
    std::size_t T_count
>
using DatumArray = boost::mp11::mp_repeat_c<
    DatumStruct< DatumElement< NoName, T_Child > >,
    T_count
>;

template<
    typename T_Child,
    std::size_t T_count
>
using DA = DatumArray<
    T_Child,
    T_count
>;

} // namespace llama

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

/** Anonymous naming for a \ref DatumElement. Especially used for a \ref
 *  DatumArray. Two DatumElements with the same identifier llama::NoName will
 *  not match.
 * */
struct NoName {};

/** The run-time specified user domain
 * \tparam T_dim compile time dimensionality of the user domain
 * */
template< std::size_t T_dim >
using UserDomain = Array<
    std::size_t,
    T_dim
>;

/** A list of \ref DatumElement which may be used to define a datum domain.
 * \tparam T_Leaves... List of \ref DatumElement
 * */
template<
    typename... T_Leaves
>
using DatumStruct = boost::mp11::mp_list<
    T_Leaves...
>;

/// Shortcut for \ref DatumStruct
template<
    typename... T_Leaves
>
using DS = DatumStruct<
    T_Leaves...
>;

/** Datum domain tree node which may either a leaf or refer to a child tree
 *  presented as another \ref DatumStruct.
 * \tparam T_Identifier Name of the node. May be any type (struct, class). \ref
 *  llama::NoName is considered a special identifier, which never matches.
 * \tparam T_Type Type of the node. May be either another sub tree consisting of
 *  a nested \ref DatumStruct or any other type making it a leaf of this type.
 * */
template<
    typename T_Identifier,
    typename T_Type
>
using DatumElement = boost::mp11::mp_list<
    T_Identifier,
    T_Type
>;

/// Shortcut for \ref DatumElement
template<
    typename T_Identifier,
    typename T_Type
>
using DE = DatumElement<
    T_Identifier,
    T_Type
>;

} // namespace llama

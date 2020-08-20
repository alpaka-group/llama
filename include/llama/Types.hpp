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
#include <type_traits>

namespace llama
{
    /** Anonymous naming for a \ref DatumElement. Especially used for a \ref
     *  DatumArray. Two DatumElements with the same identifier llama::NoName
     * will not match.
     * */
    struct NoName
    {};

    /** The run-time specified user domain
     * \tparam Dim compile time dimensionality of the user domain
     * */
    template<std::size_t Dim>
    struct UserDomain : Array<std::size_t, Dim>
    {};

    static_assert(std::is_trivially_default_constructible_v<
                  UserDomain<1>>); // so UserDomain<1>{} will produce a zeroed
                                   // coord. Should hold for all dimensions, but
                                   // just checking for <1> here.

    /** A list of \ref DatumElement which may be used to define a datum domain.
     * \tparam T_Leaves... List of \ref DatumElement
     * */
    template<typename... T_Leaves>
    using DatumStruct = boost::mp11::mp_list<T_Leaves...>;

    /// Shortcut for \ref DatumStruct
    template<typename... T_Leaves>
    using DS = DatumStruct<T_Leaves...>;

    /** Datum domain tree node which may either a leaf or refer to a child tree
     *  presented as another \ref DatumStruct.
     * \tparam T_Identifier Name of the node. May be any type (struct, class).
     * \ref llama::NoName is considered a special identifier, which never
     * matches. \tparam T_Type Type of the node. May be either another sub tree
     * consisting of a nested \ref DatumStruct or any other type making it a
     * leaf of this type.
     * */
    template<typename T_Identifier, typename T_Type>
    using DatumElement = boost::mp11::mp_list<T_Identifier, T_Type>;

    /// Shortcut for \ref DatumElement
    template<typename T_Identifier, typename T_Type>
    using DE = DatumElement<T_Identifier, T_Type>;

    /** An array of anonymous but identical \ref DatumElement "DatumElements".
     * Can be used anywhere where \ref DatumStruct may used. \tparam T_Child
     * type to repeat. May be either another sub tree consisting of a nested
     * \ref DatumStruct resp. DatumArray or any other type making it an array of
     * leaves of this type. \tparam T_count number of repetitions of T_Child
     * */
    template<typename Child, std::size_t Count>
    using DatumArray = boost::mp11::
        mp_repeat_c<DatumStruct<DatumElement<NoName, Child>>, Count>;

    /// Shortcut for \ref DatumArray
    template<typename Child, std::size_t Count>
    using DA = DatumArray<Child, Count>;

    struct NrAndOffset
    {
        std::size_t nr;
        std::size_t offset;
    };
}

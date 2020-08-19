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

#include "Functions.hpp"
#include "GetType.hpp"

#include <boost/mp11.hpp>
#include <type_traits>

namespace llama
{
    namespace internal
    {
        template<
            typename T_DatumDomain,
            typename T_DatumCoord,
            std::size_t T_pos,
            typename T_SFinae,
            typename... T_UID>
        struct GetCoordFromUIDImpl
        {
            static_assert(
                boost::mp11::mp_size<T_DatumDomain>::value != 0,
                "UID combination is not valid");
        };

        template<
            typename T_DatumDomain,
            typename T_DatumCoord,
            std::size_t T_pos,
            typename T_FirstUID,
            typename... T_UID>
        struct GetCoordFromUIDImpl<
            T_DatumDomain,
            T_DatumCoord,
            T_pos,
            std::enable_if_t<std::is_same<
                T_FirstUID,
                GetDatumElementUID<boost::mp11::mp_first<T_DatumDomain>>>::
                                 value>,
            T_FirstUID,
            T_UID...>
        {
            using type = typename GetCoordFromUIDImpl<
                GetDatumElementType<boost::mp11::mp_first<T_DatumDomain>>,
                typename T_DatumCoord::template PushBack<T_pos>,
                0,
                void,
                T_UID...>::type;
        };

        template<
            typename T_DatumDomain,
            typename T_DatumCoord,
            std::size_t T_pos,
            typename T_FirstUID,
            typename... T_UID>
        struct GetCoordFromUIDImpl<
            T_DatumDomain,
            T_DatumCoord,
            T_pos,
            std::enable_if_t<!std::is_same<
                T_FirstUID,
                GetDatumElementUID<boost::mp11::mp_first<T_DatumDomain>>>::
                                 value>,
            T_FirstUID,
            T_UID...>
        {
            using type = typename GetCoordFromUIDImpl<
                boost::mp11::mp_pop_front<T_DatumDomain>,
                T_DatumCoord,
                T_pos + 1,
                void,
                T_FirstUID,
                T_UID...>::type;
        };

        template<
            typename T_DatumDomain,
            typename T_DatumCoord,
            std::size_t T_pos>
        struct GetCoordFromUIDImpl<T_DatumDomain, T_DatumCoord, T_pos, void>
        {
            using type = T_DatumCoord;
        };
    }

    /** Converts a coordinate in a datum domain given as UID to a \ref
     * DatumCoord . \tparam T_DatumDomain the datum domain (\ref DatumStruct)
     * \tparam T_UID... the uid of in the datum domain, may also be empty (for
     *  `DatumCoord< >`)
     * \returns a \ref DatumCoord with the datum domain tree coordinates as
     * template parameters
     */
    template<typename T_DatumDomain, typename... T_UID>
    using GetCoordFromUID = typename internal::
        GetCoordFromUIDImpl<T_DatumDomain, DatumCoord<>, 0, void, T_UID...>::
            type;

    namespace internal
    {
        template<
            typename T_DatumDomain,
            typename T_DatumCoord,
            typename... T_UID>
        struct GetCoordFromUIDRelativeImpl
        {
            using AbsolutCoord = typename internal::GetCoordFromUIDImpl<
                GetType<T_DatumDomain, T_DatumCoord>,
                T_DatumCoord,
                0,
                void,
                T_UID...>::type;
            // Only returning the datum coord relative to T_DatumCoord
            using type = typename AbsolutCoord::template Back<
                AbsolutCoord::size - T_DatumCoord::size>;
        };
    }

    /** Converts a coordinate in a datum domain given as UID to a \ref
     * DatumCoord relative to a given datum coord in the tree. The returned
     * datum coord is also relative to the input datum coord, that is the sub
     * tree. \tparam T_DatumDomain the datum domain (\ref DatumStruct) \tparam
     * T_DatumCoord datum coord to start the translation from UID to datum coord
     * \tparam T_UID... the uid of in the datum domain, may also be empty (for
     *  `DatumCoord< >`)
     * \returns a \ref DatumCoord with the datum domain tree coordinates as
     * template parameters
     */
    template<typename T_DatumDomain, typename T_DatumCoord, typename... T_UID>
    using GetCoordFromUIDRelative =
        typename internal::GetCoordFromUIDRelativeImpl<
            T_DatumDomain,
            T_DatumCoord,
            T_UID...>::type;

    /** Returns the type of a node in a datum domain tree for a coordinate given
     * as UID \tparam T_DatumDomain the datum domain (probably \ref DatumStruct)
     * \tparam T_DatumCoord the coordinate
     * \return type at the specified node
     */
    template<typename T_DatumDomain, typename... T_UIDs>
    using GetTypeFromUID = GetType<
        T_DatumDomain,
        GetCoordFromUID<T_DatumDomain, T_UIDs...>>;
}

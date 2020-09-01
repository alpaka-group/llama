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

#include "DatumCoord.hpp"
#include "Types.hpp"

#include <boost/mp11.hpp>
#include <type_traits>

namespace llama
{
    /** helper function to get the UID of a \ref DatumElement node
     * \tparam DatumElement \ref DatumElement to get the UID of
     * \return UID of the element
     */
    template<typename DatumElement>
    using GetDatumElementUID = boost::mp11::mp_first<DatumElement>;

    /** helper function to get the type of a \ref DatumElement node
     * \tparam DatumElement \ref DatumElement to get the type of
     * \return type of the element
     */
    template<typename DatumElement>
    using GetDatumElementType = boost::mp11::mp_second<DatumElement>;

    namespace internal
    {
        template<typename T, typename TargetDatumCoord, typename IterCoord>
        constexpr auto linearBytePosImpl(T *, TargetDatumCoord, IterCoord)
        {
            return sizeof(T)
                * static_cast<std::size_t>(
                       DatumCoordIsBigger<TargetDatumCoord, IterCoord>);
        }

        template<
            typename... DatumElements,
            typename TargetDatumCoord,
            std::size_t... IterCoords>
        constexpr auto linearBytePosImpl(
            DatumStruct<DatumElements...> *,
            TargetDatumCoord,
            DatumCoord<IterCoords...>)
        {
            std::size_t acc = 0;
            boost::mp11::mp_for_each<
                boost::mp11::mp_iota_c<sizeof...(DatumElements)>>([&](
                auto i) constexpr {
                constexpr auto index = decltype(i)::value;
                using Element = boost::mp11::
                    mp_at_c<DatumStruct<DatumElements...>, index>;

                acc += linearBytePosImpl(
                    (GetDatumElementType<Element> *)nullptr,
                    TargetDatumCoord{},
                    DatumCoord<IterCoords..., index>{});
            });
            return acc;
        }
    }

    /** Gives the byte position of an element in a datum domain if it would be a
     *  normal struct
     * \tparam DatumDomain datum domain tree
     * \tparam Coords... coordinate in datum domain tree
     * \return the byte position as compile time value in "value"
     */
    template<typename DatumDomain, std::size_t... Coords>
    constexpr auto linearBytePos() -> std::size_t
    {
        return internal::linearBytePosImpl(
            (DatumDomain *)nullptr, DatumCoord<Coords...>{}, DatumCoord<>{});
    }

    /** Gives the size a datum domain if it would be a normal struct
     * \tparam DatumDomain datum domain tree
     * \return the byte position as compile time value in "value"
     */
    template<typename DatumDomain>
    static constexpr auto SizeOf = sizeof(DatumDomain);

    template<typename... DatumElements>
    static constexpr auto SizeOf<DatumStruct<
        DatumElements...>> = (SizeOf<GetDatumElementType<DatumElements>> + ...);

    template<typename T>
    inline constexpr auto IsDatumStruct = false;

    template<typename... DatumElements>
    inline constexpr auto IsDatumStruct<DatumStruct<DatumElements...>> = true;

    namespace internal
    {
        template<typename DatumDomain, typename DatumCoord>
        struct GetTypeImpl;

        template<
            typename... Children,
            std::size_t HeadCoord,
            std::size_t... TailCoords>
        struct GetTypeImpl<
            DatumStruct<Children...>,
            DatumCoord<HeadCoord, TailCoords...>>
        {
            using ChildType = GetDatumElementType<
                boost::mp11::mp_at_c<DatumStruct<Children...>, HeadCoord>>;
            using type =
                typename GetTypeImpl<ChildType, DatumCoord<TailCoords...>>::
                    type;
        };

        template<typename T>
        struct GetTypeImpl<T, DatumCoord<>>
        {
            using type = T;
        };
    }

    /** Returns the type of a node in a datum domain tree for a coordinate given
     * as \ref DatumCoord \tparam DatumDomain the datum domain (probably \ref
     * DatumStruct) \tparam DatumCoord the coordinate \return type at the
     * specified node
     */
    template<typename DatumDomain, typename DatumCoord>
    using GetType =
        typename internal::GetTypeImpl<DatumDomain, DatumCoord>::type;

    namespace internal
    {
        template<typename CurrTag, typename DatumDomain, typename DatumCoord>
        struct GetUIDImpl;

        template<
            typename CurrTag,
            typename... DatumElements,
            std::size_t FirstCoord,
            std::size_t... Coords>
        struct GetUIDImpl<
            CurrTag,
            DatumStruct<DatumElements...>,
            DatumCoord<FirstCoord, Coords...>>
        {
            using DatumElement = boost::mp11::
                mp_at_c<boost::mp11::mp_list<DatumElements...>, FirstCoord>;
            using ChildTag = GetDatumElementUID<DatumElement>;
            using ChildType = GetDatumElementType<DatumElement>;
            using type = typename GetUIDImpl<
                ChildTag,
                ChildType,
                DatumCoord<Coords...>>::type;
        };

        template<typename CurrTag, typename T>
        struct GetUIDImpl<CurrTag, T, DatumCoord<>>
        {
            using type = CurrTag;
        };
    }

    /** return the unique identifier of the \ref DatumElement at a \ref
     *  DatumCoord inside the datum domain tree.
     * \tparam DatumDomain the datum domain, probably of type \ref DatumStruct
     * or \ref DatumArray \tparam DatumCoord the datum coord, probably of type
     * \ref DatumCoord \return unique identifer type
     * */
    template<typename DatumDomain, typename DatumCoord>
    using GetUID =
        typename internal::GetUIDImpl<NoName, DatumDomain, DatumCoord>::type;

    /** Tells whether two coordinates in two datum domains have the same UID.
     * \tparam DDA first user domain
     * \tparam BaseA First part of the coordinate in the first user domain as
     *  \ref DatumCoord. This will be used for getting the UID, but not for the
     *  comparison.
     * \tparam LocalA Second part of the coordinate in the first user domain
     * as \ref DatumCoord. This will be used for the comparison with the second
     *  datum domain.
     * \tparam DDB second user domain
     * \tparam BaseB First part of the coordinate in the second user domain as
     *  \ref DatumCoord. This will be used for getting the UID, but not for the
     *  comparison.
     * \tparam LocalB Second part of the coordinate in the second user domain
     * as \ref DatumCoord. This will be used for the comparison with the first
     *  datum domain.
     */
    template<
        typename DatumDomainA,
        typename BaseA,
        typename LocalA,
        typename DatumDomainB,
        typename BaseB,
        typename LocalB>
    inline constexpr auto CompareUID = false;

    template<
        typename DatumDomainA,
        std::size_t... BaseCoordsA,
        typename LocalA,
        typename DatumDomainB,
        std::size_t... BaseCoordsB,
        typename LocalB>
    inline constexpr auto CompareUID<
        DatumDomainA,
        DatumCoord<BaseCoordsA...>,
        LocalA,
        DatumDomainB,
        DatumCoord<BaseCoordsB...>,
        LocalB> = []() constexpr
    {
        if constexpr(LocalA::size != LocalB::size)
            return false;
        else if constexpr(LocalA::size == 0 && LocalB::size == 0)
            return true;
        else
        {
            using TagCoordA = DatumCoord<BaseCoordsA..., LocalA::front>;
            using TagCoordB = DatumCoord<BaseCoordsB..., LocalB::front>;

            return std::is_same_v<
                       GetUID<DatumDomainA, TagCoordA>,
                       GetUID<
                           DatumDomainB,
                           TagCoordB>> && CompareUID<DatumDomainA, TagCoordA, PopFront<LocalA>, DatumDomainB, TagCoordB, PopFront<LocalB>>;
        }
    }
    ();

    namespace internal
    {
        template<typename DatumDomain, typename DatumCoord, typename... UID>
        struct GetCoordFromUIDImpl
        {
            static_assert(
                boost::mp11::mp_size<DatumDomain>::value != 0,
                "UID combination is not valid");
        };

        template<
            typename... DatumElements,
            std::size_t... ResultCoords,
            typename FirstUID,
            typename... UID>
        struct GetCoordFromUIDImpl<
            DatumStruct<DatumElements...>,
            DatumCoord<ResultCoords...>,
            FirstUID,
            UID...>
        {
            template<typename DatumElement>
            struct HasTag :
                    std::is_same<GetDatumElementUID<DatumElement>, FirstUID>
            {};

            static constexpr auto tagIndex = boost::mp11::mp_find_if<
                boost::mp11::mp_list<DatumElements...>,
                HasTag>::value;
            static_assert(
                tagIndex < sizeof...(DatumElements),
                "FirstUID was not found inside this DatumStruct");

            using ChildType = GetDatumElementType<
                boost::mp11::mp_at_c<DatumStruct<DatumElements...>, tagIndex>>;

            using type = typename GetCoordFromUIDImpl<
                ChildType,
                DatumCoord<ResultCoords..., tagIndex>,
                UID...>::type;
        };

        template<typename DatumDomain, typename DatumCoord>
        struct GetCoordFromUIDImpl<DatumDomain, DatumCoord>
        {
            using type = DatumCoord;
        };
    }

    /** Converts a coordinate in a datum domain given as UID to a \ref
     * DatumCoord . \tparam DatumDomain the datum domain (\ref DatumStruct)
     * \tparam UID... the uid of in the datum domain, may also be empty (for
     *  `DatumCoord< >`)
     * \returns a \ref DatumCoord with the datum domain tree coordinates as
     * template parameters
     */
    template<typename DatumDomain, typename... UID>
    using GetCoordFromUID = typename internal::
        GetCoordFromUIDImpl<DatumDomain, DatumCoord<>, UID...>::type;

    namespace internal
    {
        template<typename DatumDomain, typename DatumCoord, typename... UID>
        struct GetCoordFromUIDRelativeImpl
        {
            using AbsolutCoord = typename internal::GetCoordFromUIDImpl<
                GetType<DatumDomain, DatumCoord>,
                DatumCoord,
                UID...>::type;
            // Only returning the datum coord relative to DatumCoord
            using type = DatumCoordFromList<boost::mp11::mp_drop_c<
                typename AbsolutCoord::List,
                DatumCoord::size>>;
        };
    }

    /** Converts a coordinate in a datum domain given as UID to a \ref
     * DatumCoord relative to a given datum coord in the tree. The returned
     * datum coord is also relative to the input datum coord, that is the sub
     * tree. \tparam DatumDomain the datum domain (\ref DatumStruct) \tparam
     * DatumCoord datum coord to start the translation from UID to datum coord
     * \tparam UID... the uid of in the datum domain, may also be empty (for
     *  `DatumCoord< >`)
     * \returns a \ref DatumCoord with the datum domain tree coordinates as
     * template parameters
     */
    template<typename DatumDomain, typename DatumCoord, typename... UID>
    using GetCoordFromUIDRelative = typename internal::
        GetCoordFromUIDRelativeImpl<DatumDomain, DatumCoord, UID...>::type;

    /** Returns the type of a node in a datum domain tree for a coordinate given
     * as UID \tparam DatumDomain the datum domain (probably \ref DatumStruct)
     * \tparam DatumCoord the coordinate
     * \return type at the specified node
     */
    template<typename DatumDomain, typename... UIDs>
    using GetTypeFromUID
        = GetType<DatumDomain, GetCoordFromUID<DatumDomain, UIDs...>>;
}

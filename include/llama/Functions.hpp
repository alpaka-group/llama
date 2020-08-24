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
    /** helper function to get the type of a \ref DatumElement node
     * \tparam T_DatumElement \ref DatumElement to get the type of
     * \return type of the element
     */
    template<typename T_DatumElement>
    using GetDatumElementType = boost::mp11::mp_second<T_DatumElement>;

    /** helper function to get the UID of a \ref DatumElement node
     * \tparam T_DatumElement \ref DatumElement to get the UID of
     * \return UID of the element
     */
    template<typename T_DatumElement>
    using GetDatumElementUID = boost::mp11::mp_first<T_DatumElement>;

    namespace internal
    {
        template<
            typename DatumDomain,
            typename TargetDatumCoord,
            typename IterCoord>
        constexpr auto
        linearBytePosImpl(DatumDomain *, TargetDatumCoord, IterCoord)
        {
            return sizeof(DatumDomain)
                * static_cast<std::size_t>(
                       DatumCoordIsBigger<TargetDatumCoord, IterCoord>::value);
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
    struct LinearBytePos
    {
        static constexpr std::size_t value = internal::linearBytePosImpl(
            (DatumDomain *)nullptr,
            DatumCoord<Coords...>{},
            DatumCoord<>{});
    };

    /** Gives the size a datum domain if it would be a normal struct
     * \tparam T_DatumDomain datum domain tree
     * \return the byte position as compile time value in "value"
     */
    template<typename T_DatumDomain>
    struct SizeOf
    {
        static constexpr std::size_t value = sizeof(T_DatumDomain);
    };

    template<typename T_FirstDatumElement, typename... T_DatumElements>
    struct SizeOf<DatumStruct<T_FirstDatumElement, T_DatumElements...>>
    {
        static constexpr std::size_t value
            = SizeOf<GetDatumElementType<T_FirstDatumElement>>::value
            + SizeOf<DatumStruct<T_DatumElements...>>::value;
    };

    template<>
    struct SizeOf<DatumStruct<>>
    {
        static constexpr std::size_t value = 0;
    };

    namespace internal
    {
        template<typename T_DatumDomain>
        class StubTypeImpl
        {
            struct impl
            {
                using type = T_DatumDomain;
                unsigned char stub[SizeOf<T_DatumDomain>::value];
            };

        public:
            using type = impl;
        };
    }

    /** Returns a type for a datum domain with the same size as \ref SizeOf of
     * this datum domain, useful e.g. if an external memory allocation function
     * needs a type and a number of elements instead of the total size in bytes.
     * \tparam T_DatumDomain the datum domain type to create a stub type for
     *  Additionally, this type has a type member named \a type which holds the
     *  original datum domain type.
     */
    template<typename T_DatumDomain>
    using StubType = typename internal::StubTypeImpl<T_DatumDomain>::type;

    template<typename T_Type>
    struct is_DatumStruct
    {
        static constexpr bool value = false;
    };

    template<typename... T_DatumStructContent>
    struct is_DatumStruct<DatumStruct<T_DatumStructContent...>>
    {
        static constexpr bool value = true;
    };

    namespace internal
    {
        template<typename DatumDomain, typename T_DatumCoord>
        struct GetTypeImpl;

        template<
            typename DatumDomain,
            std::size_t HeadCoord,
            std::size_t... TailCoords>
        struct GetTypeImpl<DatumDomain, DatumCoord<HeadCoord, TailCoords...>>
        {
            using _DateElement = boost::mp11::mp_at_c<DatumDomain, HeadCoord>;
            using type = typename GetTypeImpl<
                GetDatumElementType<_DateElement>,
                DatumCoord<TailCoords...>>::type;
        };

        template<typename T>
        struct GetTypeImpl<T, DatumCoord<>>
        {
            using type = T;
        };
    }

    /** Returns the type of a node in a datum domain tree for a coordinate given
     * as \ref DatumCoord \tparam DatumDomain the datum domain (probably \ref
     * DatumStruct) \tparam T_DatumCoord the coordinate \return type at the
     * specified node
     */
    template<typename DatumDomain, typename T_DatumCoord>
    using GetType =
        typename internal::GetTypeImpl<DatumDomain, T_DatumCoord>::type;

    namespace internal
    {
        template<typename DatumElement, std::size_t... DatumDomainCoords>
        struct GetUIDImpl;

        template<
            typename DatumElement,
            std::size_t FirstDatumDomainCoord,
            std::size_t... DatumDomainCoords>
        struct GetUIDImpl<
            DatumElement,
            FirstDatumDomainCoord,
            DatumDomainCoords...>
        {
            using _DateElement = boost::mp11::mp_at_c<
                GetDatumElementType<DatumElement>,
                FirstDatumDomainCoord>;
            using type =
                typename GetUIDImpl<_DateElement, DatumDomainCoords...>::type;
        };

        template<typename DatumElement>
        struct GetUIDImpl<DatumElement>
        {
            using type = GetDatumElementUID<DatumElement>;
        };

        template<typename DatumElement, typename T_DatumDomainCoord>
        struct GetUIDfromDatumCoord;

        template<typename DatumElement, std::size_t... DatumDomainCoords>
        struct GetUIDfromDatumCoord<
            DatumElement,
            DatumCoord<DatumDomainCoords...>>
        {
            using type =
                typename GetUIDImpl<DatumElement, DatumDomainCoords...>::type;
        };
    }

    /** return the unique identifier of the \ref DatumElement at a \ref
     *  DatumCoord inside the datum domain tree.
     * \tparam T_DatumDomain the datum domain, probably of type \ref DatumStruct
     * or \ref DatumArray \tparam T_DatumCoord the datum coord, probably of type
     * \ref DatumCoord \return unique identifer type
     * */
    template<typename T_DatumDomain, typename T_DatumCoord>
    using GetUID = typename internal::GetUIDfromDatumCoord<
        DatumElement<NoName, T_DatumDomain>,
        T_DatumCoord>::type;

    /** Tells whether two coordinates in two datum domains have the same UID.
     * \tparam T_DDA first user domain
     * \tparam T_BaseA First part of the coordinate in the first user domain as
     *  \ref DatumCoord. This will be used for getting the UID, but not for the
     *  comparison.
     * \tparam T_LocalA Second part of the coordinate in the first user domain
     * as \ref DatumCoord. This will be used for the comparison with the second
     *  datum domain.
     * \tparam T_DDB second user domain
     * \tparam T_BaseB First part of the coordinate in the second user domain as
     *  \ref DatumCoord. This will be used for getting the UID, but not for the
     *  comparison.
     * \tparam T_LocalB Second part of the coordinate in the second user domain
     * as \ref DatumCoord. This will be used for the comparison with the first
     *  datum domain.
     */
    template<
        typename T_DDA,
        typename T_BaseA,
        typename T_LocalA,
        typename T_DDB,
        typename T_BaseB,
        typename T_LocalB>
    struct CompareUID
    {
        /// true if the two UIDs are exactly the same, otherwise false.
        static constexpr bool value = []() constexpr
        {
            if constexpr(T_LocalA::size != T_LocalB::size)
                return false;
            else if constexpr(T_LocalA::size == 0 && T_LocalB::size == 0)
                return true;
            else
            {
                return std::is_same<
                           GetUID<
                               T_DDA,
                               typename T_BaseA::template PushBack<
                                   T_LocalA::front>>,
                           GetUID<
                               T_DDB,
                               typename T_BaseB::template PushBack<
                                   T_LocalB::front>>>::value
                    && CompareUID<
                           T_DDA,
                           typename T_BaseA::template PushBack<T_LocalA::front>,
                           typename T_LocalA::PopFront,
                           T_DDB,
                           typename T_BaseB::template PushBack<T_LocalB::front>,
                           typename T_LocalB::PopFront>::value;
            }
        }
        ();
    };

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
    using GetTypeFromUID
        = GetType<T_DatumDomain, GetCoordFromUID<T_DatumDomain, T_UIDs...>>;
}

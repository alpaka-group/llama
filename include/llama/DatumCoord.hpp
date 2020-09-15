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

#include <array>
#include <boost/mp11.hpp>
#include <type_traits>

namespace llama
{
    template<std::size_t... Coords>
    struct DatumCoord;

    namespace internal
    {
        template<class L>
        struct mp_unwrap_sizes_impl;

        template<template<class...> class L, typename... T>
        struct mp_unwrap_sizes_impl<L<T...>>
        {
            using type = DatumCoord<T::value...>;
        };

        template<typename L>
        using mp_unwrap_sizes = typename mp_unwrap_sizes_impl<L>::type;
    }

    /// Converts a type list of integral constants into a \ref DatumCoord.
    template<typename L>
    using DatumCoordFromList = internal::mp_unwrap_sizes<L>;

    /// Represents a coordinate for an element inside of datum domain tree.
    /// \tparam Coords... the compile time coordinate.
    template<std::size_t... Coords>
    struct DatumCoord
    {
        /// The list of integral coordinates as `boost::mp11::mp_list`.
        using List = boost::mp11::mp_list_c<std::size_t, Coords...>;

        static constexpr std::size_t front = boost::mp11::mp_front<List>::value;
        static constexpr std::size_t back = boost::mp11::mp_back<List>::value;
        static constexpr std::size_t size = sizeof...(Coords);
    };

    template<>
    struct DatumCoord<>
    {
        using List = boost::mp11::mp_list_c<std::size_t>;

        static constexpr std::size_t size = 0;
    };

    /// Concatenate two \ref DatumCoords.
    template<typename DatumCoord1, typename DatumCoord2>
    using Cat = DatumCoordFromList<boost::mp11::mp_append<
        typename DatumCoord1::List,
        typename DatumCoord2::List>>;

    /// DatumCoord without first coordinate component.
    template<typename DatumCoord>
    using PopFront = DatumCoordFromList<
        boost::mp11::mp_pop_front<typename DatumCoord::List>>;

    namespace internal
    {
        template<typename First, typename Second>
        struct DatumCoordIsBiggerImpl;

        template<std::size_t... Coords1, std::size_t... Coords2>
        struct DatumCoordIsBiggerImpl<
            DatumCoord<Coords1...>,
            DatumCoord<Coords2...>>
        {
            static constexpr auto value = []() constexpr
            {
                // CTAD does not work if Coords1/2 is an empty pack
                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
                for(auto i = 0; i < std::min(a1.size(), a2.size()); i++)
                {
                    if(a1[i] > a2[i])
                        return true;
                    if(a1[i] < a2[i])
                        return false;
                }
                return false;
            }
            ();
        };
    }

    /// Checks wether the first DatumCoord is bigger than the second.
    template<typename First, typename Second>
    inline constexpr auto DatumCoordIsBigger
        = internal::DatumCoordIsBiggerImpl<First, Second>::value;

    namespace internal
    {
        template<typename First, typename Second>
        struct DatumCoordIsSameImpl;

        template<std::size_t... Coords1, std::size_t... Coords2>
        struct DatumCoordIsSameImpl<
            DatumCoord<Coords1...>,
            DatumCoord<Coords2...>>
        {
            static constexpr auto value = []() constexpr
            {
                // CTAD does not work if Coords1/2 is an empty pack
                std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
                std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
                for(auto i = 0; i < std::min(a1.size(), a2.size()); i++)
                    if(a1[i] != a2[i])
                        return false;
                return true;
            }
            ();
        };
    }

    /// Checks wether two \ref DatumCoords are the same or one is the prefix of
    /// the other.
    template<typename First, typename Second>
    inline constexpr auto DatumCoordIsSame
        = internal::DatumCoordIsSameImpl<First, Second>::value;
}

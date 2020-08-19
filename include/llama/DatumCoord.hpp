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

/// Documentation of this file is in DatumCoord.dox!

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

    template<std::size_t... Coords>
    struct DatumCoord
    {
        using coord_list = boost::mp11::mp_list_c<std::size_t, Coords...>;

        static constexpr std::size_t front
            = boost::mp11::mp_front<coord_list>::value;
        static constexpr std::size_t size = sizeof...(Coords);
        static constexpr std::size_t back
            = boost::mp11::mp_back<coord_list>::value;

        using PopFront
            = internal::mp_unwrap_sizes<boost::mp11::mp_pop_front<coord_list>>;

        template<std::size_t NewCoord>
        using PushFront = DatumCoord<NewCoord, Coords...>;

        template<std::size_t NewCoord>
        using PushBack = DatumCoord<Coords..., NewCoord>;

        using IncBack = std::conditional_t<
            (sizeof...(Coords) > 1),
            typename PopFront::IncBack::template PushFront<front>,
            DatumCoord<front + 1>>;

        template<std::size_t N>
        using Front
            = internal::mp_unwrap_sizes<boost::mp11::mp_take_c<coord_list, N>>;

        template<std::size_t N>
        using Back = internal::mp_unwrap_sizes<boost::mp11::mp_reverse<
            boost::mp11::mp_take_c<boost::mp11::mp_reverse<coord_list>, N>>>;

        template<typename OtherDatumCoord>
        using Cat = internal::mp_unwrap_sizes<boost::mp11::mp_append<
            coord_list,
            typename OtherDatumCoord::coord_list>>;
    };

    template<>
    struct DatumCoord<>
    {
        using coord_list = boost::mp11::mp_list_c<std::size_t>;

        static constexpr std::size_t size = 0;

        using IncBack = DatumCoord<1>;

        template<std::size_t NewCoord>
        using PushFront = DatumCoord<NewCoord>;

        template<std::size_t NewCoord>
        using PushBack = DatumCoord<NewCoord>;

        template<std::size_t N>
        using Front = DatumCoord<>;

        template<std::size_t N>
        using Back = DatumCoord<>;

        template<typename OtherDatumCoord>
        using Cat = OtherDatumCoord;
    };

    template<typename T_First, typename T_Second>
    struct DatumCoordIsBigger;

    template<std::size_t... Coords1, std::size_t... Coords2>
    struct DatumCoordIsBigger<DatumCoord<Coords1...>, DatumCoord<Coords2...>>
    {
        static constexpr bool value = []() constexpr
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

    template<typename T_First, typename T_Second>
    struct DatumCoordIsSame;

    template<std::size_t... Coords1, std::size_t... Coords2>
    struct DatumCoordIsSame<DatumCoord<Coords1...>, DatumCoord<Coords2...>>
    {
        static constexpr bool value = []() constexpr
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

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

#include <tuple>
#include <type_traits>

/// Documentation of this file is in DatumCoord.dox!

namespace llama
{
    template<std::size_t... T_coords>
    struct DatumCoord;

    namespace internal
    {
        template<std::size_t T_rest, std::size_t... T_coords>
        struct DatumCoordFront;

        template<
            std::size_t T_rest,
            std::size_t T_coord,
            std::size_t... T_coords>
        struct DatumCoordFront<T_rest, T_coord, T_coords...>
        {
            using type = typename DatumCoordFront<T_rest - 1, T_coords...>::
                type::template PushFront<T_coord>;
        };

        template<std::size_t T_coord, std::size_t... T_coords>
        struct DatumCoordFront<0, T_coord, T_coords...>
        {
            using type = DatumCoord<>;
        };

        template<std::size_t T_rest, std::size_t... T_coords>
        struct DatumCoordBack;

        template<
            std::size_t T_rest,
            std::size_t T_coord,
            std::size_t... T_coords>
        struct DatumCoordBack<T_rest, T_coord, T_coords...>
        {
            using type = typename DatumCoordBack<T_rest - 1, T_coords...>::type;
        };

        template<std::size_t T_coord, std::size_t... T_coords>
        struct DatumCoordBack<0, T_coord, T_coords...>
        {
            using type = DatumCoord<T_coord, T_coords...>;
        };

        template<>
        struct DatumCoordBack<0>
        {
            using type = DatumCoord<>;
        };

    } // namespace internal

    template<std::size_t T_coord, std::size_t... T_coords>
    struct DatumCoord<T_coord, T_coords...>
    {
        using type = DatumCoord<T_coord, T_coords...>;
        static constexpr std::size_t front = T_coord;
        static constexpr std::size_t size = sizeof...(T_coords) + 1;
        static constexpr std::size_t back = DatumCoord<T_coords...>::back;
        using PopFront = DatumCoord<T_coords...>;
        using IncBack = typename PopFront::IncBack::template PushFront<front>;
        template<std::size_t T_newCoord = 0>
        using PushFront = DatumCoord<T_newCoord, T_coord, T_coords...>;
        template<std::size_t T_newCoord = 0>
        using PushBack = DatumCoord<T_coord, T_coords..., T_newCoord>;
        template<std::size_t T_size>
        using Front = typename internal::
            DatumCoordFront<T_size, T_coord, T_coords...>::type;
        template<std::size_t T_size>
        using Back = typename internal::
            DatumCoordBack<size - T_size, T_coord, T_coords...>::type;
        template<typename T_Other>
        using Cat = typename DatumCoord<T_coords...>::template Cat<
            T_Other>::template PushFront<T_coord>;
    };

    template<std::size_t T_coord>
    struct DatumCoord<T_coord>
    {
        using type = DatumCoord<T_coord>;
        static constexpr std::size_t front = T_coord;
        static constexpr std::size_t size = 1;
        static constexpr std::size_t back = T_coord;
        using PopFront = DatumCoord<>;
        using IncBack = DatumCoord<T_coord + 1>;
        template<std::size_t T_newCoord = 0>
        using PushFront = DatumCoord<T_newCoord, T_coord>;
        template<std::size_t T_newCoord = 0>
        using PushBack = DatumCoord<T_coord, T_newCoord>;
        template<std::size_t T_size>
        using Front = typename internal::DatumCoordFront<T_size, T_coord>::type;
        template<std::size_t T_size>
        using Back =
            typename internal::DatumCoordBack<size - T_size, T_coord>::type;
        template<typename T_Other>
        using Cat = typename T_Other::template PushFront<T_coord>;
    };

    template<>
    struct DatumCoord<>
    {
        using type = DatumCoord<>;
        static constexpr std::size_t size = 0;
        using IncBack = DatumCoord<1>;
        template<std::size_t T_newCoord = 0>
        using PushFront = DatumCoord<T_newCoord>;
        template<std::size_t T_newCoord = 0>
        using PushBack = DatumCoord<T_newCoord>;
        template<std::size_t T_size>
        using Front = DatumCoord<>;
        template<std::size_t T_size>
        using Back = DatumCoord<>;
        template<typename T_Other>
        using Cat = T_Other;
    };

    template<typename T_First, typename T_Second, typename T_SFinae = void>
    struct DatumCoordIsBigger;

    template<typename T_First, typename T_Second>
    struct DatumCoordIsBigger<
        T_First,
        T_Second,
        typename std::enable_if<(
            T_First::size == 1 || T_Second::size == 1)>::type>
    {
        static constexpr bool value = (T_First::front > T_Second::front);
    };

    template<typename T_First, typename T_Second>
    struct DatumCoordIsBigger<
        T_First,
        T_Second,
        typename std::enable_if<(
            T_First::size > 1 && T_Second::size > 1
            && T_First::front == T_Second::front)>::type>
    {
        static constexpr bool value = DatumCoordIsBigger<
            typename T_First::PopFront,
            typename T_Second::PopFront>::value;
    };

    template<typename T_First, typename T_Second>
    struct DatumCoordIsBigger<
        T_First,
        T_Second,
        typename std::enable_if<(
            T_First::size > 1 && T_Second::size > 1
            && T_First::front < T_Second::front)>::type>
    {
        static constexpr bool value = false;
    };

    template<typename T_First, typename T_Second>
    struct DatumCoordIsBigger<
        T_First,
        T_Second,
        typename std::enable_if<(
            T_First::size > 1 && T_Second::size > 1
            && T_First::front > T_Second::front)>::type>
    {
        static constexpr bool value = true;
    };

    template<typename T_First, typename T_Second, typename T_SFinae = void>
    struct DatumCoordIsSame;

    template<typename T_First, typename T_Second>
    struct DatumCoordIsSame<
        T_First,
        T_Second,
        typename std::enable_if<(
            T_First::size < 1 || T_Second::size < 1)>::type>
    {
        static constexpr bool value = true;
    };

    template<typename T_First, typename T_Second>
    struct DatumCoordIsSame<
        T_First,
        T_Second,
        typename std::enable_if<
            (T_First::size == 1 && T_Second::size >= 1)
            || (T_First::size >= 1 && T_Second::size == 1)>::type>
    {
        static constexpr bool value = (T_First::front == T_Second::front);
    };

    template<typename T_First, typename T_Second>
    struct DatumCoordIsSame<
        T_First,
        T_Second,
        typename std::enable_if<(
            T_First::size > 1 && T_Second::size > 1
            && T_First::front == T_Second::front)>::type>
    {
        static constexpr bool value = DatumCoordIsSame<
            typename T_First::PopFront,
            typename T_Second::PopFront>::value;
    };

    template<typename T_First, typename T_Second>
    struct DatumCoordIsSame<
        T_First,
        T_Second,
        typename std::enable_if<(
            T_First::size > 1 && T_Second::size > 1
            && T_First::front != T_Second::front)>::type>
    {
        static constexpr bool value = false;
    };

} // namespace llama

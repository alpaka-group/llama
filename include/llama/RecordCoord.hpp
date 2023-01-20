// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Meta.hpp"

#include <array>
#include <ostream>
#include <type_traits>

namespace llama
{
    /// Represents a coordinate for a record inside the record dimension tree.
    /// \tparam Coords... the compile time coordinate.
    template<std::size_t... Coords>
    struct RecordCoord
    {
        /// The list of integral coordinates as `mp_list`.
        using List = mp_list_c<std::size_t, Coords...>;

        static constexpr std::size_t front = mp_front<List>::value;
        static constexpr std::size_t back = mp_back<List>::value;
        static constexpr std::size_t size = sizeof...(Coords);
    };

    template<>
    struct RecordCoord<>
    {
        using List = mp_list_c<std::size_t>;

        static constexpr std::size_t size = 0;
    };

    template<std::size_t... CoordsA, std::size_t... CoordsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<CoordsA...>, RecordCoord<CoordsB...>)
    {
        return false;
    }

    template<std::size_t... Coords>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(RecordCoord<Coords...>, RecordCoord<Coords...>)
    {
        return true;
    }

    template<std::size_t... CoordsA, std::size_t... CoordsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(RecordCoord<CoordsA...> a, RecordCoord<CoordsB...> b)
    {
        return !(a == b);
    }

    template<typename T>
    inline constexpr bool isRecordCoord = false;

    template<std::size_t... Coords>
    inline constexpr bool isRecordCoord<RecordCoord<Coords...>> = true;

    template<std::size_t... RCs>
    auto operator<<(std::ostream& os, RecordCoord<RCs...>) -> std::ostream&
    {
        os << "RecordCoord<";
        bool first = true;
        for(auto rc : std::array<std::size_t, sizeof...(RCs)>{RCs...})
        {
            if(first)
                first = false;
            else
                os << ", ";
            os << rc;
        }
        os << ">";
        return os;
    }

    inline namespace literals
    {
        /// Literal operator for converting a numeric literal into a \ref RecordCoord.
        template<char... Digits>
        constexpr auto operator"" _RC()
        {
            constexpr auto coord = []() constexpr
            {
                const char digits[] = {(Digits - 48)...};
                std::size_t acc = 0;
                std ::size_t powerOf10 = 1;
                for(int i = sizeof...(Digits) - 1; i >= 0; i--)
                {
                    acc += digits[i] * powerOf10;
                    powerOf10 *= 10;
                }
                return acc;
            }();
            return RecordCoord<coord>{};
        }
    } // namespace literals

    /// Converts a type list of integral constants into a \ref RecordCoord.
    template<typename L>
    using RecordCoordFromList = internal::mp_unwrap_values_into<L, RecordCoord>;

    /// Concatenate a set of \ref RecordCoord%s.
    template<typename... RecordCoords>
    using Cat = RecordCoordFromList<mp_append<typename RecordCoords::List...>>;

    /// Concatenate a set of \ref RecordCoord%s instances.
    template<typename... RecordCoords>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto cat(RecordCoords...)
    {
        return Cat<RecordCoords...>{};
    }

    /// RecordCoord without first coordinate component.
    template<typename RecordCoord>
    using PopFront = RecordCoordFromList<mp_pop_front<typename RecordCoord::List>>;

    namespace internal
    {
        template<std::size_t... Coords1, std::size_t... Coords2>
        constexpr auto recordCoordCommonPrefixIsBiggerImpl(RecordCoord<Coords1...>, RecordCoord<Coords2...>) -> bool
        {
            // CTAD does not work if Coords1/2 is an empty pack
            std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
            std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
            for(std::size_t i = 0; i < std::min(a1.size(), a2.size()); i++)
            {
                if(a1[i] > a2[i])
                    return true;
                if(a1[i] < a2[i])
                    return false;
            }
            return false;
        };
    } // namespace internal

    /// Checks wether the first RecordCoord is bigger than the second.
    template<typename First, typename Second>
    inline constexpr auto recordCoordCommonPrefixIsBigger
        = internal::recordCoordCommonPrefixIsBiggerImpl(First{}, Second{});

    namespace internal
    {
        template<std::size_t... Coords1, std::size_t... Coords2>
        constexpr auto recordCoordCommonPrefixIsSameImpl(RecordCoord<Coords1...>, RecordCoord<Coords2...>) -> bool
        {
            // CTAD does not work if Coords1/2 is an empty pack
            std::array<std::size_t, sizeof...(Coords1)> a1{Coords1...};
            std::array<std::size_t, sizeof...(Coords2)> a2{Coords2...};
            for(std::size_t i = 0; i < std::min(a1.size(), a2.size()); i++)
                if(a1[i] != a2[i])
                    return false;
            return true;
        };
    } // namespace internal

    /// Checks whether two \ref RecordCoord%s are the same or one is the prefix of the other.
    template<typename First, typename Second>
    inline constexpr auto recordCoordCommonPrefixIsSame
        = internal::recordCoordCommonPrefixIsSameImpl(First{}, Second{});
} // namespace llama

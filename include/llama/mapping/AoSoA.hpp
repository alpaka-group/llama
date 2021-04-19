// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// The maximum number of vector lanes that can be used to fetch each leaf type in the datum domain into a vector
    /// register of the given size in bits.
    template <typename DatumDomain, std::size_t VectorRegisterBits>
    inline constexpr std::size_t maxLanes = []() constexpr
    {
        auto max = std::numeric_limits<std::size_t>::max();
        forEachLeaf<DatumDomain>(
            [&](auto coord)
            {
                using AttributeType = GetType<DatumDomain, decltype(coord)>;
                max = std::min(max, VectorRegisterBits / (sizeof(AttributeType) * CHAR_BIT));
            });
        return max;
    }
    ();

    /// Array of struct of arrays mapping. Used to create a \ref View via \ref
    /// allocView.
    /// \tparam Lanes The size of the inner arrays of this array of struct of
    /// arrays.
    /// \tparam LinearizeArrayDomainFunctor Defines how the
    /// user domain should be mapped into linear numbers and how big the linear
    /// domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        std::size_t Lanes,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct AoSoA
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        constexpr AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr AoSoA(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            const auto flatArrayIndex = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize);
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = (sizeOf<DatumDomain> * Lanes) * blockIndex
                + offsetOf<DatumDomain, DatumCoord<DatumDomainCoord...>> * Lanes
                + sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>) * laneIndex;
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };

    template <std::size_t Lanes, typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct PreconfiguredAoSoA
    {
        template <typename ArrayDomain, typename DatumDomain>
        using type = AoSoA<ArrayDomain, DatumDomain, Lanes, LinearizeArrayDomainFunctor>;
    };
} // namespace llama::mapping
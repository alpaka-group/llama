// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Types.hpp"
#include "Common.hpp"

namespace llama::mapping
{
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

        AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        AoSoA(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            const auto flatArrayIndex = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize);
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = (sizeOf<DatumDomain> * Lanes) * blockIndex
                + offsetOf<DatumDomain, DatumDomainCoord...> * Lanes
                + sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>) * laneIndex;
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };
} // namespace llama::mapping

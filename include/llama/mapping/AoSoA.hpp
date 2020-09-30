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
    /// \tparam LinearizeUserDomainFunctor Defines how the
    /// user domain should be mapped into linear numbers and how big the linear
    /// domain gets.
    template <
        typename T_UserDomain,
        typename T_DatumDomain,
        std::size_t Lanes,
        typename LinearizeUserDomainFunctor = LinearizeUserDomainCpp>
    struct AoSoA
    {
        using ArrayDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        AoSoA(ArrayDomain size, DatumDomain = {}) : userDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const -> std::size_t
        {
            return LinearizeUserDomainFunctor {}.size(userDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            const auto userDomainIndex = LinearizeUserDomainFunctor {}(coord, userDomainSize);
            const auto blockIndex = userDomainIndex / Lanes;
            const auto laneIndex = userDomainIndex % Lanes;
            const auto offset = (sizeOf<DatumDomain> * Lanes) * blockIndex
                + offsetOf<DatumDomain, DatumDomainCoord...> * Lanes
                + sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>) * laneIndex;
            return {0, offset};
        }

        ArrayDomain userDomainSize;
    };
} // namespace llama::mapping

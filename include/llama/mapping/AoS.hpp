// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Types.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam LinearizeUserDomainAdressFunctor Defines how the
    /// user domain should be mapped into linear numbers.
    /// \tparam ExtentUserDomainAdressFunctor Defines how the total number of
    /// \ref UserDomain indices is calculated.
    template<
        typename T_UserDomain,
        typename T_DatumDomain,
        typename LinearizeUserDomainAdressFunctor = LinearizeUserDomainAdress,
        typename ExtentUserDomainAdressFunctor = ExtentUserDomainAdress>
    struct AoS
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        AoS(UserDomain size) : userDomainSize(size) {}

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const
            -> std::size_t
        {
            return ExtentUserDomainAdressFunctor{}(
                userDomainSize)*sizeOf<DatumDomain>;
        }

        template<std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain coord) const
            -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto offset
                = LinearizeUserDomainAdressFunctor{}(coord, userDomainSize)
                    * sizeOf<
                        DatumDomain> + offsetOf<DatumDomain, DatumDomainCoord...>;
            return {0, offset};
        }

        UserDomain userDomainSize;
    };
}

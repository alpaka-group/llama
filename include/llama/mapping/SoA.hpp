// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Functions.hpp"
#include "../Types.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam LinearizeUserDomainFunctor Defines how the
    /// user domain should be mapped into linear numbers and how big the linear
    /// domain gets.
    template<
        typename T_UserDomain,
        typename T_DatumDomain,
        typename LinearizeUserDomainFunctor = LinearizeUserDomainCpp>
    struct SoA
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        SoA(UserDomain size) : userDomainSize(size) {}

        LLAMA_FN_HOST_ACC_INLINE
        auto getBlobSize(std::size_t const) const -> std::size_t
        {
            return LinearizeUserDomainFunctor{}.size(userDomainSize)
                * sizeOf<DatumDomain>;
        }

        template<std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain coord) const
            -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto offset
                = LinearizeUserDomainFunctor{}(coord, userDomainSize)
                    * sizeof(
                        GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>)
                + offsetOf<
                      DatumDomain,
                      DatumDomainCoord...> * LinearizeUserDomainFunctor{}.size(userDomainSize);
            return {0, offset};
        }

        UserDomain userDomainSize = {};
    };
}

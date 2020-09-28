// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Functions.hpp"
#include "../Types.hpp"

namespace llama::mapping
{
    /// Maps all UserDomain coordinates into the same location and layouts
    /// struct members consecutively. This mapping is used for temporary, single
    /// element views.
    template <typename T_UserDomain, typename T_DatumDomain>
    struct One
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;

        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const -> std::size_t
        {
            return sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain) const -> NrAndOffset
        {
            constexpr auto offset = offsetOf<DatumDomain, DatumDomainCoord...>;
            return {0, offset};
        }
    };
} // namespace llama::mapping

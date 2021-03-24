// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"

namespace llama::mapping
{
    /// Maps all ArrayDomain coordinates into the same location and layouts
    /// struct members consecutively. This mapping is used for temporary, single
    /// element views.
    template <typename T_ArrayDomain, typename T_DatumDomain>
    struct One
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;

        static constexpr std::size_t blobCount = 1;

        constexpr One() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr One(ArrayDomain, DatumDomain = {})
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDomain) const -> NrAndOffset
        {
            constexpr auto offset = offsetOf<DatumDomain, DatumCoord<DatumDomainCoord...>>;
            return {0, offset};
        }
    };
} // namespace llama::mapping

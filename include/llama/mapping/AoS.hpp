// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam LinearizeArrayDomainFunctor Defines how the
    /// user domain should be mapped into linear numbers and how big the linear
    /// domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct AoS
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = MakeDatumDomain<T_DatumDomain>;
        static constexpr std::size_t blobCount = 1;

        constexpr AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr AoS(ArrayDomain size, T_DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto offset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize)
                    * sizeOf<DatumDomain> + offsetOf<DatumDomain, DatumCoord<DatumDomainCoord...>>;
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };
} // namespace llama::mapping

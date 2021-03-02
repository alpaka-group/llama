// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If false, struct members are tighly packed.
    /// \tparam LinearizeArrayDomainFunctor Defines how the user domain
    /// should be mapped into linear numbers and how big the linear domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        bool AlignAndPad = false,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct AoS
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        constexpr AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr AoS(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain, AlignAndPad>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto offset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize)
                    * sizeOf<DatumDomain,
                             AlignAndPad> + offsetOf<DatumDomain, DatumCoord<DatumDomainCoord...>, AlignAndPad>;
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };

    template <
        typename ArrayDomain,
        typename DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using AlignedAoS = AoS<ArrayDomain, DatumDomain, true, LinearizeArrayDomainFunctor>;

    template <
        typename ArrayDomain,
        typename DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using PackedAoS = AoS<ArrayDomain, DatumDomain, false, LinearizeArrayDomainFunctor>;

    template <bool AlignAndPad = false, typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct PreconfiguredAoS
    {
        template <typename ArrayDomain, typename DatumDomain>
        using type = AoS<ArrayDomain, DatumDomain, AlignAndPad, LinearizeArrayDomainFunctor>;
    };
} // namespace llama::mapping

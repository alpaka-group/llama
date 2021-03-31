// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If false, struct members are tighly packed.
    /// \tparam LinearizeArrayDomainFunctor Defines how the array domain
    /// should be mapped into linear numbers and how big the linear domain gets.
    template <
        typename T_ArrayDomain,
        typename T_RecordDim,
        bool AlignAndPad = false,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct AoS
    {
        using ArrayDomain = T_ArrayDomain;
        using RecordDim = T_RecordDim;
        static constexpr std::size_t blobCount = 1;

        constexpr AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr AoS(ArrayDomain size, RecordDim = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<RecordDim, AlignAndPad>;
        }

        template <std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            const auto offset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize)
                    * sizeOf<RecordDim, AlignAndPad> + offsetOf<RecordDim, RecordCoord<RecordCoords...>, AlignAndPad>;
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };

    /// Array of struct mapping preserving the alignment of the element types by inserting padding. See \see AoS.
    template <typename ArrayDomain, typename RecordDim, typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using AlignedAoS = AoS<ArrayDomain, RecordDim, true, LinearizeArrayDomainFunctor>;

    /// Array of struct mapping packing the element types tighly, violating the types alignment requirements. See \see
    /// AoS.
    template <typename ArrayDomain, typename RecordDim, typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using PackedAoS = AoS<ArrayDomain, RecordDim, false, LinearizeArrayDomainFunctor>;

    template <bool AlignAndPad = false, typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct PreconfiguredAoS
    {
        template <typename ArrayDomain, typename RecordDim>
        using type = AoS<ArrayDomain, RecordDim, AlignAndPad, LinearizeArrayDomainFunctor>;
    };
} // namespace llama::mapping

// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If false, struct members are tighly packed.
    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    template <
        typename T_ArrayDims,
        typename T_RecordDim,
        bool AlignAndPad = false,
        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct AoS
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        static constexpr std::size_t blobCount = 1;

        constexpr AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr AoS(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim, AlignAndPad>;
        }

        template <std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
        {
            const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                    * sizeOf<RecordDim, AlignAndPad> + offsetOf<RecordDim, RecordCoord<RecordCoords...>, AlignAndPad>;
            return {0, offset};
        }

        ArrayDims arrayDimsSize;
    };

    /// Array of struct mapping preserving the alignment of the element types by inserting padding. See \see AoS.
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using AlignedAoS = AoS<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

    /// Array of struct mapping packing the element types tighly, violating the types alignment requirements. See \see
    /// AoS.
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using PackedAoS = AoS<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

    template <bool AlignAndPad = false, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct PreconfiguredAoS
    {
        template <typename ArrayDims, typename RecordDim>
        using type = AoS<ArrayDims, RecordDim, AlignAndPad, LinearizeArrayDimsFunctor>;
    };
} // namespace llama::mapping

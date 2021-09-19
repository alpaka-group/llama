// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If false, struct members are tightly packed.
    /// \tparam T_LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
    /// \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayDims,
        typename TRecordDim,
        bool AlignAndPad = true,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDim = FlattenRecordDimInOrder>
    struct AoS
    {
        using ArrayDims = TArrayDims;
        using RecordDim = TRecordDim;
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        static constexpr std::size_t blobCount = 1;

        constexpr AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit AoS(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return arrayDimsSize;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDimsFunctor{}.size(arrayDimsSize)
                * flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad>;
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord, RecordCoord<RecordCoords...> = {})
            const -> NrAndOffset
        {
            constexpr std::size_t flatFieldIndex =
#ifdef __NVCC__
                *& // mess with nvcc compiler state to workaround bug
#endif
                 Flattener::template flatIndex<RecordCoords...>;
            const auto offset
                = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                    * flatSizeOf<
                        typename Flattener::FlatRecordDim,
                        AlignAndPad> + flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, AlignAndPad>;
            return {0, offset};
        }

    private:
        using Flattener = FlattenRecordDim<TRecordDim>;
        ArrayDims arrayDimsSize;
    };

    /// Array of struct mapping preserving the alignment of the field types by inserting padding.
    /// \see AoS
    template<typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using AlignedAoS = AoS<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

    /// Array of struct mapping preserving the alignment of the field types by inserting padding and permuting the
    /// field order to minimize this padding. \see AoS
    template<typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MinAlignedAoS = AoS<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor, FlattenRecordDimMinimizePadding>;

    /// Array of struct mapping packing the field types tightly, violating the types alignment requirements.
    /// \see AoS
    template<typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using PackedAoS = AoS<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

    template<bool AlignAndPad = true, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct PreconfiguredAoS
    {
        template<typename ArrayDims, typename RecordDim>
        using type = AoS<ArrayDims, RecordDim, AlignAndPad, LinearizeArrayDimsFunctor>;
    };

    template<typename Mapping>
    inline constexpr bool isAoS = false;

    template<
        typename ArrayDims,
        typename RecordDim,
        bool AlignAndPad,
        typename LinearizeArrayDimsFunctor,
        template<typename>
        typename FlattenRecordDim>
    inline constexpr bool
        isAoS<AoS<ArrayDims, RecordDim, AlignAndPad, LinearizeArrayDimsFunctor, FlattenRecordDim>> = true;
} // namespace llama::mapping

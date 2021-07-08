// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Maps all ArrayDims coordinates into the same location and layouts struct members consecutively. This mapping is
    /// used for temporary, single element views.
    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If false, struct members are tightly packed.
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
    /// FlattenRecordDimInOrder and \ref FlattenRecordDimMinimizePadding.
    template<
        typename T_ArrayDims,
        typename T_RecordDim,
        bool AlignAndPad = true,
        template<typename> typename FlattenRecordDim = FlattenRecordDimMinimizePadding>
    struct One
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;

        static constexpr std::size_t blobCount = 1;

        constexpr One() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr One(ArrayDims, RecordDim = {})
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            // TODO: not sure if this is the right approach, since we take any ArrayDims in the ctor
            ArrayDims ad;
            for(auto i = 0; i < ArrayDims::rank; i++)
                ad[i] = 1;
            return ad;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad, false>; // no tail padding
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims) const -> NrAndOffset
        {
            constexpr std::size_t flatIndex =
#ifdef __NVCC__
                Flattener{}.template flatIndex<RecordCoords...>;
#else
                Flattener::template flatIndex<RecordCoords...>;
#endif
            constexpr auto offset = flatOffsetOf<typename Flattener::FlatRecordDim, flatIndex, AlignAndPad>;
            return {0, offset};
        }

    private:
        using Flattener = FlattenRecordDim<T_RecordDim>;
    };

    /// One mapping preserving the alignment of the field types by inserting padding.
    /// \see One
    template<typename ArrayDims, typename RecordDim>
    using AlignedOne = One<ArrayDims, RecordDim, true, FlattenRecordDimInOrder>;

    /// One mapping preserving the alignment of the field types by inserting padding and permuting the field order to
    /// minimize this padding.
    /// \see One
    template<typename ArrayDims, typename RecordDim>
    using MinAlignedOne = One<ArrayDims, RecordDim, true, FlattenRecordDimMinimizePadding>;

    /// One mapping packing the field types tightly, violating the types' alignment requirements.
    /// \see One
    template<typename ArrayDims, typename RecordDim>
    using PackedOne = One<ArrayDims, RecordDim, false, FlattenRecordDimInOrder>;
} // namespace llama::mapping

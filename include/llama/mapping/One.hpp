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
    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
    /// \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayDims,
        typename TRecordDim,
        bool AlignAndPad = true,
        template<typename> typename FlattenRecordDim = FlattenRecordDimMinimizePadding>
    struct One
    {
        using ArrayDims = TArrayDims;
        using RecordDim = TRecordDim;

        static constexpr std::size_t blobCount = 1;

        constexpr One() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit One(ArrayDims, RecordDim = {})
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            // TODO(bgruber): not sure if this is the right approach, since we take any ArrayDims in the ctor
            ArrayDims ad;
            if constexpr(ArrayDims::rank > 0)
                for(auto i = 0; i < ArrayDims::rank; i++)
                    ad[i] = 1;
            return ad;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad, false>; // no tail padding
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims, RecordCoord<RecordCoords...> = {}) const
            -> NrAndOffset
        {
            constexpr std::size_t flatFieldIndex =
#ifdef __NVCC__
                *& // mess with nvcc compiler state to workaround bug
#endif
                 Flattener::template flatIndex<RecordCoords...>;
            constexpr auto offset = flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, AlignAndPad>;
            return {0, offset};
        }

    private:
        using Flattener = FlattenRecordDim<TRecordDim>;
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

    template<typename Mapping>
    inline constexpr bool isOne = false;

    template<typename ArrayDims, typename RecordDim, bool AlignAndPad, template<typename> typename FlattenRecordDim>
    inline constexpr bool isOne<One<ArrayDims, RecordDim, AlignAndPad, FlattenRecordDim>> = true;
} // namespace llama::mapping

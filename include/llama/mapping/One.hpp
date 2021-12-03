// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Maps all array dimension indices to the same location and layouts struct members consecutively. This mapping is
    /// used for temporary, single element views.
    /// \tparam AlignAndPad If true, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If false, struct members are tightly packed.
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
    /// \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        bool AlignAndPad = true,
        template<typename> typename FlattenRecordDim = FlattenRecordDimMinimizePadding>
    struct One : TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;

        static constexpr std::size_t blobCount = 1;

        constexpr One() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit One(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return ArrayExtents{*this};
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return flatSizeOf<typename Flattener::FlatRecordDim, AlignAndPad, false>; // no tail padding
        }

        template<std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            ArrayIndex,
            Array<std::size_t, N> = {},
            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset
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
    template<typename ArrayExtents, typename RecordDim>
    using AlignedOne = One<ArrayExtents, RecordDim, true, FlattenRecordDimInOrder>;

    /// One mapping preserving the alignment of the field types by inserting padding and permuting the field order to
    /// minimize this padding.
    /// \see One
    template<typename ArrayExtents, typename RecordDim>
    using MinAlignedOne = One<ArrayExtents, RecordDim, true, FlattenRecordDimMinimizePadding>;

    /// One mapping packing the field types tightly, violating the types' alignment requirements.
    /// \see One
    template<typename ArrayExtents, typename RecordDim>
    using PackedOne = One<ArrayExtents, RecordDim, false, FlattenRecordDimInOrder>;

    template<typename Mapping>
    inline constexpr bool isOne = false;

    template<typename ArrayExtents, typename RecordDim, bool AlignAndPad, template<typename> typename FlattenRecordDim>
    inline constexpr bool isOne<One<ArrayExtents, RecordDim, AlignAndPad, FlattenRecordDim>> = true;
} // namespace llama::mapping

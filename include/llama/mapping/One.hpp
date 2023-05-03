// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "../Core.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Maps all array dimension indices to the same location and layouts struct members consecutively. This mapping is
    /// used for temporary, single element views.
    /// \tparam TFieldAlignment If Align, padding bytes are inserted to guarantee that struct members are properly
    /// aligned. If false, struct members are tightly packed.
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
    /// \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        FieldAlignment TFieldAlignment = FieldAlignment::Align,
        template<typename> typename FlattenRecordDim = FlattenRecordDimMinimizePadding>
    struct One : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename Base::size_type;

    public:
        inline static constexpr FieldAlignment fieldAlignment = TFieldAlignment;
        using Flattener = FlattenRecordDim<TRecordDim>;
        static constexpr std::size_t blobCount = 1;

#ifndef __NVCC__
        using Base::Base;
#else
        constexpr One() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr explicit One(TArrayExtents extents, TRecordDim = {}) : Base(extents)
        {
        }
#endif

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
        {
            return flatSizeOf<
                typename Flattener::FlatRecordDim,
                fieldAlignment == FieldAlignment::Align,
                false>; // no tail padding
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            typename Base::ArrayIndex,
            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
        {
            constexpr std::size_t flatFieldIndex =
#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6
                *& // mess with nvcc compiler state to workaround bug
#endif
                 Flattener::template flatIndex<RecordCoords...>;
            constexpr auto offset = static_cast<size_type>(flatOffsetOf<
                                                           typename Flattener::FlatRecordDim,
                                                           flatFieldIndex,
                                                           fieldAlignment == FieldAlignment::Align>);
            return {size_type{0}, offset};
        }
    };

    /// One mapping preserving the alignment of the field types by inserting padding.
    /// \see One
    template<typename ArrayExtents, typename RecordDim>
    using AlignedOne = One<ArrayExtents, RecordDim, FieldAlignment::Align, FlattenRecordDimInOrder>;

    /// One mapping preserving the alignment of the field types by inserting padding and permuting the field order to
    /// minimize this padding.
    /// \see One
    template<typename ArrayExtents, typename RecordDim>
    using MinAlignedOne = One<ArrayExtents, RecordDim, FieldAlignment::Align, FlattenRecordDimMinimizePadding>;

    /// One mapping packing the field types tightly, violating the types' alignment requirements.
    /// \see One
    template<typename ArrayExtents, typename RecordDim>
    using PackedOne = One<ArrayExtents, RecordDim, FieldAlignment::Pack, FlattenRecordDimInOrder>;

    /// Binds parameters to a \ref One mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        FieldAlignment FieldAlignment = FieldAlignment::Align,
        template<typename> typename FlattenRecordDim = FlattenRecordDimMinimizePadding>
    struct BindOne
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = One<ArrayExtents, RecordDim, FieldAlignment, FlattenRecordDim>;
    };

    template<typename Mapping>
    inline constexpr bool isOne = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        FieldAlignment FieldAlignment,
        template<typename>
        typename FlattenRecordDim>
    inline constexpr bool isOne<One<ArrayExtents, RecordDim, FieldAlignment, FlattenRecordDim>> = true;
} // namespace llama::mapping

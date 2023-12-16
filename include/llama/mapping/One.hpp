// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "../Core.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Maps all array dimension indices to the same location and layouts struct members consecutively. This mapping is
    /// used for temporary, single element views.
    /// \tparam TFieldAlignment If Align, padding bytes are inserted to guarantee that struct members are properly
    /// aligned. If false, struct members are tightly packed.
    /// \tparam PermuteFields Defines how the record dimension's fields should be permuted. See \ref
    /// PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref PermuteFieldsDecreasingAlignment and
    /// \ref PermuteFieldsMinimizePadding.
    LLAMA_EXPORT
    template<
        typename TArrayExtents,
        typename TRecordDim,
        FieldAlignment TFieldAlignment = FieldAlignment::Align,
        template<typename> typename PermuteFields = PermuteFieldsMinimizePadding>
    struct One : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename Base::size_type;

    public:
        inline static constexpr FieldAlignment fieldAlignment = TFieldAlignment;
        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
        static constexpr std::size_t blobCount = 1;

#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ >= 12
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
                typename Permuter::FlatRecordDim,
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
                 Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>;
            constexpr auto offset = static_cast<size_type>(flatOffsetOf<
                                                           typename Permuter::FlatRecordDim,
                                                           flatFieldIndex,
                                                           fieldAlignment == FieldAlignment::Align>);
            return {size_type{0}, offset};
        }
    };

    /// One mapping preserving the alignment of the field types by inserting padding.
    /// \see One
    LLAMA_EXPORT
    template<typename ArrayExtents, typename RecordDim>
    using AlignedOne = One<ArrayExtents, RecordDim, FieldAlignment::Align, PermuteFieldsInOrder>;

    /// One mapping preserving the alignment of the field types by inserting padding and permuting the field order to
    /// minimize this padding.
    /// \see One
    LLAMA_EXPORT
    template<typename ArrayExtents, typename RecordDim>
    using MinAlignedOne = One<ArrayExtents, RecordDim, FieldAlignment::Align, PermuteFieldsMinimizePadding>;

    /// One mapping packing the field types tightly, violating the types' alignment requirements.
    /// \see One
    LLAMA_EXPORT
    template<typename ArrayExtents, typename RecordDim>
    using PackedOne = One<ArrayExtents, RecordDim, FieldAlignment::Pack, PermuteFieldsInOrder>;

    /// Binds parameters to a \ref One mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    LLAMA_EXPORT
    template<
        FieldAlignment FieldAlignment = FieldAlignment::Align,
        template<typename> typename PermuteFields = PermuteFieldsMinimizePadding>
    struct BindOne
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = One<ArrayExtents, RecordDim, FieldAlignment, PermuteFields>;
    };

    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool isOne = false;

    LLAMA_EXPORT
    template<
        typename ArrayExtents,
        typename RecordDim,
        FieldAlignment FieldAlignment,
        template<typename>
        typename PermuteFields>
    inline constexpr bool isOne<One<ArrayExtents, RecordDim, FieldAlignment, PermuteFields>> = true;
} // namespace llama::mapping

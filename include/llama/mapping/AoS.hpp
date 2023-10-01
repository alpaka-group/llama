// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam Alignment If Align, padding bytes are inserted to guarantee that struct members are properly aligned.
    /// If Pack, struct members are tightly packed.
    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam PermuteFields Defines how the record dimension's fields should be permuted. See \ref
    /// PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref PermuteFieldsDecreasingAlignment and
    /// \ref PermuteFieldsMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        FieldAlignment TFieldAlignment = FieldAlignment::Align,
        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
        template<typename> typename PermuteFields = PermuteFieldsInOrder>
    struct AoS : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename Base::size_type;

    public:
        inline static constexpr FieldAlignment fieldAlignment = TFieldAlignment;
        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
        inline static constexpr std::size_t blobCount = 1;

        using Base::Base;

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
        {
            return LinearizeArrayIndexFunctor{}.size(Base::extents())
                * flatSizeOf<typename Permuter::FlatRecordDim, fieldAlignment == FieldAlignment::Align>;
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            typename Base::ArrayIndex ai,
            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
        {
            constexpr std::size_t flatFieldIndex =
#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6
                *& // mess with nvcc compiler state to workaround bug
#endif
                 Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>;
            const auto offset
                = LinearizeArrayIndexFunctor{}(ai, Base::extents())
                    * static_cast<size_type>(
                        flatSizeOf<typename Permuter::FlatRecordDim, fieldAlignment == FieldAlignment::Align>)
                + static_cast<size_type>(flatOffsetOf<
                                         typename Permuter::FlatRecordDim,
                                         flatFieldIndex,
                                         fieldAlignment == FieldAlignment::Align>);
            return {size_type{0}, offset};
        }
    };

    // we can drop this when inherited ctors also inherit deduction guides
    template<typename TArrayExtents, typename TRecordDim>
    AoS(TArrayExtents, TRecordDim) -> AoS<TArrayExtents, TRecordDim>;

    /// Array of struct mapping preserving the alignment of the field types by inserting padding.
    /// \see AoS
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    using AlignedAoS = AoS<ArrayExtents, RecordDim, FieldAlignment::Align, LinearizeArrayIndexFunctor>;

    /// Array of struct mapping preserving the alignment of the field types by inserting padding and permuting the
    /// field order to minimize this padding. \see AoS
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    using MinAlignedAoS = AoS<
        ArrayExtents,
        RecordDim,
        FieldAlignment::Align,
        LinearizeArrayIndexFunctor,
        PermuteFieldsMinimizePadding>;

    /// Array of struct mapping packing the field types tightly, violating the type's alignment requirements.
    /// \see AoS
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    using PackedAoS = AoS<ArrayExtents, RecordDim, FieldAlignment::Pack, LinearizeArrayIndexFunctor>;

    /// Binds parameters to an \ref AoS mapping except for array and record dimension, producing a quoted meta
    /// function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        FieldAlignment Alignment = FieldAlignment::Align,
        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    struct BindAoS
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = AoS<ArrayExtents, RecordDim, Alignment, LinearizeArrayIndexFunctor>;
    };

    template<typename Mapping>
    inline constexpr bool isAoS = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        FieldAlignment FieldAlignment,
        typename LinearizeArrayIndexFunctor,
        template<typename>
        typename PermuteFields>
    inline constexpr bool
        isAoS<AoS<ArrayExtents, RecordDim, FieldAlignment, LinearizeArrayIndexFunctor, PermuteFields>>
        = true;
} // namespace llama::mapping

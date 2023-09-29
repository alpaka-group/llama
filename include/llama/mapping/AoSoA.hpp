// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// The maximum number of vector lanes that can be used to fetch each leaf type in the record dimension into a
    /// vector register of the given size in bits.
    template<typename RecordDim, std::size_t VectorRegisterBits>
    inline constexpr std::size_t maxLanes = []() constexpr
    {
        auto max = std::numeric_limits<std::size_t>::max();
        forEachLeafCoord<RecordDim>(
            [&](auto rc)
            {
                using AttributeType = GetType<RecordDim, decltype(rc)>;
                max = std::min(max, VectorRegisterBits / (sizeof(AttributeType) * CHAR_BIT));
            });
        return max;
    }();

    /// Array of struct of arrays mapping. Used to create a \ref View via \ref allocView.
    /// \tparam Lanes The size of the inner arrays of this array of struct of arrays.
    /// \tparam PermuteFields Defines how the record dimension's fields should be permuted. See \ref
    /// PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref PermuteFieldsDecreasingAlignment and
    /// \ref PermuteFieldsMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename TArrayExtents::value_type Lanes,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename PermuteFields = PermuteFieldsInOrder>
    struct AoSoA : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename Base::size_type;

    public:
        inline static constexpr typename TArrayExtents::value_type lanes = Lanes;
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
        inline static constexpr std::size_t blobCount = 1;

#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ >= 12
        using Base::Base;
#else
        constexpr AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr explicit AoSoA(TArrayExtents extents, TRecordDim = {}) : Base(extents)
        {
        }
#endif

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(size_type) const -> size_type
        {
            const auto rs = static_cast<size_type>(sizeOf<TRecordDim>);
            return roundUpToMultiple(LinearizeArrayDimsFunctor{}.size(Base::extents()) * rs, Lanes * rs);
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
            const auto flatArrayIndex = LinearizeArrayDimsFunctor{}(ai, Base::extents());
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = static_cast<size_type>(sizeOf<TRecordDim> * Lanes) * blockIndex
                + static_cast<size_type>(flatOffsetOf<typename Permuter::FlatRecordDim, flatFieldIndex, false>) * Lanes
                + static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>)) * laneIndex;
            return {0, offset};
        }
    };

    /// Binds parameters to an \ref AoSoA mapping except for array and record dimension, producing a quoted meta
    /// function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<std::size_t Lanes, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct BindAoSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = AoSoA<ArrayExtents, RecordDim, Lanes, LinearizeArrayDimsFunctor>;
    };

    template<typename Mapping>
    inline constexpr bool isAoSoA = false;

    template<typename AD, typename RD, typename AD::value_type L>
    inline constexpr bool isAoSoA<AoSoA<AD, RD, L>> = true;

} // namespace llama::mapping

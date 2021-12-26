// SPDX-License-Identifier: GPL-3.0-or-later

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
    }
    ();

    /// Array of struct of arrays mapping. Used to create a \ref View via \ref allocView.
    /// \tparam Lanes The size of the inner arrays of this array of struct of arrays.
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
    /// FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
    /// \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        std::size_t Lanes,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDim = FlattenRecordDimInOrder>
    struct AoSoA : private TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        static constexpr std::size_t blobCount = 1;

        constexpr AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr explicit AoSoA(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return ArrayExtents{*this};
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return roundUpToMultiple(
                LinearizeArrayDimsFunctor{}.size(extents()) * sizeOf<RecordDim>,
                Lanes * sizeOf<RecordDim>);
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> = {}) const
            -> NrAndOffset
        {
            constexpr std::size_t flatFieldIndex =
#ifdef __NVCC__
                *& // mess with nvcc compiler state to workaround bug
#endif
                 Flattener::template flatIndex<RecordCoords...>;
            const auto flatArrayIndex = LinearizeArrayDimsFunctor{}(ai, extents());
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = (sizeOf<RecordDim> * Lanes) * blockIndex
                + flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, false> * Lanes
                + sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>) * laneIndex;
            return {0, offset};
        }

    private:
        using Flattener = FlattenRecordDim<TRecordDim>;
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

    template<typename AD, typename RD, std::size_t L>
    inline constexpr bool isAoSoA<AoSoA<AD, RD, L>> = true;

} // namespace llama::mapping

// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam SeparateBuffers If true, every element of the record dimension is mapped to its own buffer.
    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    template <
        typename T_ArrayDims,
        typename T_RecordDim,
        bool SeparateBuffers = false,
        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct SoA
    {
        using ArrayDims = T_ArrayDims;
        using RecordDim = T_RecordDim;
        static constexpr std::size_t blobCount = []() constexpr
        {
            if constexpr (SeparateBuffers)
                return boost::mp11::mp_size<llama::FlattenRecordDim<RecordDim>>::value;
            else
                return 1;
        }
        ();

        constexpr SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr SoA(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(std::size_t blobIndex) const -> std::size_t
        {
            if constexpr (SeparateBuffers)
            {
                constexpr llama::Array<std::size_t, blobCount> typeSizes = []() constexpr
                {
                    llama::Array<std::size_t, blobCount> r{};
                    std::size_t i = 0;
                    forEachLeaf<RecordDim>([&](auto coord) constexpr
                                           { r[i++] = sizeof(GetType<RecordDim, decltype(coord)>); });
                    return r;
                }
                ();
                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * typeSizes[blobIndex];
            }
            else
                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim>;
        }

        template <std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
        {
            if constexpr (SeparateBuffers)
            {
                using TargetRecordCoord = RecordCoord<RecordCoords...>;
                constexpr auto blob = [&]() constexpr
                {
                    std::size_t index = 0;
                    bool found = false;
                    forEachLeaf<RecordDim>([&](auto c) constexpr
                                           {
                                               if constexpr (std::is_same_v<decltype(c), TargetRecordCoord>)
                                                   found = true;
                                               else if (!found)
                                                   index++;
                                           });
                    if (!found)
                        return std::numeric_limits<std::size_t>::max();
                    return index;
                }
                ();
                static_assert(
                    blob != std::numeric_limits<std::size_t>::max(),
                    "Passed TargetRecordCoord must be in record dimension");

                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                    * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>);
                return {blob, offset};
            }
            else
            {
                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                        * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>)
                    + offsetOf<
                          RecordDim,
                          RecordCoord<RecordCoords...>> * LinearizeArrayDimsFunctor{}.size(arrayDimsSize);
                return {0, offset};
            }
        }

        ArrayDims arrayDimsSize;
    };

    /// Struct of array mapping storing the entire layout in a single blob. See \see SoA.
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using SingleBlobSoA = SoA<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing each attribute of the record dimension in a separate blob. See \see SoA.
    template <typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MultiBlobSoA = SoA<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

    template <bool SeparateBuffers = false, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct PreconfiguredSoA
    {
        template <typename ArrayDims, typename RecordDim>
        using type = SoA<ArrayDims, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>;
    };
} // namespace llama::mapping

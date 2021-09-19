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
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened if SeparateBuffers is
    /// false. See \ref FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref
    /// FlattenRecordDimDecreasingAlignment and \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayDims,
        typename TRecordDim,
        bool SeparateBuffers = true,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDimSingleBlob = FlattenRecordDimInOrder>
    struct SoA
    {
        using ArrayDims = TArrayDims;
        using RecordDim = TRecordDim;
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        static constexpr std::size_t blobCount
            = SeparateBuffers ? boost::mp11::mp_size<FlatRecordDim<RecordDim>>::value : 1;

        constexpr SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit SoA(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return arrayDimsSize;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(std::size_t blobIndex) const -> std::size_t
        {
            if constexpr(SeparateBuffers)
            {
                constexpr Array<std::size_t, blobCount> typeSizes = []() constexpr
                {
                    Array<std::size_t, blobCount> r{};
                    forEachLeaf<RecordDim>([&r, i = 0](auto coord) mutable constexpr
                                           { r[i++] = sizeof(GetType<RecordDim, decltype(coord)>); });
                    return r;
                }
                ();
                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * typeSizes[blobIndex];
            }
            else
            {
                return LinearizeArrayDimsFunctor{}.size(arrayDimsSize) * sizeOf<RecordDim>;
            }
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDims coord, RecordCoord<RecordCoords...> = {})
            const -> NrAndOffset
        {
            if constexpr(SeparateBuffers)
            {
                constexpr auto blob = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                    * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>);
                return {blob, offset};
            }
            else
            {
                constexpr std::size_t flatFieldIndex =
#ifdef __NVCC__
                    *& // mess with nvcc compiler state to workaround bug
#endif
                     Flattener::template flatIndex<RecordCoords...>;
                const auto offset = LinearizeArrayDimsFunctor{}(coord, arrayDimsSize)
                        * sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>)
                    + flatOffsetOf<
                          typename Flattener::FlatRecordDim,
                          flatFieldIndex,
                          false> * LinearizeArrayDimsFunctor{}.size(arrayDimsSize);
                return {0, offset};
            }
        }

    private:
        using Flattener = FlattenRecordDimSingleBlob<TRecordDim>;
        ArrayDims arrayDimsSize;
    };

    /// Struct of array mapping storing the entire layout in a single blob.
    /// \see SoA
    template<typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using SingleBlobSoA = SoA<ArrayDims, RecordDim, false, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
    /// \see SoA
    template<typename ArrayDims, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MultiBlobSoA = SoA<ArrayDims, RecordDim, true, LinearizeArrayDimsFunctor>;

    template<bool SeparateBuffers = true, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct PreconfiguredSoA
    {
        template<typename ArrayDims, typename RecordDim>
        using type = SoA<ArrayDims, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>;
    };

    template<typename Mapping>
    inline constexpr bool isSoA = false;

    template<typename ArrayDims, typename RecordDim, bool SeparateBuffers, typename LinearizeArrayDimsFunctor>
    inline constexpr bool isSoA<SoA<ArrayDims, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>> = true;
} // namespace llama::mapping

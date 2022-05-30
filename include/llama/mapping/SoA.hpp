// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam SeparateBuffers If true, every element of the record dimension is mapped to its own buffer.
    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam FlattenRecordDimSingleBlob Defines how the record dimension's fields should be flattened if
    /// SeparateBuffers is false. See \ref FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref
    /// FlattenRecordDimDecreasingAlignment and \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        bool SeparateBuffers = true,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDimSingleBlob = FlattenRecordDimInOrder>
    struct SoA : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using Flattener = FlattenRecordDimSingleBlob<TRecordDim>;
        using size_type = typename TArrayExtents::value_type;

    public:
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        static constexpr std::size_t blobCount
            = SeparateBuffers ? boost::mp11::mp_size<FlatRecordDim<TRecordDim>>::value : 1;

        using Base::Base;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize([[maybe_unused]] size_type blobIndex) const -> size_type
        {
            if constexpr(SeparateBuffers)
            {
                constexpr auto typeSizes = []() constexpr
                {
                    Array<size_type, blobCount> r{};
                    forEachLeafCoord<TRecordDim>([&r, i = 0](auto rc) mutable constexpr {
                        r[i++] = sizeof(GetType<TRecordDim, decltype(rc)>);
                    });
                    return r;
                }
                ();
                return LinearizeArrayDimsFunctor{}.size(Base::extents()) * typeSizes[blobIndex];
            }
            else
            {
                return LinearizeArrayDimsFunctor{}.size(Base::extents()) * static_cast<size_type>(sizeOf<TRecordDim>);
            }
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            typename Base::ArrayIndex ad,
            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
        {
            if constexpr(SeparateBuffers)
            {
                constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
                const auto offset = LinearizeArrayDimsFunctor{}(ad, Base::extents())
                    * static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>));
                return {blob, offset};
            }
            else
            {
                constexpr std::size_t flatFieldIndex =
#ifdef __NVCC__
                    *& // mess with nvcc compiler state to workaround bug
#endif
                     Flattener::template flatIndex<RecordCoords...>;
                const auto offset = LinearizeArrayDimsFunctor{}(ad, Base::extents())
                        * static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>))
                    + static_cast<size_type>(flatOffsetOf<typename Flattener::FlatRecordDim, flatFieldIndex, false>)
                        * LinearizeArrayDimsFunctor{}.size(Base::extents());
                return {0, offset};
            }
        }
    };

    // we can drop this when inherited ctors also inherit deduction guides
    template<typename TArrayExtents, typename TRecordDim>
    SoA(TArrayExtents, TRecordDim) -> SoA<TArrayExtents, TRecordDim>;

    /// Struct of array mapping storing the entire layout in a single blob.
    /// \see SoA
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using SingleBlobSoA = SoA<ArrayExtents, RecordDim, false, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
    /// \see SoA
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MultiBlobSoA = SoA<ArrayExtents, RecordDim, true, LinearizeArrayDimsFunctor>;

    /// Binds parameters to an \ref SoA mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<bool SeparateBuffers = true, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct BindSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = SoA<ArrayExtents, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>;
    };

    template<typename Mapping>
    inline constexpr bool isSoA = false;

    template<typename ArrayExtents, typename RecordDim, bool SeparateBuffers, typename LinearizeArrayDimsFunctor>
    inline constexpr bool isSoA<SoA<ArrayExtents, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>> = true;
} // namespace llama::mapping

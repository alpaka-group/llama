// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam SeparateBuffers If true, every element of the record dimension is mapped to its own buffer.
    /// \tparam AlignSubArrays Only relevant when SeparateBuffers == false. If true, aligns the sub arrays created
    /// within the single blob by inserting padding.
    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam FlattenRecordDimSingleBlob Defines how the record dimension's fields should be flattened if
    /// SeparateBuffers is false. See \ref FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref
    /// FlattenRecordDimDecreasingAlignment and \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        bool SeparateBuffers = true,
        bool AlignSubArrays = false,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDimSingleBlob = FlattenRecordDimInOrder>
    struct SoA : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename TArrayExtents::value_type;

    public:
        inline static constexpr bool separateBuffers = SeparateBuffers;
        inline static constexpr bool alignSubArrays = AlignSubArrays;
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        using Flattener = FlattenRecordDimSingleBlob<TRecordDim>;
        inline static constexpr std::size_t blobCount
            = SeparateBuffers ? boost::mp11::mp_size<FlatRecordDim<TRecordDim>>::value : 1;

        using Base::Base;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize([[maybe_unused]] size_type blobIndex) const -> size_type
        {
            const auto flatSize = LinearizeArrayDimsFunctor{}.size(Base::extents());
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
                return flatSize * typeSizes[blobIndex];
            }
            else if constexpr(AlignSubArrays)
            {
                size_type size = 0;
                using namespace boost::mp11;
                using FRD = typename Flattener::FlatRecordDim;
                mp_for_each<mp_transform<mp_identity, FRD>>(
                    [&](auto ti)
                    {
                        using FieldType = typename decltype(ti)::type;
                        size = roundUpToMultiple(size, static_cast<size_type>(alignof(FieldType)));
                        size += static_cast<size_type>(sizeof(FieldType)) * flatSize;
                    });
                return size;
            }
            else
            {
                return flatSize * static_cast<size_type>(sizeOf<TRecordDim>);
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
                const auto subArrayOffset = LinearizeArrayDimsFunctor{}(ad, Base::extents())
                    * static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>));
                constexpr std::size_t flatFieldIndex =
#ifdef __NVCC__
                    *& // mess with nvcc compiler state to workaround bug
#endif
                     Flattener::template flatIndex<RecordCoords...>;
                const auto flatSize = LinearizeArrayDimsFunctor{}.size(Base::extents());
                using FRD = typename Flattener::FlatRecordDim;
                if constexpr(AlignSubArrays)
                {
                    // TODO(bgruber): we can take a shortcut here if we know that flatSize is a multiple of all type's
                    // alignment. We can also precompute a table of sub array starts (and maybe store it), or rely on
                    // the compiler pulling it out of loops.
                    using namespace boost::mp11;
                    size_type offset = 0;
                    mp_for_each<mp_transform<mp_identity, mp_take_c<FRD, flatFieldIndex>>>(
                        [&](auto ti)
                        {
                            using FieldType = typename decltype(ti)::type;
                            offset = roundUpToMultiple(offset, static_cast<size_type>(alignof(FieldType)));
                            offset += static_cast<size_type>(sizeof(FieldType)) * flatSize;
                        });
                    offset = roundUpToMultiple(offset, static_cast<size_type>(alignof(mp_at_c<FRD, flatFieldIndex>)));
                    offset += subArrayOffset;
                    return {0, offset};
                }
                else
                {
                    const auto offset
                        = subArrayOffset + static_cast<size_type>(flatOffsetOf<FRD, flatFieldIndex, false>) * flatSize;
                    return {0, offset};
                }
            }
        }
    };

    // we can drop this when inherited ctors also inherit deduction guides
    template<typename TArrayExtents, typename TRecordDim>
    SoA(TArrayExtents, TRecordDim) -> SoA<TArrayExtents, TRecordDim>;

    /// Struct of array mapping storing the entire layout in a single blob. The starts of the sub arrays are aligned by
    /// inserting padding. \see SoA
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using AlignedSingleBlobSoA = SoA<ArrayExtents, RecordDim, false, true, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing the entire layout in a single blob. The sub arrays are tightly packed,
    /// violating the type's alignment requirements. \see SoA
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using PackedSingleBlobSoA = SoA<ArrayExtents, RecordDim, false, false, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
    /// \see SoA
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MultiBlobSoA = SoA<ArrayExtents, RecordDim, true, false, LinearizeArrayDimsFunctor>;

    /// Binds parameters to an \ref SoA mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        bool SeparateBuffers = true,
        bool AlignSubArrays = false,
        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct BindSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = SoA<ArrayExtents, RecordDim, SeparateBuffers, AlignSubArrays, LinearizeArrayDimsFunctor>;
    };

    template<typename Mapping>
    inline constexpr bool isSoA = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        bool SeparateBuffers,
        bool AlignSubArrays,
        typename LinearizeArrayDimsFunctor>
    inline constexpr bool
        isSoA<SoA<ArrayExtents, RecordDim, SeparateBuffers, AlignSubArrays, LinearizeArrayDimsFunctor>> = true;
} // namespace llama::mapping

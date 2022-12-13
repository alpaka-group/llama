// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    enum class Blobs
    {
        Single,
        OnePerField
    };

    enum class SubArrayAlignment
    {
        Pack,
        Align
    };

    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam TBlobs If OnePerField, every element of the record dimension is mapped to its own blob.
    /// \tparam TSubArrayAlignment Only relevant when TBlobs == Single, ignored otherwise. If Align, aligns the sub
    /// arrays created within the single blob by inserting padding.
    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam FlattenRecordDimSingleBlob Defines how the record dimension's fields should be flattened if Blobs is
    /// Single. See \ref FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref
    /// FlattenRecordDimDecreasingAlignment and \ref FlattenRecordDimMinimizePadding.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        Blobs TBlobs = Blobs::OnePerField,
        SubArrayAlignment TSubArrayAlignment = SubArrayAlignment::Pack,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDimSingleBlob = FlattenRecordDimInOrder>
    struct SoA : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename TArrayExtents::value_type;

    public:
        inline static constexpr Blobs blobs = TBlobs;
        inline static constexpr SubArrayAlignment subArrayAlignment = TSubArrayAlignment;
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        using Flattener = FlattenRecordDimSingleBlob<TRecordDim>;
        inline static constexpr std::size_t blobCount
            = blobs == Blobs::OnePerField ? boost::mp11::mp_size<FlatRecordDim<TRecordDim>>::value : 1;

#ifndef __NVCC__
        using Base::Base;
#else
        constexpr SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr explicit SoA(TArrayExtents extents, TRecordDim = {}) : Base(extents)
        {
        }
#endif

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize([[maybe_unused]] size_type blobIndex) const -> size_type
        {
            const auto flatSize = LinearizeArrayDimsFunctor{}.size(Base::extents());
            if constexpr(blobs == Blobs::OnePerField)
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
            else if constexpr(subArrayAlignment == SubArrayAlignment::Align)
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
            if constexpr(blobs == Blobs::OnePerField)
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
                if constexpr(subArrayAlignment == SubArrayAlignment::Align)
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
    using AlignedSingleBlobSoA
        = SoA<ArrayExtents, RecordDim, Blobs::Single, SubArrayAlignment::Align, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing the entire layout in a single blob. The sub arrays are tightly packed,
    /// violating the type's alignment requirements. \see SoA
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using PackedSingleBlobSoA
        = SoA<ArrayExtents, RecordDim, Blobs::Single, SubArrayAlignment::Pack, LinearizeArrayDimsFunctor>;

    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
    /// \see SoA
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    using MultiBlobSoA
        = SoA<ArrayExtents, RecordDim, Blobs::OnePerField, SubArrayAlignment::Pack, LinearizeArrayDimsFunctor>;

    /// Binds parameters to an \ref SoA mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        Blobs Blobs = Blobs::OnePerField,
        SubArrayAlignment SubArrayAlignment = SubArrayAlignment::Pack,
        typename LinearizeArrayDimsFunctor = LinearizeArrayDimsCpp>
    struct BindSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayDimsFunctor>;
    };

    template<typename Mapping>
    inline constexpr bool isSoA = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        Blobs Blobs,
        SubArrayAlignment SubArrayAlignment,
        typename LinearizeArrayDimsFunctor>
    inline constexpr bool
        isSoA<SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayDimsFunctor>> = true;
} // namespace llama::mapping

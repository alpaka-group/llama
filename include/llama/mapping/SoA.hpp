// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    LLAMA_EXPORT
    enum class Blobs
    {
        Single,
        OnePerField
    };

    LLAMA_EXPORT
    enum class SubArrayAlignment
    {
        Pack,
        Align
    };

    /// Struct of array mapping. Used to create a \ref View via \ref allocView. We recommend to use multiple blobs when
    /// the array extents are dynamic and an aligned single blob version when they are static.
    /// \tparam TBlobs If OnePerField, every element of the record dimension is mapped to its own blob.
    /// \tparam TSubArrayAlignment Only relevant when TBlobs == Single, ignored otherwise. If Align, aligns the sub
    /// arrays created within the single blob by inserting padding. If the array extents are dynamic, this may add some
    /// overhead to the mapping logic.
    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam PermuteFieldsSingleBlob Defines how the record dimension's fields should be permuted if Blobs is
    /// Single. See \ref PermuteFieldsInOrder, \ref PermuteFieldsIncreasingAlignment, \ref
    /// PermuteFieldsDecreasingAlignment and \ref PermuteFieldsMinimizePadding.
    LLAMA_EXPORT
    template<
        typename TArrayExtents,
        typename TRecordDim,
        Blobs TBlobs = Blobs::OnePerField,
        SubArrayAlignment TSubArrayAlignment
        = TBlobs == Blobs::Single ? SubArrayAlignment::Align : SubArrayAlignment::Pack,
        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
        template<typename> typename PermuteFieldsSingleBlob = PermuteFieldsInOrder>
    struct SoA : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename TArrayExtents::value_type;

    public:
        inline static constexpr Blobs blobs = TBlobs;
        inline static constexpr SubArrayAlignment subArrayAlignment = TSubArrayAlignment;
        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
        using Permuter = PermuteFieldsSingleBlob<FlatRecordDim<TRecordDim>>;
        inline static constexpr std::size_t blobCount
            = blobs == Blobs::OnePerField ? mp_size<FlatRecordDim<TRecordDim>>::value : 1;

#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ >= 12
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
            const auto flatSize = LinearizeArrayIndexFunctor{}.size(Base::extents());
            if constexpr(blobs == Blobs::OnePerField)
            {
                constexpr auto typeSizes = []() constexpr
                {
                    Array<size_type, blobCount> r{};
                    forEachLeafCoord<TRecordDim>([&r, i = 0](auto rc) mutable constexpr
                                                 { r[i++] = sizeof(GetType<TRecordDim, decltype(rc)>); });
                    return r;
                }();
                return flatSize * typeSizes[blobIndex];
            }
            else if constexpr(subArrayAlignment == SubArrayAlignment::Align)
            {
                size_type size = 0;
                using FRD = typename Permuter::FlatRecordDim;
                mp_for_each<mp_transform<mp_identity, FRD>>(
                    [&](auto ti) LLAMA_LAMBDA_INLINE
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

    private:
        static LLAMA_CONSTEVAL auto computeSubArrayOffsets()
        {
            using FRD = typename Permuter::FlatRecordDim;
            constexpr auto staticFlatSize = LinearizeArrayIndexFunctor{}.size(TArrayExtents{});
            constexpr auto subArrays = mp_size<FRD>::value;
            Array<size_type, subArrays> r{};
            // r[0] == 0, only compute the following offsets
            mp_for_each<mp_iota_c<subArrays - 1>>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    r[i + 1] = r[i];
                    using ThisFieldType = mp_at_c<FRD, i>;
                    r[i + 1] += static_cast<size_type>(sizeof(ThisFieldType)) * staticFlatSize;
                    using NextFieldType = mp_at_c<FRD, i + 1>;
                    r[i + 1] = roundUpToMultiple(r[i + 1], static_cast<size_type>(alignof(NextFieldType)));
                });
            return r;
        }

    public:
        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            typename Base::ArrayIndex ai,
            RecordCoord<RecordCoords...> rc = {}) const -> NrAndOffset<size_type>
        {
            return blobNrAndOffset(LinearizeArrayIndexFunctor{}(ai, Base::extents()), rc);
        }

        // Exposed for aosoaCommonBlockCopy. Should be private ...
        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(
            size_type flatArrayIndex,
            RecordCoord<RecordCoords...> = {}) const -> NrAndOffset<size_type>
        {
            const size_type elementOffset
                = flatArrayIndex * static_cast<size_type>(sizeof(GetType<TRecordDim, RecordCoord<RecordCoords...>>));
            if constexpr(blobs == Blobs::OnePerField)
            {
                constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
                return {blob, elementOffset};
            }
            else
            {
                constexpr std::size_t flatFieldIndex =
#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 6
                    *& // mess with nvcc compiler state to workaround bug
#endif
                     Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>;
                const size_type flatSize = LinearizeArrayIndexFunctor{}.size(Base::extents());
                using FRD = typename Permuter::FlatRecordDim;
                if constexpr(subArrayAlignment == SubArrayAlignment::Align)
                {
                    if constexpr(TArrayExtents::rankStatic == TArrayExtents::rank)
                    {
                        // full array extents are known statically, we can precompute the sub array offsets
                        constexpr auto subArrayOffsets = computeSubArrayOffsets();
                        return {0, subArrayOffsets[flatFieldIndex] + elementOffset};
                    }
                    else
                    {
                        // TODO(bgruber): we can take a shortcut here if we know that flatSize is a multiple of all
                        // type's alignment. We can also precompute a table of sub array starts (and maybe store it),
                        // or rely on the compiler it out of loops.
                        size_type offset = 0;
                        mp_for_each<mp_iota_c<flatFieldIndex>>(
                            [&](auto ic) LLAMA_LAMBDA_INLINE
                            {
                                constexpr auto i = decltype(ic)::value;
                                using ThisFieldType = mp_at_c<FRD, i>;
                                offset += static_cast<size_type>(sizeof(ThisFieldType)) * flatSize;
                                using NextFieldType = mp_at_c<FRD, i + 1>;
                                offset = roundUpToMultiple(offset, static_cast<size_type>(alignof(NextFieldType)));
                            });
                        offset += elementOffset;
                        return {0, offset};
                    }
                }
                else
                {
                    const auto offset
                        = elementOffset + static_cast<size_type>(flatOffsetOf<FRD, flatFieldIndex, false>) * flatSize;
                    return {0, offset};
                }
            }
        }
    };

    // we can drop this when inherited ctors also inherit deduction guides
    LLAMA_EXPORT
    template<typename TArrayExtents, typename TRecordDim>
    SoA(TArrayExtents, TRecordDim) -> SoA<TArrayExtents, TRecordDim>;

    /// Struct of array mapping storing the entire layout in a single blob. The starts of the sub arrays are aligned by
    /// inserting padding. \see SoA
    LLAMA_EXPORT
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    using AlignedSingleBlobSoA
        = SoA<ArrayExtents, RecordDim, Blobs::Single, SubArrayAlignment::Align, LinearizeArrayIndexFunctor>;

    /// Struct of array mapping storing the entire layout in a single blob. The sub arrays are tightly packed,
    /// violating the type's alignment requirements. \see SoA
    LLAMA_EXPORT
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    using PackedSingleBlobSoA
        = SoA<ArrayExtents, RecordDim, Blobs::Single, SubArrayAlignment::Pack, LinearizeArrayIndexFunctor>;

    /// Struct of array mapping storing each attribute of the record dimension in a separate blob.
    /// \see SoA
    LLAMA_EXPORT
    template<typename ArrayExtents, typename RecordDim, typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    using MultiBlobSoA
        = SoA<ArrayExtents, RecordDim, Blobs::OnePerField, SubArrayAlignment::Pack, LinearizeArrayIndexFunctor>;

    /// Binds parameters to an \ref SoA mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    LLAMA_EXPORT
    template<
        Blobs Blobs = Blobs::OnePerField,
        SubArrayAlignment SubArrayAlignment = SubArrayAlignment::Pack,
        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight>
    struct BindSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayIndexFunctor>;
    };

    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool isSoA = false;

    LLAMA_EXPORT
    template<
        typename ArrayExtents,
        typename RecordDim,
        Blobs Blobs,
        SubArrayAlignment SubArrayAlignment,
        typename LinearizeArrayIndexFunctor>
    inline constexpr bool isSoA<SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayIndexFunctor>>
        = true;
} // namespace llama::mapping

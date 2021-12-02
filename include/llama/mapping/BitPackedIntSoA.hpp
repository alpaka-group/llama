// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "BitPackedIntRef.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename A, typename B>
        using HasLargerSize = boost::mp11::mp_bool<sizeof(A) < sizeof(B)>;

        template<typename RecordDim>
        using LargestIntegral = boost::mp11::mp_max_element<FlatRecordDim<RecordDim>, HasLargerSize>;

        template<typename T>
        struct MakeUnsigned : std::make_unsigned<T>
        {
        };

        template<>
        struct MakeUnsigned<bool>
        {
            using type = std::uint8_t;
        };
    } // namespace internal

    /// Struct of array mapping using bit packing to reduce size/precision of integral data types. If your record
    /// dimension contains non-integral types, split them off using the \ref Split mapping first.
    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and how
    /// big the linear domain gets.
    /// \tparam StoredIntegral Integral type used as storage of reduced precision integers.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename LinearizeArrayDimsFunctor = mapping::LinearizeArrayDimsCpp,
        typename StoredIntegral = typename internal::MakeUnsigned<internal::LargestIntegral<TRecordDim>>::type>
    struct BitPackedIntSoA : TArrayExtents
    {
        static_assert(
            boost::mp11::mp_all_of<FlatRecordDim<TRecordDim>, std::is_integral>::value,
            "All record dimension field types must be integral");

        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<FlatRecordDim<RecordDim>>::value;

        constexpr BitPackedIntSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr BitPackedIntSoA(unsigned bits, ArrayExtents extents, RecordDim = {})
            : ArrayExtents(extents)
            , bits{bits}
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return *this; // NOLINT(cppcoreguidelines-slicing)
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(std::size_t /*blobIndex*/) const -> std::size_t
        {
            constexpr auto bitsPerStoredIntegral = sizeof(StoredIntegral) * CHAR_BIT;
            const auto bitsNeeded = LinearizeArrayDimsFunctor{}.size(extents()) * bits;
            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(ArrayIndex ai, RecordCoord<RecordCoords...>, Blobs& blobs)
            const
        {
            constexpr auto blob = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents()) * bits;

            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
            using DstType = GetType<RecordDim, RecordCoord<RecordCoords...>>;
            return llama::internal::BitPackedIntRef<DstType, QualifiedStoredIntegral*>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                bits};
        }

    private:
        unsigned bits = 0;
    };
} // namespace llama::mapping

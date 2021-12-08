// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "BitPackedFloatRef.hpp"
#include "Common.hpp"

#include <cstdint>
#include <limits>

namespace llama::mapping
{
    /// Struct of array mapping using bit packing to reduce size/precision of floating-point data types. The bit layout
    /// is [1 sign bit, exponentBits bits from the exponent, mantissaBits bits from the mantissa]+ and tries to follow
    /// IEEE 754. Infinity and NAN are supported. If the packed exponent bits are not big enough to hold a number, it
    /// will be set to infinity (preserving the sign). If your record dimension contains non-floating-point types,
    /// split them off using the \ref Split mapping first.
    /// \tparam LinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and how
    /// big the linear domain gets.
    /// \tparam StoredIntegral Integral type used as storage of reduced precision floating-point values.
    // TODO(bgruber): we could also split each float in the record dimension into 3 integers, sign bit, exponent and
    // mantissa. might not be efficient though
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename LinearizeArrayDimsFunctor = llama::mapping::LinearizeArrayDimsCpp,
        typename StoredIntegral = std::conditional_t<
            boost::mp11::mp_contains<llama::FlatRecordDim<TRecordDim>, double>::value,
            std::uint64_t,
            std::uint32_t>>
    struct BitPackedFloatSoA : TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        constexpr BitPackedFloatSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr BitPackedFloatSoA(unsigned exponentBits, unsigned mantissaBits, ArrayExtents extents, RecordDim = {})
            : ArrayExtents(extents)
            , exponentBits{exponentBits}
            , mantissaBits{mantissaBits}
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
            const auto bitsNeeded = LinearizeArrayDimsFunctor{}.size(extents()) * (exponentBits + mantissaBits + 1);
            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            ArrayIndex ai,
            llama::RecordCoord<RecordCoords...>,
            Blobs& blobs) const
        {
            constexpr auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;
            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents()) * (exponentBits + mantissaBits + 1);

            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
            using DstType = llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>;
            return llama::internal::BitPackedFloatRef<DstType, QualifiedStoredIntegral*>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                exponentBits,
                mantissaBits
#ifndef NDEBUG
                ,
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0] + blobSize(blob))
#endif
            };
        }

    private:
        unsigned exponentBits = 0;
        unsigned mantissaBits = 0;
    };
} // namespace llama::mapping

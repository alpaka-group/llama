// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "BitPackedIntSoA.hpp"

#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace llama::mapping
{
    namespace internal
    {
        template<typename T>
        struct FloatBitTraits;

        template<>
        struct FloatBitTraits<float>
        {
            static inline constexpr unsigned mantissa = 23;
            static inline constexpr unsigned exponent = 8;
        };

        template<>
        struct FloatBitTraits<double>
        {
            static inline constexpr unsigned mantissa = 52;
            static inline constexpr unsigned exponent = 11;
        };

        template<typename Integral>
        auto repackFloat(
            Integral inFloat,
            unsigned inMantissaBits,
            unsigned inExponentBits,
            unsigned outMantissaBits,
            unsigned outExponentBits) -> Integral
        {
            const Integral inMantissaMask = (Integral{1} << inMantissaBits) - 1u;
            const Integral inExponentMask = (Integral{1} << inExponentBits) - 1u;

            Integral inMantissa = inFloat & inMantissaMask;
            const Integral inExponent = (inFloat >> inMantissaBits) & inExponentMask;
            const Integral inSign = inFloat >> inExponentBits >> inMantissaBits;

            const Integral outExponentMask = (Integral{1} << outExponentBits) - 1u;
            Integral outExponent;
            if(inExponent == inExponentMask) [[unlikely]]
                outExponent = outExponentMask; // propagate +/- inf/nan
            else if(inExponent == 0) [[unlikely]]
                outExponent = 0; // propagate -/+ zero
            else
            {
                const int outExponentMax = 1 << (outExponentBits - 1); // NOLINT(hicpp-signed-bitwise)
                const int outExponentMin = -outExponentMax + 1;
                const int outExponentBias = outExponentMax - 1;
                const int inExponentBias = (1 << (inExponentBits - 1)) - 1; // NOLINT(hicpp-signed-bitwise)

                const int exponent = static_cast<int>(inExponent) - inExponentBias;
                const auto clampedExponent = std::clamp(exponent, outExponentMin, outExponentMax);
                if(clampedExponent == outExponentMin || clampedExponent == outExponentMax)
                    inMantissa = 0; // when exponent changed, let value become inf and not nan
                outExponent = clampedExponent + outExponentBias;
            }
            assert(outExponent < (1u << outExponentBits));

            const Integral packedMantissa = inMantissaBits > outMantissaBits
                ? inMantissa >> (inMantissaBits - outMantissaBits)
                : inMantissa << (outMantissaBits - inMantissaBits);
            const Integral packedExponent = outExponent << outMantissaBits;
            const Integral packedSign = inSign << outExponentBits << outMantissaBits;

            const auto outFloat = static_cast<Integral>(packedMantissa | packedExponent | packedSign);
            return outFloat;
        }

        /// A proxy type representing a reference to a reduced precision floating-point value, stored in a buffer at a
        /// specified bit offset.
        /// @tparam Integral Integral data type which can be loaded and store through this reference.
        /// @tparam StoredIntegralPointer Pointer to integral type used for storing the bits.
        template<typename Float, typename StoredIntegralPointer>
        struct BitPackedFloatRef : ProxyRefOpMixin<BitPackedFloatRef<Float, StoredIntegralPointer>, Float>
        {
        private:
            static_assert(
                std::is_same_v<Float, float> || std::is_same_v<Float, double>,
                "Types other than float or double are not implemented yet");
            static_assert(
                std::numeric_limits<Float>::is_iec559,
                "Only IEEE754/IEC559 floating point formats are implemented");

            using FloatBits = std::conditional_t<std::is_same_v<Float, float>, std::uint32_t, std::uint64_t>;

            internal::BitPackedIntRef<FloatBits, StoredIntegralPointer> intref;
            unsigned exponentBits = 0;
            unsigned mantissaBits = 0;

        public:
            using value_type = Float;

            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedFloatRef(
            StoredIntegralPointer p,
            std::size_t bitOffset,
            unsigned exponentBits,
            unsigned mantissaBits
#ifndef NDEBUG
            ,
            StoredIntegralPointer endPtr
#endif
            )
            : intref{p, bitOffset, exponentBits + mantissaBits + 1,
#ifndef NDEBUG
            endPtr
#endif
        }
            , exponentBits(exponentBits)
            , mantissaBits(mantissaBits)
            {
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator Float() const
            {
                using Bits = internal::FloatBitTraits<Float>;
                const FloatBits packedFloat = intref;
                const FloatBits unpackedFloat
                    = repackFloat(packedFloat, mantissaBits, exponentBits, Bits::mantissa, Bits::exponent);
                Float f;
                std::memcpy(&f, &unpackedFloat, sizeof(Float));
                return f;
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Float f) -> BitPackedFloatRef&
            {
                using Bits = internal::FloatBitTraits<Float>;
                FloatBits unpackedFloat = 0;
                std::memcpy(&unpackedFloat, &f, sizeof(Float));
                const FloatBits packedFloat
                    = repackFloat(unpackedFloat, Bits::mantissa, Bits::exponent, mantissaBits, exponentBits);
                intref = packedFloat;
                return *this;
            }
        };
    } // namespace internal

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
            return internal::BitPackedFloatRef<DstType, QualifiedStoredIntegral*>{
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

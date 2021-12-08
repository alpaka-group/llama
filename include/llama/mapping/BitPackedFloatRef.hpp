#pragma once

#include "BitPackedIntRef.hpp"

#include <climits>
#include <cstring>
#include <type_traits>

namespace llama::internal
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
    struct BitPackedFloatRef
    {
        static_assert(
            std::is_same_v<Float, float> || std::is_same_v<Float, double>,
            "Types other than float or double are not implemented yet");
        static_assert(
            std::numeric_limits<Float>::is_iec559,
            "Only IEEE754/IEC559 floating point formats are implemented");

        using FloatBits = std::conditional_t<std::is_same_v<Float, float>, std::uint32_t, std::uint64_t>;

    private:
        llama::internal::BitPackedIntRef<FloatBits, StoredIntegralPointer> intref;
        unsigned exponentBits = 0;
        unsigned mantissaBits = 0;

    public:
        BitPackedFloatRef(
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
        operator Float() const
        {
            using Bits = internal::FloatBitTraits<Float>;
            const FloatBits packedFloat = intref;
            const FloatBits unpackedFloat
                = repackFloat(packedFloat, mantissaBits, exponentBits, Bits::mantissa, Bits::exponent);
            Float f;
            std::memcpy(&f, &unpackedFloat, sizeof(Float));
            return f;
        }

        auto operator=(Float f) -> BitPackedFloatRef&
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
} // namespace llama::internal

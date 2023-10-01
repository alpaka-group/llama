// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "BitPackedInt.hpp"
#include "Common.hpp"

#include <algorithm>
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
            inline static constexpr unsigned mantissa = 23;
            inline static constexpr unsigned exponent = 8;
        };

        template<>
        struct FloatBitTraits<double>
        {
            inline static constexpr unsigned mantissa = 52;
            inline static constexpr unsigned exponent = 11;
        };

        template<typename Integral>
        LLAMA_FN_HOST_ACC_INLINE auto repackFloat(
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
            if(inExponent == inExponentMask) [[LLAMA_UNLIKELY]]
                outExponent = outExponentMask; // propagate +/- inf/nan
            else if(inExponent == 0) [[LLAMA_UNLIKELY]]
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

        // TODO(bgruber): Boost.Hana generalizes these sorts of computations on mixed constants and values
        template<typename E, typename M>
        LLAMA_FN_HOST_ACC_INLINE auto integBits(E e, M m)
        {
            return llama::internal::BoxedValue{e.value() + m.value() + 1};
        }

        template<auto E, auto M>
        LLAMA_FN_HOST_ACC_INLINE auto integBits(
            llama::internal::BoxedValue<Constant<E>>,
            llama::internal::BoxedValue<Constant<M>>)
        {
            return llama::internal::BoxedValue<Constant<E + M + 1>>{};
        }

        /// A proxy type representing a reference to a reduced precision floating-point value, stored in a buffer at a
        /// specified bit offset.
        /// @tparam Float Floating-point data type which can be loaded and store through this reference.
        /// @tparam StoredIntegralCV Integral type used for storing the bits with CV qualifiers.
        /// @tparam SizeType Type used to store sizes and offsets.
        template<typename Float, typename StoredIntegralCV, typename VHExp, typename VHMan, typename SizeType>
        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
        struct LLAMA_DECLSPEC_EMPTY_BASES BitPackedFloatRef
            : private VHExp
            , private VHMan
            , ProxyRefOpMixin<BitPackedFloatRef<Float, StoredIntegralCV, VHExp, VHMan, SizeType>, Float>
        {
        private:
            static_assert(
                std::is_same_v<Float, float> || std::is_same_v<Float, double>,
                "Types other than float or double are not implemented yet");
            static_assert(
                std::numeric_limits<Float>::is_iec559,
                "Only IEEE754/IEC559 floating point formats are implemented");

            using FloatBits = std::conditional_t<std::is_same_v<Float, float>, std::uint32_t, std::uint64_t>;

            BitPackedIntRef<
                FloatBits,
                StoredIntegralCV,
                decltype(integBits(std::declval<VHExp>(), std::declval<VHMan>())),
                SizeType,
                SignBit::Discard>
                intref;

        public:
            using value_type = Float;

            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedFloatRef(
                StoredIntegralCV* p,
                SizeType bitOffset,
                VHExp vhExp,
                VHMan vhMan)
                : VHExp{vhExp}
                , VHMan{vhMan}
                , intref{
                      p,
                      bitOffset,
                      integBits(vhExp, vhMan),
                  }
            {
            }

            BitPackedFloatRef(const BitPackedFloatRef&) = default;

            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const BitPackedFloatRef& other) -> BitPackedFloatRef&
            {
                *this = static_cast<value_type>(other);
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator Float() const
            {
                using Bits = FloatBitTraits<Float>;
                const FloatBits packedFloat = intref;
                const FloatBits unpackedFloat
                    = repackFloat(packedFloat, VHMan::value(), VHExp::value(), Bits::mantissa, Bits::exponent);
                Float f;
                std::memcpy(&f, &unpackedFloat, sizeof(Float));
                return f;
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Float f) -> BitPackedFloatRef&
            {
                using Bits = FloatBitTraits<Float>;
                FloatBits unpackedFloat = 0;
                std::memcpy(&unpackedFloat, &f, sizeof(Float));
                const FloatBits packedFloat
                    = repackFloat(unpackedFloat, Bits::mantissa, Bits::exponent, VHMan::value(), VHExp::value());
                intref = packedFloat;
                return *this;
            }
        };

        template<typename RecordDim>
        using StoredIntegralFor
            = std::conditional_t<mp_contains<FlatRecordDim<RecordDim>, double>::value, std::uint64_t, std::uint32_t>;
    } // namespace internal

    // TODO(bgruber): I would like to allow zero mantissa bits, which would then no longer support INF. Likewise,
    // support to skip the sign bit would also be great.
    /// Struct of array mapping using bit packing to reduce size/precision of floating-point data types. The bit layout
    /// is [1 sign bit, exponentBits bits from the exponent, mantissaBits bits from the mantissa]+ and tries to follow
    /// IEEE 754. Infinity and NAN are supported. If the packed exponent bits are not big enough to hold a number, it
    /// will be set to infinity (preserving the sign). If your record dimension contains non-floating-point types,
    /// split them off using the \ref Split mapping first.
    /// \tparam ExponentBits If ExponentBits is llama::Constant<N>, the compile-time N specifies the number of bits to
    /// use to store the exponent. If ExponentBits is llama::Value<T>, the number of bits is specified at runtime,
    /// passed to the constructor and stored as type T. Must not be zero.
    /// \tparam MantissaBits Like ExponentBits but for the mantissa bits. Must not be zero (otherwise values turn INF).
    /// \tparam TLinearizeArrayIndexFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam TStoredIntegral Integral type used as storage of reduced precision floating-point values.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename ExponentBits = typename TArrayExtents::value_type,
        typename MantissaBits = ExponentBits,
        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
        typename TStoredIntegral = internal::StoredIntegralFor<TRecordDim>>
    struct LLAMA_DECLSPEC_EMPTY_BASES BitPackedFloatSoA
        : MappingBase<TArrayExtents, TRecordDim>
        , llama::internal::BoxedValue<ExponentBits, 0>
        , llama::internal::BoxedValue<MantissaBits, 1>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using VHExp = llama::internal::BoxedValue<ExponentBits, 0>;
        using VHMan = llama::internal::BoxedValue<MantissaBits, 1>;
        using size_type = typename TArrayExtents::value_type;

    public:
        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
        using StoredIntegral = TStoredIntegral;
        static constexpr std::size_t blobCount = mp_size<FlatRecordDim<TRecordDim>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto exponentBits() const -> size_type
        {
            return static_cast<size_type>(VHExp::value());
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto mantissaBits() const -> size_type
        {
            return static_cast<size_type>(VHMan::value());
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit BitPackedFloatSoA(
            TArrayExtents extents = {},
            ExponentBits exponentBits = {},
            MantissaBits mantissaBits = {},
            TRecordDim = {})
            : Base(extents)
            , VHExp{exponentBits}
            , VHMan{mantissaBits}
        {
            assert(this->exponentBits() > 0);
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
        {
            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(StoredIntegral) * CHAR_BIT);
            const auto bitsNeeded
                = LinearizeArrayIndexFunctor{}.size(Base::extents()) * (exponentBits() + mantissaBits() + 1);
            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Base::ArrayIndex ai,
            RecordCoord<RecordCoords...>,
            Blobs& blobs) const
        {
            constexpr auto blob = llama::flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
            const auto bitOffset
                = LinearizeArrayIndexFunctor{}(ai, Base::extents()) * (exponentBits() + mantissaBits() + 1);

            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return internal::BitPackedFloatRef<DstType, QualifiedStoredIntegral, VHExp, VHMan, size_type>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                static_cast<const VHExp&>(*this),
                static_cast<const VHMan&>(*this)};
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
    };

    /// Binds parameters to a \ref BitPackedFloatSoA mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        typename ExponentBits = unsigned,
        typename MantissaBits = ExponentBits,
        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
        typename StoredIntegral = void>
    struct BindBitPackedFloatSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = BitPackedFloatSoA<
            ArrayExtents,
            RecordDim,
            ExponentBits,
            MantissaBits,
            LinearizeArrayIndexFunctor,
            std::conditional_t<
                !std::is_void_v<StoredIntegral>,
                StoredIntegral,
                internal::StoredIntegralFor<RecordDim>>>;
    };

    template<typename Mapping>
    inline constexpr bool isBitPackedFloatSoA = false;

    template<typename... Ts>
    inline constexpr bool isBitPackedFloatSoA<BitPackedFloatSoA<Ts...>> = true;

    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename ExponentBits = typename TArrayExtents::value_type,
        typename MantissaBits = ExponentBits,
        typename TLinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
        template<typename> typename PermuteFields = PermuteFieldsInOrder,
        typename TStoredIntegral = internal::StoredIntegralFor<TRecordDim>>
    struct LLAMA_DECLSPEC_EMPTY_BASES BitPackedFloatAoS
        : MappingBase<TArrayExtents, TRecordDim>
        , llama::internal::BoxedValue<ExponentBits, 0>
        , llama::internal::BoxedValue<MantissaBits, 1>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using VHExp = llama::internal::BoxedValue<ExponentBits, 0>;
        using VHMan = llama::internal::BoxedValue<MantissaBits, 1>;
        using size_type = typename TArrayExtents::value_type;

    public:
        using LinearizeArrayIndexFunctor = TLinearizeArrayIndexFunctor;
        using StoredIntegral = TStoredIntegral;

        using Permuter = PermuteFields<FlatRecordDim<TRecordDim>>;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto exponentBits() const -> size_type
        {
            return static_cast<size_type>(VHExp::value());
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto mantissaBits() const -> size_type
        {
            return static_cast<size_type>(VHMan::value());
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit BitPackedFloatAoS(
            TArrayExtents extents = {},
            ExponentBits exponentBits = {},
            MantissaBits mantissaBits = {},
            TRecordDim = {})
            : Base(extents)
            , VHExp{exponentBits}
            , VHMan{mantissaBits}
        {
            assert(this->exponentBits() > 0);
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
        {
            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(StoredIntegral) * CHAR_BIT);
            const auto bitsNeeded = TLinearizeArrayIndexFunctor{}.size(Base::extents())
                * static_cast<size_type>(exponentBits() + mantissaBits() + 1)
                * static_cast<size_type>(flatFieldCount<TRecordDim>);
            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Base::ArrayIndex ai,
            RecordCoord<RecordCoords...>,
            Blobs& blobs) const
        {
            constexpr auto flatFieldIndex = static_cast<size_type>(
                Permuter::template permute<flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>>);
            const auto bitOffset = ((TLinearizeArrayIndexFunctor{}(ai, Base::extents())
                                     * static_cast<size_type>(flatFieldCount<TRecordDim>))
                                    + flatFieldIndex)
                * static_cast<size_type>(exponentBits() + mantissaBits() + 1);

            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return internal::BitPackedFloatRef<DstType, QualifiedStoredIntegral, VHExp, VHMan, size_type>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[0][0]),
                bitOffset,
                static_cast<const VHExp&>(*this),
                static_cast<const VHMan&>(*this)};
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
    };

    template<
        typename ExponentBits = unsigned,
        typename MantissaBits = ExponentBits,
        typename LinearizeArrayIndexFunctor = LinearizeArrayIndexRight,
        template<typename> typename PermuteFields = PermuteFieldsInOrder,
        typename StoredIntegral = void>
    struct BindBitPackedFloatAoS
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = BitPackedFloatAoS<
            ArrayExtents,
            RecordDim,
            ExponentBits,
            MantissaBits,
            LinearizeArrayIndexFunctor,
            PermuteFields,
            std::conditional_t<
                !std::is_void_v<StoredIntegral>,
                StoredIntegral,
                internal::StoredIntegralFor<RecordDim>>>;
    };

    template<typename Mapping>
    inline constexpr bool isBitPackedFloatAoS = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        typename ExponentBits,
        typename MantissaBits,
        typename LinearizeArrayIndexFunctor,
        template<typename>
        typename PermuteFields,
        typename StoredIntegral>
    inline constexpr bool isBitPackedFloatAoS<BitPackedFloatAoS<
        ArrayExtents,
        RecordDim,
        ExponentBits,
        MantissaBits,
        LinearizeArrayIndexFunctor,
        PermuteFields,
        StoredIntegral>>
        = true;
} // namespace llama::mapping

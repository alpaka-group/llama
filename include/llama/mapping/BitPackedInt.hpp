// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "../Core.hpp"
#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"

#include <climits>
#include <type_traits>

namespace llama::mapping
{
    enum class SignBit
    {
        Keep,
        Discard
    };

    namespace internal
    {
        template<typename Integral>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto makeMask(Integral bits) -> Integral
        {
            return bits >= sizeof(Integral) * CHAR_BIT ? ~Integral{0} : (Integral{1} << bits) - 1u;
        }

        template<bool KeepSignBit, typename Integral, typename StoredIntegral>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto bitunpack(
            const StoredIntegral* ptr,
            StoredIntegral bitOffset,
            StoredIntegral bitCount) -> Integral
        {
            constexpr auto bitsPerIntegral = static_cast<StoredIntegral>(sizeof(Integral) * CHAR_BIT);
            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
            static_assert(bitsPerIntegral <= bitsPerStoredIntegral);
            assert(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
#ifdef __clang__
            // this is necessary to silence the clang static analyzer
            __builtin_assume(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
#endif

            const auto* p = ptr + bitOffset / bitsPerStoredIntegral;
            const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;
            //            assert(p < endPtr);
            auto v = p[0] >> innerBitOffset;

            const auto innerBitEndOffset = innerBitOffset + bitCount;
            if(innerBitEndOffset <= bitsPerStoredIntegral)
            {
                const auto mask = makeMask(bitCount);
                v &= mask;
            }
            else
            {
                const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
                const auto bitsLoaded = bitsPerStoredIntegral - innerBitOffset;
                const auto mask = makeMask(excessBits);
                //                assert(p + 1 < endPtr);
                v |= (p[1] & mask) << bitsLoaded;
            }
            if constexpr(std::is_signed_v<Integral> && KeepSignBit)
            {
                // perform sign extension
                if((v & (StoredIntegral{1} << (bitCount - 1))) && bitCount < bitsPerStoredIntegral)
                    v |= ~StoredIntegral{0} << bitCount;
            }
            return static_cast<Integral>(v);
        }

        template<bool KeepSignBit, typename StoredIntegral, typename Integral>
        LLAMA_FN_HOST_ACC_INLINE constexpr void bitpack(
            StoredIntegral* ptr,
            StoredIntegral bitOffset,
            StoredIntegral bitCount,
            Integral value)
        {
            constexpr auto bitsPerIntegral = static_cast<StoredIntegral>(sizeof(Integral) * CHAR_BIT);
            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
            static_assert(bitsPerIntegral <= bitsPerStoredIntegral);
            assert(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
#ifdef __clang__
            // this is necessary to silence the clang static analyzer
            __builtin_assume(bitCount > 0 && bitCount <= bitsPerStoredIntegral);
#endif

            // NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
            const auto unsignedValue = static_cast<StoredIntegral>(value);
            const auto mask = makeMask(bitCount);
            StoredIntegral valueBits;
            if constexpr(std::is_signed_v<Integral> && KeepSignBit)
            {
                const auto magnitudeMask = makeMask(bitCount - 1);
                const auto isSigned = value < 0;
                valueBits = (StoredIntegral{isSigned} << (bitCount - 1)) | (unsignedValue & magnitudeMask);
            }
            else
            {
                valueBits = unsignedValue & mask;
            }

            auto* p = ptr + bitOffset / bitsPerStoredIntegral;
            const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;

            {
                const auto clearMask = ~(mask << innerBitOffset);
                //                assert(p < endPtr);
                auto mem = p[0] & clearMask; // clear previous bits
                mem |= valueBits << innerBitOffset; // write new bits
                p[0] = mem;
            }

            const auto innerBitEndOffset = innerBitOffset + bitCount;
            if(innerBitEndOffset > bitsPerStoredIntegral)
            {
                const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
                const auto bitsWritten = bitsPerStoredIntegral - innerBitOffset;
                const auto clearMask = ~makeMask(excessBits);
                //                assert(p + 1 < endPtr);
                auto mem = p[1] & clearMask; // clear previous bits
                mem |= valueBits >> bitsWritten; // write new bits
                p[1] = mem;
            }
        }

        template<typename Integral, typename StoredIntegral>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto bitunpack1(const StoredIntegral* ptr, StoredIntegral bitOffset)
            -> Integral
        {
            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
            const auto bit
                = (ptr[bitOffset / bitsPerStoredIntegral] >> (bitOffset % bitsPerStoredIntegral)) & StoredIntegral{1};
            return static_cast<Integral>(bit);
        }

        template<typename StoredIntegral, typename Integral>
        LLAMA_FN_HOST_ACC_INLINE constexpr void bitpack1(StoredIntegral* ptr, StoredIntegral bitOffset, Integral value)
        {
            constexpr auto bitsPerStoredIntegral = static_cast<StoredIntegral>(sizeof(StoredIntegral) * CHAR_BIT);
            const auto bitOff = bitOffset % bitsPerStoredIntegral;
            auto& dst = ptr[bitOffset / bitsPerStoredIntegral];
            dst &= ~(StoredIntegral{1} << bitOff); // clear bit
            const auto bit = (static_cast<StoredIntegral>(value) & StoredIntegral{1});
            dst |= (bit << bitOff); // set bit
        }

        /// A proxy type representing a reference to a reduced precision integral value, stored in a buffer at a
        /// specified bit offset.
        /// @tparam Integral Integral data type which can be loaded and store through this reference.
        /// @tparam StoredIntegralCV Integral type used for storing the bits with CV qualifiers.
        /// @tparam SizeType Type used to store sizes and offsets.
        template<typename Integral, typename StoredIntegralCV, typename VHBits, typename SizeType, SignBit SignBit>
        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
        struct BitPackedIntRef
            : private VHBits
            , ProxyRefOpMixin<BitPackedIntRef<Integral, StoredIntegralCV, VHBits, SizeType, SignBit>, Integral>
        {
        private:
            using StoredIntegral = std::remove_cv_t<StoredIntegralCV>;
            StoredIntegralCV* ptr;
            SizeType bitOffset;

        public:
            using value_type = Integral;

            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedIntRef(
                StoredIntegralCV* ptr,
                SizeType bitOffset,
                VHBits vhBits)
                : VHBits{vhBits}
                , ptr{ptr}
                , bitOffset{bitOffset}
            {
            }

            BitPackedIntRef(const BitPackedIntRef&) = default;

            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const BitPackedIntRef& other) -> BitPackedIntRef&
            {
                *this = static_cast<value_type>(other);
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator Integral() const
            {
                // fast path for single bits without sign handling
                if constexpr(std::is_empty_v<VHBits>)
                {
                    if constexpr(VHBits::value() == 1 && (std::is_unsigned_v<Integral> || SignBit == SignBit::Discard))
                    {
                        return bitunpack1<Integral>(ptr, static_cast<StoredIntegral>(bitOffset));
                    }
                }

                return bitunpack<SignBit == SignBit::Keep, Integral>(
                    ptr,
                    static_cast<StoredIntegral>(bitOffset),
                    static_cast<StoredIntegral>(VHBits::value()));
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Integral value) -> BitPackedIntRef&
            {
                // fast path for single bits without sign handling
                if constexpr(std::is_empty_v<VHBits>)
                {
                    if constexpr(VHBits::value() == 1 && (std::is_unsigned_v<Integral> || SignBit == SignBit::Discard))
                    {
                        bitpack1(ptr, static_cast<StoredIntegral>(bitOffset), value);
                    }
                }

                bitpack<SignBit == SignBit::Keep>(
                    ptr,
                    static_cast<StoredIntegral>(bitOffset),
                    static_cast<StoredIntegral>(VHBits::value()),
                    value);
                return *this;
            }
        };

        template<typename A, typename B>
        using HasLargerSize = mp_bool<sizeof(A) < sizeof(B)>;

        template<typename RecordDim>
        using LargestIntegral = mp_max_element<FlatRecordDim<RecordDim>, HasLargerSize>;

        template<typename RecordDim>
        using StoredUnsignedFor = std::
            conditional_t<(sizeof(LargestIntegral<RecordDim>) > sizeof(std::uint32_t)), std::uint64_t, std::uint32_t>;

        template<
            typename TArrayExtents,
            typename TRecordDim,
            typename Bits,
            SignBit SignBit,
            typename TLinearizeArrayDimsFunctor,
            typename TStoredIntegral>
        struct BitPackedIntCommon
            : MappingBase<TArrayExtents, TRecordDim>
            , protected llama::internal::BoxedValue<Bits>
        {
            using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
            using StoredIntegral = TStoredIntegral;

            static_assert(std::is_integral_v<StoredIntegral>);
            static_assert(std::is_unsigned_v<StoredIntegral>);

            // We could allow more integer types as storage type, but that needs to be thought through carefully
            static_assert(
                std::is_same_v<StoredIntegral, std::uint32_t> || std::is_same_v<StoredIntegral, std::uint64_t>);

        protected:
            using Base = MappingBase<TArrayExtents, TRecordDim>;
            using VHBits = llama::internal::BoxedValue<Bits>;
            using size_type = typename TArrayExtents::value_type;

            template<typename T>
            using IsAllowedFieldType = mp_or<std::is_integral<T>, std::is_enum<T>>;

            static_assert(
                mp_all_of<FlatRecordDim<TRecordDim>, IsAllowedFieldType>::value,
                "All record dimension field types must be integral");

            template<typename T>
            using IsFieldTypeSmallerOrEqualStorageIntegral = mp_bool<sizeof(T) <= sizeof(StoredIntegral)>;

            static_assert(
                mp_all_of<FlatRecordDim<TRecordDim>, IsFieldTypeSmallerOrEqualStorageIntegral>::value,
                "The integral type used for storage must be at least as big as the type of the values to retrieve");

        public:
            LLAMA_FN_HOST_ACC_INLINE
            constexpr auto bits() const -> size_type
            {
                return static_cast<size_type>(VHBits::value());
            }

            template<typename B = Bits, std::enable_if_t<isConstant<B>, int> = 0>
            LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntCommon(
                TArrayExtents extents = {},
                Bits bits = {},
                TRecordDim = {})
                : Base(extents)
                , VHBits{bits}
            {
                static_assert(VHBits::value() > 0);
                mp_for_each<mp_transform<mp_identity, FlatRecordDim<TRecordDim>>>(
                    [&](auto t)
                    {
                        using FieldType = typename decltype(t)::type;
                        static_assert(
                            static_cast<std::size_t>(VHBits::value()) <= sizeof(FieldType) * CHAR_BIT,
                            "Storage bits must not be greater than bits of field type");
                        static_assert(
                            VHBits::value() >= 2
                                || std::is_unsigned_v<FieldType> || SignBit == llama::mapping::SignBit::Discard,
                            "When keeping the sign bit, Bits must be at least 2 with signed integers in the record "
                            "dimension");
                    });
            }

            template<typename B = Bits, std::enable_if_t<!isConstant<B>, int> = 0>
            LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntCommon(
                TArrayExtents extents,
                Bits bits,
                TRecordDim = {})
                : Base(extents)
                , VHBits{bits}
            {
#ifdef __CUDA_ARCH__
                assert(VHBits::value() > 0);
#else
                if(VHBits::value() <= 0)
                    throw std::invalid_argument("BitPackedInt* Bits must not be zero");
#endif
                mp_for_each<mp_transform<mp_identity, FlatRecordDim<TRecordDim>>>(
                    [&](auto t)
                    {
                        using FieldType [[maybe_unused]] = typename decltype(t)::type;
#ifdef __CUDA_ARCH__
                        assert(VHBits::value() <= sizeof(FieldType) * CHAR_BIT);
#else
                        if(static_cast<std::size_t>(VHBits::value()) > sizeof(FieldType) * CHAR_BIT)
                            throw std::invalid_argument(
                                "BitPackedInt* Bits must not be larger than any field type in the record dimension");
                        if(!(VHBits::value() >= 2
                             || std::is_unsigned_v<FieldType> || SignBit == llama::mapping::SignBit::Discard))
                            throw std::invalid_argument("When keeping the sign bit, Bits must be at least 2 with "
                                                        "signed integers in the record "
                                                        "dimension");
#endif
                    });
            }

            template<std::size_t... RecordCoords>
            static constexpr auto isComputed(RecordCoord<RecordCoords...>)
            {
                return true;
            }
        };
    } // namespace internal

    /// Struct of array mapping using bit packing to reduce size/precision of integral data types. If your record
    /// dimension contains non-integral types, split them off using the \ref Split mapping first.
    /// \tparam Bits If Bits is llama::Constant<N>, the compile-time N specifies the number of bits to use. If Bits is
    /// an integral type T, the number of bits is specified at runtime, passed to the constructor and stored as type T.
    /// Must not be zero and must not be bigger than the bits of TStoredIntegral.
    /// @tparam SignBit When set to SignBit::Discard, discards the sign bit when storing signed integers. All
    /// numbers will be read back positive.
    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam TStoredIntegral Integral type used as storage of reduced precision integers. Must be std::uint32_t or
    /// std::uint64_t.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename Bits = typename TArrayExtents::value_type,
        SignBit SignBit = SignBit::Keep,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        typename TStoredIntegral = internal::StoredUnsignedFor<TRecordDim>>
    struct BitPackedIntSoA
        : internal::
              BitPackedIntCommon<TArrayExtents, TRecordDim, Bits, SignBit, TLinearizeArrayDimsFunctor, TStoredIntegral>
    {
    private:
        using Base = internal::
            BitPackedIntCommon<TArrayExtents, TRecordDim, Bits, SignBit, TLinearizeArrayDimsFunctor, TStoredIntegral>;

    public:
        using Base::Base;
        using typename Base::size_type;
        using VHBits = typename Base::VHBits; // use plain using declaration with nvcc >= 11.8

        static constexpr std::size_t blobCount = mp_size<FlatRecordDim<TRecordDim>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
        {
            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(TStoredIntegral) * CHAR_BIT);
            const auto bitsNeeded = TLinearizeArrayDimsFunctor{}.size(Base::extents()) * VHBits::value();
            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Base::ArrayIndex ai,
            RecordCoord<RecordCoords...>,
            Blobs& blobs) const
        {
            constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
            const auto bitOffset = TLinearizeArrayDimsFunctor{}(ai, Base::extents()) * VHBits::value();

            using QualifiedStoredIntegral = CopyConst<Blobs, TStoredIntegral>;
            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return internal::BitPackedIntRef<DstType, QualifiedStoredIntegral, VHBits, size_type, SignBit>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                static_cast<const VHBits&>(*this)};
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
    };

    /// Binds parameters to a \ref BitPackedIntSoA mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        typename Bits = void,
        SignBit SignBit = SignBit::Keep,
        typename LinearizeArrayDimsFunctor = mapping::LinearizeArrayDimsCpp,
        typename StoredIntegral = void>
    struct BindBitPackedIntSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = BitPackedIntSoA<
            ArrayExtents,
            RecordDim,
            std::conditional_t<!std::is_void_v<Bits>, Bits, typename ArrayExtents::value_type>,
            SignBit,
            LinearizeArrayDimsFunctor,
            std::conditional_t<
                !std::is_void_v<StoredIntegral>,
                StoredIntegral,
                internal::StoredUnsignedFor<RecordDim>>>;
    };

    template<typename Mapping>
    inline constexpr bool isBitPackedIntSoA = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        typename Bits,
        SignBit SignBit,
        typename LinearizeArrayDimsFunctor,
        typename StoredIntegral>
    inline constexpr bool isBitPackedIntSoA<
        BitPackedIntSoA<ArrayExtents, RecordDim, Bits, SignBit, LinearizeArrayDimsFunctor, StoredIntegral>>
        = true;

    /// Array of struct mapping using bit packing to reduce size/precision of integral data types. If your record
    /// dimension contains non-integral types, split them off using the \ref Split mapping first.
    /// \tparam Bits If Bits is llama::Constant<N>, the compile-time N specifies the number of bits to use. If Bits is
    /// an integral type T, the number of bits is specified at runtime, passed to the constructor and stored as type T.
    /// Must not be zero and must not be bigger than the bits of TStoredIntegral.
    /// @tparam SignBit When set to SignBit::Discard, discards the sign bit when storing signed integers. All
    /// numbers will be read back positive.
    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets.
    /// \tparam FlattenRecordDim Defines how the record dimension's fields should be flattened. See \ref
    //  FlattenRecordDimInOrder, \ref FlattenRecordDimIncreasingAlignment, \ref FlattenRecordDimDecreasingAlignment and
    //  \ref FlattenRecordDimMinimizePadding.
    /// \tparam TStoredIntegral Integral type used as storage of reduced precision integers. Must be std::uint32_t or
    /// std::uint64_t.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename Bits = typename TArrayExtents::value_type,
        SignBit SignBit = SignBit::Keep,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDim = FlattenRecordDimInOrder,
        typename TStoredIntegral = internal::StoredUnsignedFor<TRecordDim>>
    struct BitPackedIntAoS
        : internal::
              BitPackedIntCommon<TArrayExtents, TRecordDim, Bits, SignBit, TLinearizeArrayDimsFunctor, TStoredIntegral>
    {
    private:
        using Base = internal::
            BitPackedIntCommon<TArrayExtents, TRecordDim, Bits, SignBit, TLinearizeArrayDimsFunctor, TStoredIntegral>;

    public:
        using Base::Base;
        using typename Base::size_type;
        using VHBits = typename Base::VHBits; // use plain using declaration with nvcc >= 11.8

        using Flattener = FlattenRecordDim<TRecordDim>;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
        {
            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(TStoredIntegral) * CHAR_BIT);
            const auto bitsNeeded = TLinearizeArrayDimsFunctor{}.size(Base::extents())
                * static_cast<size_type>(VHBits::value()) * static_cast<size_type>(flatFieldCount<TRecordDim>);
            return roundUpToMultiple(bitsNeeded, bitsPerStoredIntegral) / CHAR_BIT;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Base::ArrayIndex ai,
            RecordCoord<RecordCoords...>,
            Blobs& blobs) const
        {
            constexpr auto flatFieldIndex = static_cast<size_type>(Flattener::template flatIndex<RecordCoords...>);
            const auto bitOffset = ((TLinearizeArrayDimsFunctor{}(ai, Base::extents())
                                     * static_cast<size_type>(flatFieldCount<TRecordDim>))
                                    + flatFieldIndex)
                * static_cast<size_type>(VHBits::value());

            using QualifiedStoredIntegral = CopyConst<Blobs, TStoredIntegral>;
            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return internal::BitPackedIntRef<DstType, QualifiedStoredIntegral, VHBits, size_type, SignBit>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[0][0]),
                bitOffset,
                static_cast<const VHBits&>(*this)};
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
    };

    /// Binds parameters to a \ref BitPackedIntAoS mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        typename Bits = void,
        SignBit SignBit = SignBit::Keep,
        typename LinearizeArrayDimsFunctor = mapping::LinearizeArrayDimsCpp,
        template<typename> typename FlattenRecordDim = FlattenRecordDimInOrder,
        typename StoredIntegral = void>
    struct BindBitPackedIntAoS
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = BitPackedIntAoS<
            ArrayExtents,
            RecordDim,
            std::conditional_t<!std::is_void_v<Bits>, Bits, typename ArrayExtents::value_type>,
            SignBit,
            LinearizeArrayDimsFunctor,
            FlattenRecordDim,
            std::conditional_t<
                !std::is_void_v<StoredIntegral>,
                StoredIntegral,
                internal::StoredUnsignedFor<RecordDim>>>;
    };

    template<typename Mapping>
    inline constexpr bool isBitPackedIntAoS = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        typename Bits,
        SignBit SignBit,
        typename LinearizeArrayDimsFunctor,
        template<typename>
        typename FlattenRecordDim,
        typename StoredIntegral>
    inline constexpr bool isBitPackedIntAoS<BitPackedIntAoS<
        ArrayExtents,
        RecordDim,
        Bits,
        SignBit,
        LinearizeArrayDimsFunctor,
        FlattenRecordDim,
        StoredIntegral>>
        = true;
} // namespace llama::mapping

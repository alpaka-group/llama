// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"
#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"

#include <climits>
#include <type_traits>

namespace llama::mapping
{
    namespace internal
    {
        /// A proxy type representing a reference to a reduced precision integral value, stored in a buffer at a
        /// specified bit offset.
        /// @tparam Integral Integral data type which can be loaded and store through this reference.
        /// @tparam StoredIntegralPointer Pointer to integral type used for storing the bits.
        template<typename Integral, typename StoredIntegralPointer, typename VHBits, typename SizeType>
        struct BitPackedIntRef
            : private VHBits
            , ProxyRefOpMixin<BitPackedIntRef<Integral, StoredIntegralPointer, VHBits, SizeType>, Integral>
        {
        private:
            using StoredIntegral = std::remove_const_t<std::remove_pointer_t<StoredIntegralPointer>>;

            static_assert(std::is_integral_v<StoredIntegral>);
            static_assert(std::is_unsigned_v<StoredIntegral>);
            static_assert(
                sizeof(StoredIntegral) >= sizeof(Integral),
                "The integral type used for the storage must be at least as big as the type of the values to "
                "retrieve");

            StoredIntegralPointer ptr;
            SizeType bitOffset;
#ifndef NDEBUG
            StoredIntegralPointer endPtr;
#endif

            // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
            static constexpr auto bitsPerStoredIntegral = static_cast<SizeType>(sizeof(StoredIntegral) * CHAR_BIT);

            LLAMA_FN_HOST_ACC_INLINE static constexpr auto makeMask(StoredIntegral bits) -> StoredIntegral
            {
                return bits >= sizeof(StoredIntegral) * CHAR_BIT ? ~StoredIntegral{0}
                                                                 : (StoredIntegral{1} << bits) - 1u;
            }

        public:
            using value_type = Integral;

            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedIntRef(
                StoredIntegralPointer ptr,
                SizeType bitOffset,
                VHBits vhBits
#ifndef NDEBUG
                ,
                StoredIntegralPointer endPtr
#endif
                )
                : VHBits{vhBits}
                , ptr{ptr}
                , bitOffset{bitOffset}

#ifndef NDEBUG
                , endPtr{endPtr}
#endif
            {
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator Integral() const
            {
                auto* p = ptr + bitOffset / bitsPerStoredIntegral;
                const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;
                assert(p < endPtr);
                auto v = p[0] >> innerBitOffset;

                const auto innerBitEndOffset = innerBitOffset + VHBits::value();
                if(innerBitEndOffset <= bitsPerStoredIntegral)
                {
                    const auto mask = makeMask(VHBits::value());
                    v &= mask;
                }
                else
                {
                    const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
                    const auto bitsLoaded = bitsPerStoredIntegral - innerBitOffset;
                    const auto mask = makeMask(excessBits);
                    assert(p + 1 < endPtr);
                    v |= (p[1] & mask) << bitsLoaded;
                }
                if constexpr(std::is_signed_v<Integral>)
                {
                    // perform sign extension
                    if((v & (StoredIntegral{1} << (VHBits::value() - 1))) && VHBits::value() < bitsPerStoredIntegral)
                        v |= ~StoredIntegral{0} << VHBits::value();
                }
                return static_cast<Integral>(v);
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Integral value) -> BitPackedIntRef&
            {
                // NOLINTNEXTLINE(bugprone-signed-char-misuse,cert-str34-c)
                const auto unsignedValue = static_cast<StoredIntegral>(value);
                const auto mask = makeMask(VHBits::value());
                StoredIntegral valueBits;
                if constexpr(!std::is_signed_v<Integral>)
                    valueBits = unsignedValue & mask;
                else
                {
                    const auto magnitudeMask = makeMask(VHBits::value() - 1);
                    const auto isSigned = value < 0;
                    valueBits = (StoredIntegral{isSigned} << (VHBits::value() - 1)) | (unsignedValue & magnitudeMask);
                }

                auto* p = ptr + bitOffset / bitsPerStoredIntegral;
                const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;

                {
                    const auto clearMask = ~(mask << innerBitOffset);
                    assert(p < endPtr);
                    auto mem = p[0] & clearMask; // clear previous bits
                    mem |= valueBits << innerBitOffset; // write new bits
                    p[0] = mem;
                }

                const auto innerBitEndOffset = innerBitOffset + VHBits::value();
                if(innerBitEndOffset > bitsPerStoredIntegral)
                {
                    const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
                    const auto bitsWritten = bitsPerStoredIntegral - innerBitOffset;
                    const auto clearMask = ~makeMask(excessBits);
                    assert(p + 1 < endPtr);
                    auto mem = p[1] & clearMask; // clear previous bits
                    mem |= valueBits >> bitsWritten; // write new bits
                    p[1] = mem;
                }

                return *this;
            }
        };

        template<typename A, typename B>
        using HasLargerSize = boost::mp11::mp_bool<sizeof(A) < sizeof(B)>;

        template<typename RecordDim>
        using LargestIntegral = boost::mp11::mp_max_element<FlatRecordDim<RecordDim>, HasLargerSize>;

        template<typename RecordDim>
        using StoredUnsignedFor = std::
            conditional_t<(sizeof(LargestIntegral<RecordDim>) > sizeof(std::uint32_t)), std::uint64_t, std::uint32_t>;
    } // namespace internal

    /// Struct of array mapping using bit packing to reduce size/precision of integral data types. If your record
    /// dimension contains non-integral types, split them off using the \ref Split mapping first.
    /// \tparam Bits If Bits is llama::Constant<N>, the compile-time N specifies the number of bits to use. If Bits is
    /// an integral type T, the number of bits is specified at runtime, passed to the constructor and stored as type T.
    /// Must not be zero.
    /// \tparam TLinearizeArrayDimsFunctor Defines how the array dimensions should be mapped into linear numbers and
    /// how big the linear domain gets. \tparam TStoredIntegral Integral type used as storage of reduced precision
    /// integers.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename Bits = typename TArrayExtents::value_type,
        typename TLinearizeArrayDimsFunctor = LinearizeArrayDimsCpp,
        typename TStoredIntegral = internal::StoredUnsignedFor<TRecordDim>>
    struct BitPackedIntSoA
        : MappingBase<TArrayExtents, TRecordDim>
        , private llama::internal::BoxedValue<Bits>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using VHBits = llama::internal::BoxedValue<Bits>;
        using size_type = typename TArrayExtents::value_type;

        template<typename T>
        using IsAllowedFieldType = boost::mp11::mp_or<std::is_integral<T>, std::is_enum<T>>;

        static_assert(
            boost::mp11::mp_all_of<FlatRecordDim<TRecordDim>, IsAllowedFieldType>::value,
            "All record dimension field types must be integral");

    public:
        using LinearizeArrayDimsFunctor = TLinearizeArrayDimsFunctor;
        using StoredIntegral = TStoredIntegral;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<FlatRecordDim<TRecordDim>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto bits() const -> size_type
        {
            return static_cast<size_type>(VHBits::value());
        }

        template<typename B = Bits, std::enable_if_t<isConstant<B>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntSoA(
            TArrayExtents extents = {},
            Bits bits = {},
            TRecordDim = {})
            : Base(extents)
            , VHBits{bits}
        {
            static_assert(VHBits::value() > 0);
            boost::mp11::mp_for_each<boost::mp11::mp_transform<boost::mp11::mp_identity, FlatRecordDim<TRecordDim>>>(
                [&](auto t)
                {
                    using FieldType = typename decltype(t)::type;
                    static_assert(static_cast<std::size_t>(VHBits::value()) <= sizeof(FieldType) * CHAR_BIT);
                });
        }

        template<typename B = Bits, std::enable_if_t<!isConstant<B>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE constexpr explicit BitPackedIntSoA(TArrayExtents extents, Bits bits, TRecordDim = {})
            : Base(extents)
            , VHBits{bits}
        {
#ifdef __CUDA_ARCH__
            assert(VHBits::value() > 0);
#else
            if(VHBits::value() <= 0)
                throw std::invalid_argument("BitPackedIntSoA Bits must not be zero");
#endif
            boost::mp11::mp_for_each<boost::mp11::mp_transform<boost::mp11::mp_identity, FlatRecordDim<TRecordDim>>>(
                [&](auto t)
                {
                    using FieldType = typename decltype(t)::type;
#ifdef __CUDA_ARCH__
                    assert(VHBits::value() <= sizeof(FieldType) * CHAR_BIT);
#else
                    if(static_cast<std::size_t>(VHBits::value()) > sizeof(FieldType) * CHAR_BIT)
                        throw std::invalid_argument(
                            "BitPackedIntSoA Bits must not be larger than any field type in the record dimension");
#endif
                });
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
        {
            constexpr auto bitsPerStoredIntegral = static_cast<size_type>(sizeof(StoredIntegral) * CHAR_BIT);
            const auto bitsNeeded = LinearizeArrayDimsFunctor{}.size(Base::extents()) * VHBits::value();
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
            constexpr auto blob = flatRecordCoord<TRecordDim, RecordCoord<RecordCoords...>>;
            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, Base::extents()) * VHBits::value();

            using QualifiedStoredIntegral = CopyConst<Blobs, StoredIntegral>;
            using DstType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return internal::BitPackedIntRef<DstType, QualifiedStoredIntegral*, VHBits, size_type>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                static_cast<const VHBits&>(*this)
#ifndef NDEBUG
                    ,
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0] + blobSize(blob))
#endif
            };
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
    };

    /// Binds parameters to a \ref BitPackedIntSoA mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        typename Bits = unsigned,
        typename LinearizeArrayDimsFunctor = mapping::LinearizeArrayDimsCpp,
        typename StoredIntegral = void>
    struct BindBitPackedIntSoA
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = BitPackedIntSoA<
            ArrayExtents,
            RecordDim,
            Bits,
            LinearizeArrayDimsFunctor,
            std::conditional_t<
                !std::is_void_v<StoredIntegral>,
                StoredIntegral,
                internal::StoredUnsignedFor<RecordDim>>>;
    };

    template<typename Mapping>
    inline constexpr bool isBitPackedIntSoA = false;

    template<typename... Ts>
    inline constexpr bool isBitPackedIntSoA<BitPackedIntSoA<Ts...>> = true;
} // namespace llama::mapping

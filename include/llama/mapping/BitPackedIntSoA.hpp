// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"

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
        template<typename Integral, typename StoredIntegralPointer>
        struct BitPackedIntRef : ProxyRefOpMixin<BitPackedIntRef<Integral, StoredIntegralPointer>, Integral>
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
            std::size_t bitOffset;
            unsigned bits;
#ifndef NDEBUG
            StoredIntegralPointer endPtr;
#endif

            static constexpr auto bitsPerStoredIntegral = sizeof(StoredIntegral) * CHAR_BIT;

        public:
            using value_type = Integral;

            LLAMA_FN_HOST_ACC_INLINE constexpr BitPackedIntRef(
                StoredIntegralPointer ptr,
                std::size_t bitOffset,
                unsigned bits
#ifndef NDEBUG
                ,
                StoredIntegralPointer endPtr
#endif
                )
                : ptr{ptr}
                , bitOffset{bitOffset}
                , bits
            {
                bits
            }
#ifndef NDEBUG
            , endPtr
            {
                endPtr
            }
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

                const auto innerBitEndOffset = innerBitOffset + bits;
                if(innerBitEndOffset <= bitsPerStoredIntegral)
                {
                    const auto mask = (StoredIntegral{1} << bits) - 1u;
                    v &= mask;
                }
                else
                {
                    const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
                    const auto bitsLoaded = bitsPerStoredIntegral - innerBitOffset;
                    const auto mask = (StoredIntegral{1} << excessBits) - 1u;
                    assert(p + 1 < endPtr);
                    v |= (p[1] & mask) << bitsLoaded;
                }
                if constexpr(std::is_signed_v<Integral>)
                    if(v & (StoredIntegral{1} << (bits - 1)))
                        v |= ~StoredIntegral{0} << bits; // sign extend
                return static_cast<Integral>(v);
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(Integral value) -> BitPackedIntRef&
            {
                const auto unsignedValue = static_cast<StoredIntegral>(value);
                const auto mask = (StoredIntegral{1} << bits) - 1u;
                StoredIntegral valueBits;
                if constexpr(!std::is_signed_v<Integral>)
                    valueBits = unsignedValue & mask;
                else
                {
                    const auto magnitudeMask = (StoredIntegral{1} << (bits - 1)) - 1u;
                    const auto isSigned = value < 0;
                    valueBits = (StoredIntegral{isSigned} << (bits - 1)) | (unsignedValue & magnitudeMask);
                }

                auto* p = ptr + bitOffset / bitsPerStoredIntegral;
                const auto innerBitOffset = bitOffset % bitsPerStoredIntegral;
                const auto clearMask = ~(mask << innerBitOffset);
                assert(p < endPtr);
                auto mem = p[0] & clearMask; // clear previous bits
                mem |= valueBits << innerBitOffset; // write new bits
                p[0] = mem;

                const auto innerBitEndOffset = innerBitOffset + bits;
                if(innerBitEndOffset > bitsPerStoredIntegral)
                {
                    const auto excessBits = innerBitEndOffset - bitsPerStoredIntegral;
                    const auto bitsWritten = bitsPerStoredIntegral - innerBitOffset;
                    const auto clearMask = ~((StoredIntegral{1} << excessBits) - 1u);
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

        template<typename T, typename SFINAE = void>
        struct MakeUnsigned : std::make_unsigned<T>
        {
        };

        template<>
        struct MakeUnsigned<bool>
        {
            using type = std::uint8_t;
        };

        template<typename T>
        struct MakeUnsigned<T, std::enable_if_t<std::is_enum_v<T>>> : std::make_unsigned<std::underlying_type_t<T>>
        {
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
        template<typename T>
        using IsAllowedFieldType = boost::mp11::mp_or<std::is_integral<T>, std::is_enum<T>>;

        static_assert(
            boost::mp11::mp_all_of<FlatRecordDim<TRecordDim>, IsAllowedFieldType>::value,
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
            return internal::BitPackedIntRef<DstType, QualifiedStoredIntegral*>{
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                bits
#ifndef NDEBUG
                ,
                reinterpret_cast<QualifiedStoredIntegral*>(&blobs[blob][0] + blobSize(blob))
#endif
            };
        }

    private:
        unsigned bits = 0;
    };
} // namespace llama::mapping

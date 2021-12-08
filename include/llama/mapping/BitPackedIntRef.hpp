#pragma once

#include <climits>
#include <type_traits>

namespace llama::internal
{
    /// A proxy type representing a reference to a reduced precision integral value, stored in a buffer at a specified
    /// bit offset.
    /// @tparam Integral Integral data type which can be loaded and store through this reference.
    /// @tparam StoredIntegralPointer Pointer to integral type used for storing the bits.
    template<typename Integral, typename StoredIntegralPointer>
    struct BitPackedIntRef
    {
        using StoredIntegral = std::remove_const_t<std::remove_pointer_t<StoredIntegralPointer>>;

        static_assert(std::is_integral_v<Integral>);
        static_assert(std::is_integral_v<StoredIntegral>);
        static_assert(std::is_unsigned_v<StoredIntegral>);
        static_assert(
            sizeof(StoredIntegral) >= sizeof(Integral),
            "The integral type used for the storage must be at least as big as the type of the values to retrieve");

        StoredIntegralPointer ptr;
        std::size_t bitOffset;
        unsigned bits;
#ifndef NDEBUG
        StoredIntegralPointer endPtr;
#endif

        static constexpr auto bitsPerStoredIntegral = sizeof(StoredIntegral) * CHAR_BIT;

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        operator Integral() const
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

        auto operator=(Integral value) -> BitPackedIntRef&
        {
            const auto unsignedValue = static_cast<StoredIntegral>(value);
            const auto mask = (StoredIntegral{1} << bits) - 1u;
            StoredIntegral valueBits;
            if constexpr(std::is_unsigned_v<Integral>)
                valueBits = unsignedValue & mask;
            else
            {
                const auto magnitudeMask = (StoredIntegral{1} << (bits - std::is_signed_v<Integral>) ) - 1u;
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
} // namespace llama::internal

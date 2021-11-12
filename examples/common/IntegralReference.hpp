#pragma once

#include <climits>
#include <type_traits>

namespace internal
{
    /// A proxy type representing a reference to an integral value, stored in a buffer at a specified bit offset.
    /// @tparam Integral Integral data type which can be loaded and store through this reference.
    /// @tparam StoredIntegralPointer Pointer to integral type used for storing the bits.
    template<typename Integral, typename StoredIntegralPointer>
    struct IntegralReference
    {
        using StoredIntegral = std::remove_const_t<std::remove_pointer_t<StoredIntegralPointer>>;

        static_assert(std::is_integral_v<Integral>);
        static_assert(std::is_integral_v<StoredIntegral>);
        static_assert(
            sizeof(StoredIntegral) >= sizeof(Integral),
            "The integral type used for the storage must be at least as big as the type of the values to retrieve");

        StoredIntegralPointer ptr;
        std::size_t bitOffset;
        unsigned bits;

        static constexpr auto registerBits = sizeof(StoredIntegral) * CHAR_BIT;

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        operator Integral() const
        {
            auto* p = ptr + bitOffset / registerBits;
            const auto innerBitOffset = bitOffset % registerBits;
            auto v = p[0] >> innerBitOffset;

            const auto innerBitEndOffset = innerBitOffset + bits;
            if(innerBitEndOffset <= registerBits)
            {
                const auto mask = (StoredIntegral{1} << bits) - 1u;
                v &= mask;
            }
            else
            {
                const auto excessBits = innerBitEndOffset - registerBits;
                const auto bitsLoaded = registerBits - innerBitOffset;
                const auto mask = (StoredIntegral{1} << excessBits) - 1u;
                v |= (p[1] & mask) << bitsLoaded;
            }
            if constexpr(std::is_signed_v<Integral>)
                if((v & (StoredIntegral{1} << (bits - 1))) != 0)
                {
                    // sign extend
                    v |= static_cast<StoredIntegral>(-1) << bits;
                }
            return static_cast<Integral>(v);
        }

        auto operator=(Integral v) -> IntegralReference&
        {
            const auto mask = (StoredIntegral{1} << bits) - 1u;
            const auto vBits = (static_cast<StoredIntegral>(v) & mask);

            auto* p = ptr + bitOffset / registerBits;
            const auto innerBitOffset = bitOffset % registerBits;
            const auto clearMask = ~(mask << innerBitOffset);
            auto m = p[0] & clearMask; // clear previous bits
            m |= vBits << innerBitOffset; // write new bits
            p[0] = m;

            const auto innerBitEndOffset = innerBitOffset + bits;
            if(innerBitEndOffset > registerBits)
            {
                const auto excessBits = innerBitEndOffset - registerBits;
                const auto bitsWritten = registerBits - innerBitOffset;
                const auto clearMask = ~((StoredIntegral{1} << excessBits) - 1u);
                auto m = p[1] & clearMask; // clear previous bits
                m |= vBits >> bitsWritten; // write new bits
                p[1] = m;
            }

            return *this;
        }
    };
} // namespace internal

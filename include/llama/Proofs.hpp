// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "ArrayIndexRange.hpp"
#include "Core.hpp"

namespace llama
{
    namespace internal
    {
        constexpr auto divRoundUp(std::size_t dividend, std::size_t divisor) -> std::size_t
        {
            return (dividend + divisor - 1) / divisor;
        }
    } // namespace internal

// FIXME(bgruber): this test is actually not correct, because __cpp_constexpr_dynamic_alloc only guarantees constexpr
// std::allocator
#ifdef __cpp_constexpr_dynamic_alloc
    namespace internal
    {
        template<typename T>
        struct DynArray
        {
            constexpr DynArray() = default;

            constexpr explicit DynArray(std::size_t n) : data(new T[n]{})
            {
            }

            DynArray(const DynArray&) = delete;
            DynArray(DynArray&&) = delete;
            auto operator=(const DynArray&) -> DynArray& = delete;
            auto operator=(DynArray&&) -> DynArray& = delete;

            constexpr ~DynArray()
            {
                delete[] data;
            }

            constexpr void resize(std::size_t n)
            {
                delete[] data;
                data = new T[n]{};
            }

            T* data = nullptr; // TODO(bgruber): replace by std::unique_ptr in C++23
        };
    } // namespace internal

    /// Proofs by exhaustion of the array and record dimensions, that all values mapped to memory do not overlap.
    // Unfortunately, this only works for smallish array dimensions, because of compiler limits on constexpr evaluation
    // depth.
    template<typename Mapping>
    constexpr auto mapsNonOverlappingly(const Mapping& m) -> bool
    {
        internal::DynArray<internal::DynArray<std::uint64_t>> blobByteMapped(m.blobCount);
        for(std::size_t i = 0; i < m.blobCount; i++)
            blobByteMapped.data[i].resize(internal::divRoundUp(m.blobSize(i), 64));

        auto testAndSet = [&](auto blob, auto offset) constexpr
        {
            const auto bit = std::uint64_t{1} << (offset % 64);
            if(blobByteMapped.data[blob].data[offset / 64] & bit)
                return true;
            blobByteMapped.data[blob].data[offset / 64] |= bit;
            return false;
        };

        bool collision = false;
        forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc) constexpr {
            if(collision)
                return;
            for(auto ai : ArrayIndexRange{m.extents()})
            {
                using Type = GetType<typename Mapping::RecordDim, decltype(rc)>;
                const auto [blob, offset] = m.blobNrAndOffset(ai, rc);
                for(std::size_t b = 0; b < sizeof(Type); b++)
                    if(testAndSet(blob, offset + b))
                    {
                        collision = true;
                        break;
                    }
            }
        });
        return !collision;
    }
#endif

    /// Proofs by exhaustion of the array and record dimensions, that at least PieceLength elements are always stored
    /// contiguously.
    // Unfortunately, this only works for smallish array dimensions, because of compiler limits on constexpr evaluation
    // depth.
    template<std::size_t PieceLength, typename Mapping>
    constexpr auto mapsPiecewiseContiguous(const Mapping& m) -> bool
    {
        bool collision = false;
        forEachLeafCoord<typename Mapping::RecordDim>([&](auto rc) constexpr {
            std::size_t flatIndex = 0;
            std::size_t lastBlob = std::numeric_limits<std::size_t>::max();
            std::size_t lastOffset = std::numeric_limits<std::size_t>::max();
            for(auto ai : ArrayIndexRange{m.extents()})
            {
                using Type = GetType<typename Mapping::RecordDim, decltype(rc)>;
                const auto [blob, offset] = m.blobNrAndOffset(ai, rc);
                if(flatIndex % PieceLength != 0 && (lastBlob != blob || lastOffset + sizeof(Type) != offset))
                {
                    collision = true;
                    break;
                }
                lastBlob = blob;
                lastOffset = offset;
                flatIndex++;
            }
        });
        return !collision;
    }
} // namespace llama

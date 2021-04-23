// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// FIXME: this test is actually not correct, because __cpp_constexpr_dynamic_alloc only guarantees constexpr
// std::allocator
#ifdef __cpp_constexpr_dynamic_alloc

#    include "ArrayDimsIndexRange.hpp"
#    include "Core.hpp"

namespace llama
{
    namespace internal
    {
        template <typename Mapping, std::size_t... Is, typename ArrayDims>
        constexpr auto blobNrAndOffset(const Mapping& m, llama::RecordCoord<Is...>, ArrayDims ad)
        {
            return m.template blobNrAndOffset<Is...>(ad);
        }

        constexpr auto divRoundUp(std::size_t dividend, std::size_t divisor) -> std::size_t
        {
            return (dividend + divisor - 1) / divisor;
        }

        template <typename T>
        struct DynArray
        {
            constexpr DynArray() = default;

            constexpr DynArray(std::size_t n)
            {
                data = new T[n]{};
            }

            constexpr ~DynArray()
            {
                delete[] data;
            }

            constexpr void resize(std::size_t n)
            {
                delete[] data;
                data = new T[n]{};
            }

            T* data = nullptr;
        };
    } // namespace internal

    // Proofs by exhaustion of the array and record dimensions, that all values mapped to memory do not overlap.
    // Unfortunately, this only works for smallish array dimensions, because of compiler limits on constexpr evaluation
    // depth.
    template <typename Mapping>
    constexpr auto mapsNonOverlappingly(const Mapping& m) -> bool
    {
        internal::DynArray<internal::DynArray<std::uint64_t>> blobByteMapped(m.blobCount);
        for (auto i = 0; i < m.blobCount; i++)
            blobByteMapped.data[i].resize(internal::divRoundUp(m.blobSize(i), 64));

        auto testAndSet = [&](auto blob, auto offset) constexpr
        {
            const auto bit = std::uint64_t{1} << (offset % 64);
            if (blobByteMapped.data[blob].data[offset / 64] & bit)
                return true;
            blobByteMapped.data[blob].data[offset / 64] |= bit;
            return false;
        };

        bool collision = false;
        llama::forEachLeaf<
            typename Mapping::RecordDim>([&](auto coord) constexpr
                                         {
                                             if (collision)
                                                 return;
                                             for (auto ad : llama::ArrayDimsIndexRange{m.arrayDims()})
                                             {
                                                 using Type
                                                     = llama::GetType<typename Mapping::RecordDim, decltype(coord)>;
                                                 const auto [blob, offset] = internal::blobNrAndOffset(m, coord, ad);
                                                 for (auto b = 0; b < sizeof(Type); b++)
                                                     if (testAndSet(blob, offset + b))
                                                     {
                                                         collision = true;
                                                         break;
                                                     }
                                             }
                                         });
        return !collision;
    }
} // namespace llama

#endif

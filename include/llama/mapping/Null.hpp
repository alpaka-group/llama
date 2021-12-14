// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename T>
        struct NullReference : ProxyRefOpMixin<NullReference<T>, T>
        {
            using value_type = T;

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator T() const
            {
                return T{}; // this might not be the best design decision
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(T) -> NullReference&
            {
                return *this;
            }
        };
    } // namespace internal

    /// The Null mappings maps all elements to nothing. Writing data through a reference obtained from the Null mapping
    /// discards the value. Reading through such a reference returns a default constructed object.
    template<typename TArrayExtents, typename TRecordDim>
    struct Null : TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = 0;

        constexpr Null() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Null(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return *this; // NOLINT(cppcoreguidelines-slicing)
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(std::size_t /*blobIndex*/) const -> std::size_t
        {
            return 0;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(ArrayIndex, RecordCoord<RecordCoords...>, Blobs&) const
        {
            using FieldType = GetType<RecordDim, RecordCoord<RecordCoords...>>;
            return internal::NullReference<FieldType>{};
        }
    };
} // namespace llama::mapping

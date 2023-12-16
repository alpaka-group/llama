// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

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
    LLAMA_EXPORT
    template<typename TArrayExtents, typename TRecordDim>
    struct Null : MappingBase<TArrayExtents, TRecordDim>
    {
    private:
        using Base = MappingBase<TArrayExtents, TRecordDim>;
        using size_type = typename TArrayExtents::value_type;

    public:
        static constexpr std::size_t blobCount = 0;

        using Base::Base;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type /*blobIndex*/) const -> size_type
        {
            return 0;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Base::ArrayIndex,
            RecordCoord<RecordCoords...>,
            Blobs&) const
        {
            using FieldType = GetType<TRecordDim, RecordCoord<RecordCoords...>>;
            return internal::NullReference<FieldType>{};
        }
    };

    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool isNull = false;

    LLAMA_EXPORT
    template<typename ArrayExtents, typename RecordDim>
    inline constexpr bool isNull<Null<ArrayExtents, RecordDim>> = true;
} // namespace llama::mapping

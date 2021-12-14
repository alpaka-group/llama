// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename T>
        using ReplaceByByteArray = std::byte[sizeof(T)];

        template<typename RecordDim>
        using SplitBytes = TransformLeaves<RecordDim, ReplaceByByteArray>;
    } // namespace internal

    /// Meta mapping splitting each field in the record dimension into an array of bytes and mapping the resulting
    /// record dimension using a further mapping.
    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
    struct Bytesplit : private InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>
    {
        using Inner = InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>;

        using ArrayExtents = typename Inner::ArrayExtents;
        using ArrayIndex = typename Inner::ArrayIndex;
        using RecordDim = TRecordDim; // hide Inner::RecordDim
        using Inner::blobCount;

        using Inner::blobSize;
        using Inner::extents;
        using Inner::Inner;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit Bytesplit(TArrayExtents extents, TRecordDim = {}) : Inner(extents)
        {
        }

        template<typename... Args>
        LLAMA_FN_HOST_ACC_INLINE constexpr explicit Bytesplit(std::tuple<Args...> innerMappingArgs, TRecordDim = {})
            : Inner(std::make_from_tuple<Inner>(innerMappingArgs))
        {
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<typename RC, typename BlobArray>
        struct Reference : ProxyRefOpMixin<Reference<RC, BlobArray>, GetType<TRecordDim, RC>>
        {
            const Inner& inner;
            ArrayIndex ai;
            BlobArray& blobs;

        public:
            using value_type = GetType<TRecordDim, RC>;

            Reference(const Inner& innerMapping, ArrayIndex ai, BlobArray& blobs)
                : inner(innerMapping)
                , ai(ai)
                , blobs(blobs)
            {
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            operator value_type() const
            {
                value_type v;
                auto* p = reinterpret_cast<std::byte*>(&v);
                boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(value_type)>>(
                    [&](auto ic)
                    {
                        constexpr auto i = decltype(ic)::value;
                        auto&& ref
                            = llama::internal::resolveToMemoryReference(blobs, inner, ai, Cat<RC, RecordCoord<i>>{});
                        p[i] = ref;
                    });
                return v;
            }

            auto operator=(value_type v) -> Reference&
            {
                auto* p = reinterpret_cast<std::byte*>(&v);
                boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(value_type)>>(
                    [&](auto ic)
                    {
                        constexpr auto i = decltype(ic)::value;
                        auto&& ref
                            = llama::internal::resolveToMemoryReference(blobs, inner, ai, Cat<RC, RecordCoord<i>>{});
                        ref = p[i];
                    });
                return *this;
            }
        };

        template<std::size_t... RecordCoords, typename BlobArray>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Inner::ArrayIndex ai,
            RecordCoord<RecordCoords...>,
            BlobArray& blobs) const
        {
            return Reference<RecordCoord<RecordCoords...>, BlobArray>{*this, ai, blobs};
        }
    };
} // namespace llama::mapping

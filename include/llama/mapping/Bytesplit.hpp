// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

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

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<typename QualifiedBase, std::size_t N, typename RC, typename BlobArray>
        struct Reference
        {
            QualifiedBase& innerMapping;
            ArrayIndex ai;
            llama::Array<std::size_t, N> dynamicArrayExtents;
            BlobArray& blobs;

            using DstType = GetType<TRecordDim, RC>;

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            operator DstType() const
            {
                DstType v;
                auto* p = reinterpret_cast<std::byte*>(&v);
                boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(DstType)>>(
                    [&](auto ic)
                    {
                        constexpr auto i = decltype(ic)::value;
                        const auto [nr, off]
                            = innerMapping.blobNrAndOffset(ai, dynamicArrayExtents, Cat<RC, RecordCoord<i>>{});
                        p[i] = blobs[nr][off];
                    });
                return v;
            }

            auto operator=(DstType v) -> Reference&
            {
                auto* p = reinterpret_cast<std::byte*>(&v);
                boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(DstType)>>(
                    [&](auto ic)
                    {
                        constexpr auto i = decltype(ic)::value;
                        const auto [nr, off]
                            = innerMapping.blobNrAndOffset(ai, dynamicArrayExtents, Cat<RC, RecordCoord<i>>{});
                        blobs[nr][off] = p[i];
                    });
                return *this;
            }
        };

        template<std::size_t... RecordCoords, std::size_t N, typename BlobArray>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Inner::ArrayIndex ai,
            llama::Array<std::size_t, N> dynamicArrayExtents,
            RecordCoord<RecordCoords...>,
            BlobArray& blobs) const
        {
            return Reference<decltype(*this), N, RecordCoord<RecordCoords...>, BlobArray>{
                *this,
                ai,
                dynamicArrayExtents,
                blobs};
        }
    };
} // namespace llama::mapping

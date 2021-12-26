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
                        auto&& ref = mapToMemory(inner, ai, Cat<RC, RecordCoord<i>>{}, blobs);
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
                        auto&& ref = mapToMemory(inner, ai, Cat<RC, RecordCoord<i>>{}, blobs);
                        ref = p[i];
                    });
                return *this;
            }
        };

        template<std::size_t... RecordCoords, typename BlobArray>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(ArrayIndex ai, RecordCoord<RecordCoords...>, BlobArray& blobs)
            const
        {
            return Reference<RecordCoord<RecordCoords...>, BlobArray>{*this, ai, blobs};
        }
    };

    /// Binds parameters to a \ref Bytesplit mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<template<typename, typename> typename InnerMapping>
    struct BindBytesplit
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = Bytesplit<ArrayExtents, RecordDim, InnerMapping>;
    };

    template<typename Mapping>
    inline constexpr bool isBytesplit = false;

    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
    inline constexpr bool isBytesplit<Bytesplit<TArrayExtents, TRecordDim, InnerMapping>> = true;
} // namespace llama::mapping

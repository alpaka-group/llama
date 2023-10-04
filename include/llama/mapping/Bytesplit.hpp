// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

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
    LLAMA_EXPORT
    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
    struct Bytesplit : private InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>
    {
        using Inner = InnerMapping<TArrayExtents, internal::SplitBytes<TRecordDim>>;

        using ArrayExtents = typename Inner::ArrayExtents;
        using RecordDim = TRecordDim; // hide Inner::RecordDim
        using Inner::blobCount;

        using Inner::blobSize;
        using Inner::extents;

    private:
        using ArrayIndex = typename TArrayExtents::Index;

    public:
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
        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
        struct Reference : ProxyRefOpMixin<Reference<RC, BlobArray>, GetType<TRecordDim, RC>>
        {
        private:
            const Inner& inner;
            ArrayIndex ai;
            BlobArray& blobs;

        public:
            using value_type = GetType<TRecordDim, RC>;

            LLAMA_FN_HOST_ACC_INLINE constexpr Reference(const Inner& innerMapping, ArrayIndex ai, BlobArray& blobs)
                : inner(innerMapping)
                , ai(ai)
                , blobs(blobs)
            {
            }

            Reference(const Reference&) = default;

            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const Reference& other) -> Reference&
            {
                *this = static_cast<value_type>(other);
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator value_type() const
            {
#ifdef _MSC_VER
                // MSVC workaround. Without this, MSVC deduces the last template parameter of mapToMemory wrongly
                BlobArray& blobs = this->blobs;
#endif

                value_type v;
                auto* p = reinterpret_cast<std::byte*>(&v);
                mp_for_each<mp_iota_c<sizeof(value_type)>>(
                    [&](auto ic) LLAMA_LAMBDA_INLINE
                    {
                        constexpr auto i = decltype(ic)::value;
                        auto&& ref = mapToMemory(inner, ai, Cat<RC, RecordCoord<i>>{}, blobs);

                        p[i] = ref;
                    });
                return v;
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(value_type v) -> Reference&
            {
#ifdef _MSC_VER
                // MSVC workaround. Without this, MSVC deduces the last template parameter of mapToMemory wrongly
                BlobArray& blobs = this->blobs;
#endif

                auto* p = reinterpret_cast<std::byte*>(&v);
                mp_for_each<mp_iota_c<sizeof(value_type)>>(
                    [&](auto ic) LLAMA_LAMBDA_INLINE
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
    LLAMA_EXPORT
    template<template<typename, typename> typename InnerMapping>
    struct BindBytesplit
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = Bytesplit<ArrayExtents, RecordDim, InnerMapping>;
    };

    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool isBytesplit = false;

    LLAMA_EXPORT
    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
    inline constexpr bool isBytesplit<Bytesplit<TArrayExtents, TRecordDim, InnerMapping>> = true;
} // namespace llama::mapping

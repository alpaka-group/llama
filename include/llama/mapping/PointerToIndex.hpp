// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    struct PointerToRecordDim
    {
    };

    namespace internal
    {
        template<typename Replacement, typename T>
        struct ReplacePointerImpl
        {
            using type = std::conditional_t<std::is_same_v<T, PointerToRecordDim>, Replacement, T>;
        };

        template<typename Replacement, typename... Fields>
        struct ReplacePointerImpl<Replacement, Record<Fields...>>
        {
            using type = Record<
                Field<GetFieldTag<Fields>, typename ReplacePointerImpl<Replacement, GetFieldType<Fields>>::type>...>;
        };

        template<typename Replacement, typename T>
        using ReplacePointer = typename ReplacePointerImpl<Replacement, T>::type;
    } // namespace internal

    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
    struct PointerToIndex
        : private InnerMapping<TArrayExtents, internal::ReplacePointer<typename TArrayExtents::Index, TRecordDim>>
    {
        using Inner = InnerMapping<TArrayExtents, internal::ReplacePointer<typename TArrayExtents::Index, TRecordDim>>;
        using ArrayExtents = typename Inner::ArrayExtents;
        using ArrayIndex = typename Inner::ArrayIndex;
        using RecordDim = TRecordDim; // hide Inner::RecordDim
        using Inner::blobCount;
        using Inner::blobSize;
        using Inner::extents;
        using Inner::Inner;

        template<typename RecordCoord>
        LLAMA_FN_HOST_ACC_INLINE static constexpr auto isComputed(RecordCoord) -> bool
        {
            return std::is_same_v<GetType<RecordDim, RecordCoord>, PointerToRecordDim>;
        }

        template<std::size_t... RecordCoords, typename BlobArray>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            ArrayIndex ai,
            RecordCoord<RecordCoords...> rc,
            BlobArray& blobs) const
        {
            static_assert(isComputed(rc));
            using View = llama::View<PointerToIndex, std::decay_t<decltype(blobs[0])>, accessor::Default>;
            ArrayIndex& dstAi = mapToMemory(static_cast<const Inner&>(*this), ai, rc, blobs);
            auto& view = const_cast<View&>(reinterpret_cast<const View&>(*this));
            return PointerRef<View, RecordCoord<>>{dstAi, view};
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> rc = {})
            const -> NrAndOffset<typename ArrayExtents::value_type>
        {
            static_assert(!isComputed(rc));
            return Inner::blobNrAndOffset(ai, rc);
        }
    };

    /// Binds parameters to a \ref ChangeType mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<template<typename, typename> typename InnerMapping>
    struct BindPointerToIndex
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = PointerToIndex<ArrayExtents, RecordDim, InnerMapping>;
    };

    template<typename Mapping>
    inline constexpr bool isPointerToIndex = false;

    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
    inline constexpr bool isPointerToIndex<PointerToIndex<TArrayExtents, TRecordDim, InnerMapping>> = true;
} // namespace llama::mapping

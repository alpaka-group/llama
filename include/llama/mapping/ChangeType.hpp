// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"
#include "Projection.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename UserT, typename StoredT>
        struct ChangeTypeProjection
        {
            LLAMA_FN_HOST_ACC_INLINE static auto load(StoredT v) -> UserT
            {
                return static_cast<UserT>(v); // we could allow stronger casts here
            }

            LLAMA_FN_HOST_ACC_INLINE static auto store(UserT v) -> StoredT
            {
                return static_cast<StoredT>(v); // we could allow stronger casts here
            }
        };

        template<typename RecordDim>
        struct MakeProjectionPair
        {
            template<typename Key>
            static auto recordDimType()
            {
                if constexpr(isRecordCoord<Key>)
                    return mp_identity<GetType<RecordDim, Key>>{};
                else
                    return mp_identity<Key>{};
            }

            template<typename Pair, typename Key = mp_first<Pair>, typename StoredT = mp_second<Pair>>
            using fn = mp_list<Key, ChangeTypeProjection<typename decltype(recordDimType<Key>())::type, StoredT>>;
        };

        template<typename RecordDim, typename ReplacementMap>
        using MakeProjectionMap = mp_transform<MakeProjectionPair<RecordDim>::template fn, ReplacementMap>;
    } // namespace internal

    /// Mapping that changes the type in the record domain for a different one in storage. Conversions happen during
    /// load and store.
    /// @tparam ReplacementMap A type list of binary type lists (a map) specifiying which type or the type at a \ref
    /// RecordCoord (map key) to replace by which other type (mapped value).
    LLAMA_EXPORT
    template<
        typename ArrayExtents,
        typename RecordDim,
        template<typename, typename>
        typename InnerMapping,
        typename ReplacementMap>
    struct ChangeType
        : Projection<ArrayExtents, RecordDim, InnerMapping, internal::MakeProjectionMap<RecordDim, ReplacementMap>>
    {
    private:
        using Base = Projection<
            ArrayExtents,
            RecordDim,
            InnerMapping,
            internal::MakeProjectionMap<RecordDim, ReplacementMap>>;

    public:
        using Base::Base;
    };

    /// Binds parameters to a \ref ChangeType mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    LLAMA_EXPORT
    template<template<typename, typename> typename InnerMapping, typename ReplacementMap>
    struct BindChangeType
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = ChangeType<ArrayExtents, RecordDim, InnerMapping, ReplacementMap>;
    };

    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool isChangeType = false;

    LLAMA_EXPORT
    template<
        typename TArrayExtents,
        typename TRecordDim,
        template<typename, typename>
        typename InnerMapping,
        typename ReplacementMap>
    inline constexpr bool isChangeType<ChangeType<TArrayExtents, TRecordDim, InnerMapping, ReplacementMap>> = true;
} // namespace llama::mapping

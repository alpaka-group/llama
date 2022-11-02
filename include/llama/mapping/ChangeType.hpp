// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename ReplacementMap, typename Coord, typename UserT>
        auto replacedType()
        {
            using namespace boost::mp11;
            if constexpr(mp_map_contains<ReplacementMap, Coord>::value)
                return mp_identity<mp_second<mp_map_find<ReplacementMap, Coord>>>{};
            else if constexpr(mp_map_contains<ReplacementMap, UserT>::value)
                return mp_identity<mp_second<mp_map_find<ReplacementMap, UserT>>>{};
            else
                return mp_identity<UserT>{};
        }

        template<typename ReplacementMap, typename Coord, typename UserT>
        using ReplacedType = typename decltype(replacedType<ReplacementMap, Coord, UserT>())::type;

        template<typename ReplacementMap>
        struct MakeReplacer
        {
            template<typename Coord, typename UserT>
            using type = ReplacedType<ReplacementMap, Coord, UserT>;
        };

        template<typename RecordDim, typename ReplacementMap>
        using ReplaceTypesInRecordDim
            = TransformLeavesWithCoord<RecordDim, MakeReplacer<ReplacementMap>::template type>;

        template<typename UserT, typename StoredT>
        struct ChangeTypeReference : ProxyRefOpMixin<ChangeTypeReference<UserT, StoredT>, UserT>
        {
        private:
            StoredT& storageRef;

        public:
            using value_type = UserT;

            LLAMA_FN_HOST_ACC_INLINE constexpr explicit ChangeTypeReference(StoredT& storageRef)
                : storageRef{storageRef}
            {
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator UserT() const
            {
                return static_cast<UserT>(storageRef); // we could allow stronger casts here
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(UserT v) -> ChangeTypeReference&
            {
                storageRef = static_cast<StoredT>(v); // we could allow stronger casts here
                return *this;
            }
        };
    } // namespace internal

    /// Mapping that changes the type in the record domain for a different one in storage. Conversions happen during
    /// load and store.
    /// @tparam TReplacementMap A type list of binary type lists (a map) specifiying which type or the type at a \ref
    /// RecordCoord (map key) to replace by which other type (mapped value).
    template<
        typename TArrayExtents,
        typename TRecordDim,
        template<typename, typename>
        typename InnerMapping,
        typename TReplacementMap>
    struct ChangeType
        : private InnerMapping<TArrayExtents, internal::ReplaceTypesInRecordDim<TRecordDim, TReplacementMap>>
    {
        using Inner = InnerMapping<TArrayExtents, internal::ReplaceTypesInRecordDim<TRecordDim, TReplacementMap>>;
        using ReplacementMap = TReplacementMap;
        using ArrayExtents = typename Inner::ArrayExtents;
        using ArrayIndex = typename Inner::ArrayIndex;
        using RecordDim = TRecordDim; // hide Inner::RecordDim
        using Inner::blobCount;
        using Inner::blobSize;
        using Inner::extents;
        using Inner::Inner;

        template<typename RecordCoord>
        LLAMA_FN_HOST_ACC_INLINE static constexpr auto isComputed(RecordCoord)
        {
            using UserT = GetType<RecordDim, RecordCoord>;
            return boost::mp11::mp_map_contains<ReplacementMap, RecordCoord>::value
                || boost::mp11::mp_map_contains<ReplacementMap, UserT>::value;
        }

        // using Inner::blobNrAndOffset; // for all non-computed fields

        template<std::size_t... RecordCoords, typename BlobArray>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Inner::ArrayIndex ai,
            RecordCoord<RecordCoords...> rc,
            BlobArray& blobs) const
        {
            static_assert(isComputed(rc));
            using UserT = GetType<RecordDim, RecordCoord<RecordCoords...>>;
            using StoredT = internal::ReplacedType<ReplacementMap, RecordCoord<RecordCoords...>, UserT>;
            using QualifiedStoredT = CopyConst<BlobArray, StoredT>;
            const auto [nr, offset] = Inner::template blobNrAndOffset<RecordCoords...>(ai);
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return internal::ChangeTypeReference<UserT, QualifiedStoredT>{
                reinterpret_cast<QualifiedStoredT&>(blobs[nr][offset])};
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
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
    template<template<typename, typename> typename InnerMapping, typename ReplacementMap>
    struct BindChangeType
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = ChangeType<ArrayExtents, RecordDim, InnerMapping, ReplacementMap>;
    };

    template<typename Mapping>
    inline constexpr bool isChangeType = false;

    template<
        typename TArrayExtents,
        typename TRecordDim,
        template<typename, typename>
        typename InnerMapping,
        typename ReplacementMap>
    inline constexpr bool isChangeType<ChangeType<TArrayExtents, TRecordDim, InnerMapping, ReplacementMap>> = true;
} // namespace llama::mapping

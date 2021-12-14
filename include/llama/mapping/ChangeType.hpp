// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename ReplacementMap>
        struct MakeReplacer
        {
            static_assert(
                boost::mp11::mp_is_map<ReplacementMap>::value,
                "The Replacement map must be of the form: mp_list<mp_list<From, To>, ...>");

            template<typename UserT>
            using type = typename boost::mp11::mp_if<
                boost::mp11::mp_map_contains<ReplacementMap, UserT>,
                boost::mp11::mp_defer<boost::mp11::mp_second, boost::mp11::mp_map_find<ReplacementMap, UserT>>,
                boost::mp11::mp_identity<UserT>>::type;
        };

        template<typename RecordDim, typename ReplacementMap>
        using ReplaceType = TransformLeaves<RecordDim, MakeReplacer<ReplacementMap>::template type>;

        template<typename UserT, typename StoredT>
        struct ChangeTypeReference : ProxyRefOpMixin<ChangeTypeReference<UserT, StoredT>, UserT>
        {
        private:
            StoredT& storageRef;

        public:
            using value_type = UserT;

            LLAMA_FN_HOST_ACC_INLINE constexpr ChangeTypeReference(StoredT& storageRef) : storageRef{storageRef}
            {
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator UserT() const
            {
                return storageRef;
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(UserT v) -> ChangeTypeReference&
            {
                storageRef = v;
                return *this;
            }
        };
    } // namespace internal

    /// Mapping that changes the type in the record domain for a different one in storage. Conversions happen during
    /// load and store.
    /// /tparam ReplacementMap A type list of binary type lists (a map) specifiying which type to replace by which
    /// other type.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        template<typename, typename>
        typename InnerMapping,
        typename ReplacementMap>
    struct ChangeType : private InnerMapping<TArrayExtents, internal::ReplaceType<TRecordDim, ReplacementMap>>
    {
        using Inner = InnerMapping<TArrayExtents, internal::ReplaceType<TRecordDim, ReplacementMap>>;

        using ArrayExtents = typename Inner::ArrayExtents;
        using ArrayIndex = typename Inner::ArrayIndex;
        using RecordDim = TRecordDim; // hide Inner::RecordDim
        using Inner::blobCount;

        using Inner::blobSize;
        using Inner::extents;
        using Inner::Inner;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit ChangeType(TArrayExtents extents) : Inner(extents)
        {
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            using UserT = GetType<RecordDim, RecordCoord<RecordCoords...>>;
            return boost::mp11::mp_map_contains<ReplacementMap, UserT>::value;
        }

        using Inner::blobNrAndOffset; // for all non-computed fields

        template<std::size_t... RecordCoords, typename BlobArray>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Inner::ArrayIndex ai,
            RecordCoord<RecordCoords...>,
            BlobArray& blobs) const
        {
            using UserT = GetType<RecordDim, RecordCoord<RecordCoords...>>;
            using StoredT = boost::mp11::mp_second<boost::mp11::mp_map_find<ReplacementMap, UserT>>;
            using QualifiedStoredT = CopyConst<BlobArray, StoredT>;
            const auto [nr, offset] = Inner::template blobNrAndOffset<RecordCoords...>(ai);
            return internal::ChangeTypeReference<UserT, QualifiedStoredT>{
                reinterpret_cast<QualifiedStoredT&>(blobs[nr][offset])};
        }
    };

    template<template<typename, typename> typename InnerMapping, typename ReplacementMap>
    struct PreconfiguredChangeType
    {
        template<typename ArrayExtents, typename RecordDim>
        using type = ChangeType<ArrayExtents, RecordDim, InnerMapping, ReplacementMap>;
    };
} // namespace llama::mapping

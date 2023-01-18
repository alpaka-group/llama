// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../ProxyRefOpMixin.hpp"
#include "../View.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename F>
        struct UnaryFunctionTraits
        {
            static_assert(sizeof(F) == 0, "F is not an unary function");
        };

        template<typename Arg, typename Ret>
        struct UnaryFunctionTraits<Ret (*)(Arg)>
        {
            using ArgumentType = Arg;
            using ReturnType = Ret;
        };

        template<typename ProjectionMap, typename Coord, typename RecordDimType>
        auto projectionOrVoidImpl()
        {
            if constexpr(mp_map_contains<ProjectionMap, Coord>::value)
                return mp_identity<mp_second<mp_map_find<ProjectionMap, Coord>>>{};
            else if constexpr(mp_map_contains<ProjectionMap, RecordDimType>::value)
                return mp_identity<mp_second<mp_map_find<ProjectionMap, RecordDimType>>>{};
            else
                return mp_identity<void>{};
        }

        template<typename ProjectionMap, typename Coord, typename RecordDimType>
        using ProjectionOrVoid = typename decltype(projectionOrVoidImpl<ProjectionMap, Coord, RecordDimType>())::type;

        template<typename ProjectionMap>
        struct MakeReplacerProj
        {
            template<typename Coord, typename RecordDimType>
            static auto replacedTypeProj()
            {
                using Projection = ProjectionOrVoid<ProjectionMap, Coord, RecordDimType>;
                if constexpr(std::is_void_v<Projection>)
                    return mp_identity<RecordDimType>{};
                else
                {
                    using LoadFunc = UnaryFunctionTraits<decltype(&Projection::load)>;
                    using StoreFunc = UnaryFunctionTraits<decltype(&Projection::store)>;

                    static_assert(std::is_same_v<typename LoadFunc::ReturnType, RecordDimType>);
                    static_assert(std::is_same_v<typename StoreFunc::ArgumentType, RecordDimType>);
                    static_assert(std::is_same_v<typename LoadFunc::ArgumentType, typename StoreFunc::ReturnType>);

                    return mp_identity<typename StoreFunc::ReturnType>{};
                }
            }

            template<typename Coord, typename RecordDimType>
            using fn = typename decltype(replacedTypeProj<Coord, RecordDimType>())::type;
        };

        template<typename RecordDim, typename ProjectionMap>
        using ReplaceTypesByProjectionResults
            = TransformLeavesWithCoord<RecordDim, MakeReplacerProj<ProjectionMap>::template fn>;

        template<typename Reference, typename Projection>
        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
        struct ProjectionReference
            : ProxyRefOpMixin<
                  ProjectionReference<Reference, Projection>,
                  decltype(Projection::load(std::declval<Reference>()))>
        {
        private:
            Reference storageRef;

        public:
            using value_type = decltype(Projection::load(std::declval<Reference>()));

            LLAMA_FN_HOST_ACC_INLINE constexpr explicit ProjectionReference(Reference storageRef)
                : storageRef{storageRef}
            {
            }

            ProjectionReference(const ProjectionReference&) = default;

            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const ProjectionReference& other) -> ProjectionReference&
            {
                *this = static_cast<value_type>(other);
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator value_type() const
            {
                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                return Projection::load(storageRef);
                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(value_type v) -> ProjectionReference&
            {
                LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                storageRef = Projection::store(v);
                LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
                return *this;
            }
        };
    } // namespace internal

    /// Mapping that projects types in the record domain to different types. Projections are executed during load and
    /// store.
    /// @tparam TProjectionMap A type list of binary type lists (a map) specifing a projection (map value) for a type
    /// or the type at a \ref RecordCoord (map key). A projection is a type with two functions:
    /// struct Proj {
    ///   static auto load(auto&& fromMem);
    ///   static auto store(auto&& toMem);
    /// };
    template<
        typename TArrayExtents,
        typename TRecordDim,
        template<typename, typename>
        typename InnerMapping,
        typename TProjectionMap>
    struct Projection
        : private InnerMapping<TArrayExtents, internal::ReplaceTypesByProjectionResults<TRecordDim, TProjectionMap>>
    {
        using Inner
            = InnerMapping<TArrayExtents, internal::ReplaceTypesByProjectionResults<TRecordDim, TProjectionMap>>;
        using ProjectionMap = TProjectionMap;
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
            return !std::is_void_v<
                internal::ProjectionOrVoid<ProjectionMap, RecordCoord, GetType<RecordDim, RecordCoord>>>;
        }

        template<std::size_t... RecordCoords, typename BlobArray>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            typename Inner::ArrayIndex ai,
            RecordCoord<RecordCoords...> rc,
            BlobArray& blobs) const
        {
            static_assert(isComputed(rc));
            using RecordDimType = GetType<RecordDim, RecordCoord<RecordCoords...>>;
            using Reference = decltype(mapToMemory(static_cast<const Inner&>(*this), ai, rc, blobs));
            using Projection = internal::ProjectionOrVoid<ProjectionMap, RecordCoord<RecordCoords...>, RecordDimType>;
            static_assert(!std::is_void_v<Projection>);
            Reference r = mapToMemory(static_cast<const Inner&>(*this), ai, rc, blobs);

            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return internal::ProjectionReference<Reference, Projection>{r};
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

    /// Binds parameters to a \ref Projection mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<template<typename, typename> typename InnerMapping, typename ProjectionMap>
    struct BindProjection
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = Projection<ArrayExtents, RecordDim, InnerMapping, ProjectionMap>;
    };

    template<typename Mapping>
    inline constexpr bool isProjection = false;

    template<
        typename TArrayExtents,
        typename TRecordDim,
        template<typename, typename>
        typename InnerMapping,
        typename ReplacementMap>
    inline constexpr bool isProjection<Projection<TArrayExtents, TRecordDim, InnerMapping, ReplacementMap>> = true;
} // namespace llama::mapping

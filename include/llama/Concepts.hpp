// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Array.hpp"
#include "Core.hpp"
#include "RecordCoord.hpp"

#include <type_traits>

#if __has_include(<concepts>)
#    include <concepts>
#endif
namespace llama
{
#ifdef __cpp_lib_concepts
    LLAMA_EXPORT
    template<auto I>
    concept isConstexpr = requires { std::integral_constant<decltype(I), I>{}; };

    LLAMA_EXPORT
    template<typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayExtents;
        typename M::RecordDim;
        {
            m.extents()
        } -> std::same_as<typename M::ArrayExtents>;
        {
            +M::blobCount
        } -> std::same_as<std::size_t>;
        requires isConstexpr<M::blobCount>;
        {
            m.blobSize(typename M::ArrayExtents::value_type{})
        } -> std::same_as<typename M::ArrayExtents::value_type>;
    };

    LLAMA_EXPORT
    template<typename M, typename RC>
    concept PhysicalField = requires(M m, typename M::ArrayExtents::Index ai) {
        {
            m.blobNrAndOffset(ai, RC{})
        } -> std::same_as<NrAndOffset<typename M::ArrayExtents::value_type>>;
    };

    template<typename M>
    struct MakeIsPhysical
    {
        template<typename RC>
        using fn = mp_bool<PhysicalField<M, RC>>;
    };

    LLAMA_EXPORT
    template<typename M>
    inline constexpr bool allFieldsArePhysical
        = mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsPhysical<M>::template fn>::value;

    LLAMA_EXPORT
    template<typename M>
    concept PhysicalMapping = Mapping<M> && allFieldsArePhysical<M>;

    LLAMA_EXPORT
    template<typename R>
    concept LValueReference = std::is_lvalue_reference_v<R>;

    LLAMA_EXPORT
    template<typename R>
    concept AdlTwoStepSwappable = requires(R a, R b) { swap(a, b); } || requires(R a, R b) { std::swap(a, b); };

    LLAMA_EXPORT
    template<typename R>
    concept ProxyReference = std::is_copy_constructible_v<R> && std::is_copy_assignable_v<R> && requires(R r) {
        typename R::value_type;
        {
            static_cast<typename R::value_type>(r)
        } -> std::same_as<typename R::value_type>;
        {
            r = std::declval<typename R::value_type>()
        } -> std::same_as<R&>;
    } && AdlTwoStepSwappable<R>;

    LLAMA_EXPORT
    template<typename R>
    concept AnyReference = LValueReference<R> || ProxyReference<R>;

    LLAMA_EXPORT
    template<typename R, typename T>
    concept AnyReferenceTo = (LValueReference<R> && std::is_same_v<std::remove_cvref_t<R>, T>)
        || (ProxyReference<R> && std::is_same_v<typename R::value_type, T>);

    LLAMA_EXPORT
    template<typename M, typename RC>
    concept ComputedField
        = M::isComputed(RC{}) && requires(M m, typename M::ArrayExtents::Index ai, std::byte** blobs) {
              {
                  m.compute(ai, RC{}, blobs)
              } -> AnyReferenceTo<GetType<typename M::RecordDim, RC>>;
          };

    template<typename M>
    struct MakeIsComputed
    {
        template<typename RC>
        using fn = mp_bool<ComputedField<M, RC>>;
    };

    LLAMA_EXPORT
    template<typename M>
    inline constexpr bool allFieldsAreComputed
        = mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsComputed<M>::template fn>::value;

    LLAMA_EXPORT
    template<typename M>
    concept FullyComputedMapping = Mapping<M> && allFieldsAreComputed<M>;

    LLAMA_EXPORT
    template<
        typename M,
        typename LeafCoords = LeafRecordCoords<typename M::RecordDim>,
        std::size_t PhysicalCount = mp_count_if<LeafCoords, MakeIsPhysical<M>::template fn>::value,
        std::size_t ComputedCount = mp_count_if<LeafCoords, MakeIsComputed<M>::template fn>::value>
    inline constexpr bool allFieldsArePhysicalOrComputed
        = (PhysicalCount + ComputedCount) >= mp_size<LeafCoords>::value && PhysicalCount > 0
        && ComputedCount > 0; // == instead of >= would be better, but it's not easy to count correctly,
                              // because we cannot check whether the call to blobNrOrOffset()
                              // or compute() is actually valid

    LLAMA_EXPORT
    template<typename M>
    concept PartiallyComputedMapping = Mapping<M> && allFieldsArePhysicalOrComputed<M>;

    /// Additional semantic requirement: &b[i] + j == &b[i + j] for any integral i and j in range of the blob
    LLAMA_EXPORT
    template<typename B>
    concept Blob = requires(B b, std::size_t i) {
        // according to http://eel.is/c++draft/intro.object#3 only std::byte and unsigned char can
        // provide storage for
        // other types
        requires std::is_lvalue_reference_v<decltype(b[i])>;
        requires std::same_as<std::remove_cvref_t<decltype(b[i])>, std::byte>
            || std::same_as<std::remove_cvref_t<decltype(b[i])>, unsigned char>;
    };

    LLAMA_EXPORT
    template<typename BA>
    concept BlobAllocator = requires(BA ba, std::size_t size) {
        {
            ba(std::integral_constant<std::size_t, 16>{}, size)
        } -> Blob;
    };

    LLAMA_EXPORT
    template<typename V>
    concept AnyView = requires(V v, const V cv) {
        typename V::Mapping;
        typename V::BlobType;
        typename V::ArrayExtents;
        typename V::ArrayIndex;
        typename V::RecordDim;
        typename V::Accessor;

        typename V::iterator;
        typename V::const_iterator;

        {
            v.mapping()
        } -> std::same_as<typename V::Mapping&>;

        {
            cv.mapping()
        } -> std::same_as<const typename V::Mapping&>;

        {
            v.accessor()
        } -> std::same_as<typename V::Accessor&>;

        {
            cv.accessor()
        } -> std::same_as<const typename V::Accessor&>;

        {
            cv.extents()
        } -> std::same_as<typename V::ArrayExtents>;

        {
            v.begin()
        } -> std::same_as<typename V::iterator>;

        {
            cv.begin()
        } -> std::same_as<typename V::const_iterator>;

        {
            v.end()
        } -> std::same_as<typename V::iterator>;

        {
            cv.end()
        } -> std::same_as<typename V::const_iterator>;

        {
            v.blobs()
        } -> std::same_as<Array<typename V::BlobType, V::Mapping::blobCount>&>;
        {
            cv.blobs()
        } -> std::same_as<const Array<typename V::BlobType, V::Mapping::blobCount>&>;
    };
#endif

    namespace internal
    {
        template<typename R, typename = void>
        struct IsProxyReferenceImpl : std::false_type
        {
        };

        template<typename R>
        struct IsProxyReferenceImpl<
            R,
            std::void_t<
                typename R::value_type,
                decltype(static_cast<typename R::value_type>(std::declval<R&>())),
                decltype(std::declval<R&>() = std::declval<typename R::value_type>())>>
            : std::bool_constant<std::is_copy_constructible_v<R> && std::is_copy_assignable_v<R>>
        {
        };
    } // namespace internal

    LLAMA_EXPORT
    template<typename R>
#ifdef __cpp_lib_concepts
    inline constexpr bool isProxyReference = ProxyReference<R>;
#else
    inline constexpr bool isProxyReference = internal::IsProxyReferenceImpl<R>::value;
#endif
} // namespace llama

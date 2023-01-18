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
    // clang-format off
    template <typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayExtents;
        typename M::ArrayIndex;
        typename M::RecordDim;
        { m.extents() } -> std::same_as<typename M::ArrayExtents>;
        { +M::blobCount } -> std::same_as<std::size_t>;
        std::integral_constant<std::size_t, M::blobCount>{}; // validates constexpr-ness
        { m.blobSize(typename M::ArrayExtents::value_type{}) } -> std::same_as<typename M::ArrayExtents::value_type>;
    };

    template <typename M, typename RC>
    concept PhysicalField = requires(M m, typename M::ArrayIndex ai) {
        { m.blobNrAndOffset(ai, RC{}) } -> std::same_as<NrAndOffset<typename M::ArrayExtents::value_type>>;
    };

    template<typename M>
    struct MakeIsPhysical
    {
        template<typename RC>
        using fn = mp_bool<PhysicalField<M, RC>>;
    };

    template<typename M>
    inline constexpr bool allFieldsArePhysical
        = mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsPhysical<M>::template fn>::value;

    template <typename M>
    concept PhysicalMapping = Mapping<M> && allFieldsArePhysical<M>;

    template <typename R>
    concept LValueReference = std::is_lvalue_reference_v<R>;

    template<typename R>
    concept AdlTwoStepSwappable = requires(R a, R b) { swap(a, b); } || requires(R a, R b) { std::swap(a, b); };

    template <typename R>
    concept ProxyReference = std::is_copy_constructible_v<R> && std::is_copy_assignable_v<R> && requires(R r) {
        typename R::value_type;
        { static_cast<typename R::value_type>(r) } -> std::same_as<typename R::value_type>;
        { r = std::declval<typename R::value_type>() } -> std::same_as<R&>;
    } && AdlTwoStepSwappable<R>;

    template <typename R>
    concept AnyReference = LValueReference<R> || ProxyReference<R>;

    template <typename R, typename T>
    concept AnyReferenceTo = (LValueReference<R> && std::is_same_v<std::remove_cvref_t<R>, T>) || (ProxyReference<R> && std::is_same_v<typename R::value_type, T>);

    template <typename M, typename RC>
    concept ComputedField = M::isComputed(RC{}) && requires(M m, typename M::ArrayIndex ai, Array<Array<std::byte, 1>, 1> blobs) {
        { m.compute(ai, RC{}, blobs) } -> AnyReferenceTo<GetType<typename M::RecordDim, RC>>;
    };

    template<typename M>
    struct MakeIsComputed
    {
        template<typename RC>
        using fn = mp_bool<ComputedField<M, RC>>;
    };

    template<typename M>
    inline constexpr bool allFieldsAreComputed
        = mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsComputed<M>::template fn>::value;

    template <typename M>
    concept FullyComputedMapping = Mapping<M> && allFieldsAreComputed<M>;

    template<
        typename M,
        typename LeafCoords = LeafRecordCoords<typename M::RecordDim>,
        std::size_t PhysicalCount = mp_count_if<LeafCoords, MakeIsPhysical<M>::template fn>::value,
        std::size_t ComputedCount = mp_count_if<LeafCoords, MakeIsComputed<M>::template fn>::value>
    inline constexpr bool allFieldsArePhysicalOrComputed
        = (PhysicalCount + ComputedCount) >= mp_size<LeafCoords>::value&& PhysicalCount > 0
        && ComputedCount > 0; // == instead of >= would be better, but it's not easy to count correctly,
                              // because we cannot check whether the call to blobNrOrOffset()
                              // or compute() is actually valid

    template <typename M>
    concept PartiallyComputedMapping = Mapping<M> && allFieldsArePhysicalOrComputed<M>;

    // according to http://eel.is/c++draft/intro.object#3 only std::byte and unsigned char can provide storage for
    // other types
    template<typename B>
    concept Blob = requires(B b, std::size_t i) {
        requires std::is_same_v<std::remove_cvref_t<decltype(b[i])>, std::byte> ||
            std::is_same_v<std::remove_cvref_t<decltype(b[i])>, unsigned char>;
    };

    template <typename BA>
    concept BlobAllocator = requires(BA ba, std::integral_constant<std::size_t, 16> alignment, std::size_t size) {
        { ba(alignment, size) } -> Blob;
    };
    // clang-format on
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

    template<typename R>
#ifdef __cpp_lib_concepts
    inline constexpr bool isProxyReference = ProxyReference<R>;
#else
    inline constexpr bool isProxyReference = internal::IsProxyReferenceImpl<R>::value;
#endif
} // namespace llama

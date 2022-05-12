#pragma once

#include "Array.hpp"
#include "Core.hpp"
#include "RecordCoord.hpp"

#include <type_traits>

#if __has_include(<concepts>)
#    include <concepts>
#endif
#ifdef __cpp_lib_concepts
namespace llama
{
    // clang-format off
    template <typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayExtents;
        typename M::ArrayIndex;
        typename M::RecordDim;
        { m.extents() } -> std::same_as<typename M::ArrayExtents>;
        { +M::blobCount } -> std::same_as<std::size_t>;
        Array<int, M::blobCount>{}; // validates constexpr-ness
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
        using type = boost::mp11::mp_bool<PhysicalField<M, RC>>;
    };

    template<typename M>
    inline constexpr bool AllFieldsArePhysical
        = boost::mp11::mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsPhysical<M>::template type>::value;

    template <typename M>
    concept PhysicalMapping = Mapping<M> && AllFieldsArePhysical<M>;

    template <typename R>
    concept LValueReference = std::is_lvalue_reference_v<R>;

    template <typename R>
    concept ProxyReference = requires(R r) {
        typename R::value_type;
        { static_cast<typename R::value_type>(r) } -> std::same_as<typename R::value_type>;
        { r = typename R::value_type{} } -> std::same_as<R&>;
    };

    template <typename R>
    concept AnyReference = LValueReference<R> || ProxyReference<R>;

    template <typename M, typename RC>
    concept ComputedField = M::isComputed(RC{}) && requires(M m, typename M::ArrayIndex ai, Array<Array<std::byte, 1>, 1> blobs) {
        { m.compute(ai, RC{}, blobs) } -> AnyReference;
    };

    template<typename M>
    struct MakeIsComputed
    {
        template<typename RC>
        using type = boost::mp11::mp_bool<ComputedField<M, RC>>;
    };

    template<typename M>
    inline constexpr bool AllFieldsAreComputed
        = boost::mp11::mp_all_of<LeafRecordCoords<typename M::RecordDim>, MakeIsComputed<M>::template type>::value;

    template <typename M>
    concept FullyComputedMapping = Mapping<M> && AllFieldsAreComputed<M>;

    template<
        typename M,
        typename LeafCoords = LeafRecordCoords<typename M::RecordDim>,
        std::size_t PhysicalCount = boost::mp11::mp_count_if<LeafCoords, MakeIsPhysical<M>::template type>::value,
        std::size_t ComputedCount = boost::mp11::mp_count_if<LeafCoords, MakeIsComputed<M>::template type>::value>
    inline constexpr bool AllFieldsArePhysicalOrComputed
        = (PhysicalCount + ComputedCount) >= boost::mp11::mp_size<LeafCoords>::value&& PhysicalCount > 0
        && ComputedCount > 0; // == instead of >= would be better, but it's not easy to count correctly,
                              // because we cannot check whether the call to blobNrOrOffset()
                              // or compute() is actually valid

    template <typename M>
    concept PartiallyComputedMapping = Mapping<M> && AllFieldsArePhysicalOrComputed<M>;

    template<typename B>
    concept Blob = requires(B b, std::size_t i) {
        // according to http://eel.is/c++draft/intro.object#3 only std::byte and unsigned char can provide storage for
        // other types
        std::is_same_v<decltype(b[i]), std::byte&> || std::is_same_v<decltype(b[i]), unsigned char&>;
    };

    template <typename BA>
    concept BlobAllocator = requires(BA ba, std::integral_constant<std::size_t, 16> alignment, std::size_t size) {
        { ba(alignment, size) } -> Blob;
    };
    // clang-format on
} // namespace llama

#endif

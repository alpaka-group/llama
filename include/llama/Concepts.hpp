#pragma once

#ifdef __cpp_concepts

#    include "Core.hpp"

#    include <concepts>
#    include <type_traits>

namespace llama
{
    // clang-format off
    template <typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayDomain;
        typename M::DatumDomain;
        { M::blobCount } -> std::convertible_to<std::size_t>; // TODO: check that blobCount is constexpr
        { m.getBlobSize(std::size_t{}) } -> std::convertible_to<std::size_t>;
        { m.getBlobNrAndOffset(typename M::ArrayDomain{}) } -> std::same_as<NrAndOffset>;
    };
    // clang-format on

    template <typename B>
    concept Blob = requires(B b, std::size_t i)
    {
        // according to http://eel.is/c++draft/intro.object#3 only std::byte and unsigned char can provide storage for
        // other types
        std::is_same_v<decltype(b[i]), std::byte&> || std::is_same_v<decltype(b[i]), unsigned char&>;
    };

    // clang-format off
    template <typename BA>
    concept BlobAllocator = requires(BA ba, std::size_t i) {
        { ba(i) } -> Blob;
    };
    // clang-format on
} // namespace llama

#endif

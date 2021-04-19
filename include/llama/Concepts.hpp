#pragma once

#ifdef __cpp_concepts

#    include "Array.hpp"
#    include "Core.hpp"

#    include <concepts>
#    include <type_traits>

namespace llama
{
    // clang-format off
    template <typename M>
    concept Mapping = requires(M m) {
        typename M::ArrayDims;
        typename M::RecordDim;
        { M::blobCount } -> std::convertible_to<std::size_t>;
        Array<int, M::blobCount>{}; // validates constexpr-ness
        { m.blobSize(std::size_t{}) } -> std::same_as<std::size_t>;
        { m.blobNrAndOffset(typename M::ArrayDims{}) } -> std::same_as<NrAndOffset>;
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

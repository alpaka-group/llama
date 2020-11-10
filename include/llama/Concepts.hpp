#pragma once

#ifdef __cpp_concepts

#    include "Core.hpp"

#    include <concepts>

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
} // namespace llama

#endif

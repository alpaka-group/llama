#pragma once

#ifdef __cpp_concepts

#include "Types.hpp"

#include <concepts>

namespace llama
{
    // clang-format off
    template <typename M>
    concept Mapping = requires(M m) {
        typename M::UserDomain;
        typename M::DatumDomain;
        { M::blobCount } -> std::convertible_to<std::size_t>; // TODO: check that blobCount is constexpr
        { m.getBlobSize(std::size_t{}) } -> std::convertible_to<std::size_t>;
        { m.getBlobNrAndOffset(typename M::UserDomain{}) } -> std::same_as<NrAndOffset>;
    };
    // clang-format on
}

#endif

#pragma once

#include "Types.hpp"

#include <concepts>

namespace llama
{
    template <typename M>
    concept Mapping = requires(M m) {
        typename M::UserDomain;
        typename M::DatumDomain;
        { m.blobCount } -> std::convertible_to<std::size_t>;
        { m.getBlobSize(std::size_t{}) } -> std::integral;
        { m.getBlobNrAndOffset(typename M::UserDomain{}) } -> std::same_as<NrAndOffset>;
    };
}

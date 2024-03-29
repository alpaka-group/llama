// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    /// Meta mapping permuting the array indices before forwarding to another mapping. The array extents are not
    /// changed.
    /// @tparam Permutation The pack of integrals describing the permutation of the array indices. The inner mapping
    /// will be called with an ArrayIndex{ai[Permutation]...}.
    LLAMA_EXPORT
    template<typename Mapping, std::size_t... Permutation>
    struct PermuteArrayIndex : Mapping
    {
    private:
        using size_type = typename Mapping::ArrayExtents::value_type;
        using ArrayIndex = typename Mapping::ArrayExtents::Index;

    public:
        using Inner = Mapping;

        constexpr PermuteArrayIndex() = default;

        LLAMA_FN_HOST_ACC_INLINE
        explicit PermuteArrayIndex(Mapping mapping) : Mapping(std::move(mapping))
        {
        }

        template<typename... Args>
        LLAMA_FN_HOST_ACC_INLINE explicit PermuteArrayIndex(Args&&... innerArgs)
            : Mapping(std::forward<Args>(innerArgs)...)
        {
        }

        static_assert(
            sizeof...(Permutation) == ArrayIndex::rank,
            "The number of integral arguments to PermuteArrayIndex must be the same as ArrayExtents::rank");

        template<std::size_t... RCs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RCs...> rc = {}) const
            -> NrAndOffset<size_type>
        {
            return Inner::blobNrAndOffset(ArrayIndex{ai[Permutation]...}, rc);
        }

        template<std::size_t... RCs, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto compute(ArrayIndex ai, RecordCoord<RCs...> rc, Blobs& blobs) const
            -> decltype(auto)
        {
            return Inner::compute(ArrayIndex{ai[Permutation]...}, rc, blobs);
        }
    };

    LLAMA_EXPORT
    template<typename Mapping>
    PermuteArrayIndex(Mapping) -> PermuteArrayIndex<Mapping>;

    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool isPermuteArrayIndex = false;

    LLAMA_EXPORT
    template<typename Mapping, std::size_t... Permutation>
    inline constexpr bool isPermuteArrayIndex<PermuteArrayIndex<Mapping, Permutation...>> = true;
} // namespace llama::mapping

#pragma once

#include "Common.hpp"

#include <atomic>
#include <iostream>
#include <string>
#include <unordered_map>

namespace llama::mapping
{
    /// Forwards all calls to the inner mapping. Traces all accesses made through this mapping and prints a summary on
    /// destruction.
    /// \tparam Mapping The type of the inner mapping.
    template<typename Mapping>
    struct Trace
    {
        using ArrayExtents = typename Mapping::ArrayExtents;
        using ArrayIndex = typename Mapping::ArrayIndex;
        using RecordDim = typename Mapping::RecordDim;
        static constexpr std::size_t blobCount = Mapping::blobCount;

        constexpr Trace() = default;

        LLAMA_FN_HOST_ACC_INLINE
        explicit Trace(Mapping mapping, bool printOnDestruction = true)
            : mapping(mapping)
            , printOnDestruction(printOnDestruction)
        {
            forEachLeafCoord<RecordDim>([&](auto rc) { fieldHits[recordCoordTags<RecordDim>(rc)] = 0; });
        }

        Trace(const Trace&) = delete;
        auto operator=(const Trace&) -> Trace& = delete;

        Trace(Trace&&) noexcept = default;
        auto operator=(Trace&&) noexcept -> Trace& = default;

        ~Trace()
        {
            if(printOnDestruction && !fieldHits.empty())
                print();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return mapping.extents();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return mapping.blobSize(i);
        }

        template<std::size_t... RecordCoords, std::size_t N = 0>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(
            ArrayIndex ai,
            Array<std::size_t, N> dynamicArrayExtents = {},
            RecordCoord<RecordCoords...> rc = {}) const -> NrAndOffset
        {
            const static auto name = recordCoordTags<RecordDim>(RecordCoord<RecordCoords...>{});
            fieldHits.at(name)++;

            LLAMA_FORCE_INLINE_RECURSIVE return mapping.blobNrAndOffset(ai, dynamicArrayExtents, rc);
        }

        void print() const
        {
            std::cout << "Trace mapping, number of accesses:\n";
            for(const auto& [k, v] : fieldHits)
                std::cout << '\t' << k << ":\t" << v << '\n';
        }

        Mapping mapping;
        mutable std::unordered_map<std::string, std::atomic<std::size_t>> fieldHits;
        bool printOnDestruction;
    };
} // namespace llama::mapping

#pragma once

#include "Common.hpp"

#include <array>
#include <atomic>
#include <iostream>

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
            , fieldHits{}
            , printOnDestruction(printOnDestruction)
        {
        }

        Trace(const Trace&) = delete;
        auto operator=(const Trace&) -> Trace& = delete;

        Trace(Trace&& other) noexcept : mapping(std::move(other.mapping)), printOnDestruction(other.printOnDestruction)
        {
            for(std::size_t i = 0; i < fieldHits.size(); i++)
                fieldHits[i] = other.fieldHits[i].load();
            other.printOnDestruction = false;
        }

        auto operator=(Trace&& other) noexcept -> Trace&
        {
            mapping = std::move(other.mapping);
            printOnDestruction = other.printOnDestruction;
            for(std::size_t i = 0; i < fieldHits.size(); i++)
                fieldHits[i] = other.fieldHits[i].load();
            other.printOnDestruction = false;
        }

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

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> rc = {}) const
            -> NrAndOffset
        {
            fieldHits[flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>]++;
            LLAMA_FORCE_INLINE_RECURSIVE return mapping.blobNrAndOffset(ai, rc);
        }

        void print() const
        {
            std::cout << "Trace mapping, number of accesses:\n";
            forEachLeafCoord<RecordDim>(
                [this](auto rc)
                {
                    std::cout << '\t' << recordCoordTags<RecordDim>(rc) << ":\t"
                              << fieldHits[flatRecordCoord<RecordDim, decltype(rc)>] << '\n';
                });
        }

        Mapping mapping;
        mutable std::array<std::atomic<std::size_t>, flatFieldCount<RecordDim>> fieldHits;
        bool printOnDestruction;
    };
} // namespace llama::mapping

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
    struct Trace : Mapping
    {
        using RecordDim = typename Mapping::RecordDim;

        constexpr Trace() = default;

        LLAMA_FN_HOST_ACC_INLINE
        explicit Trace(Mapping mapping, bool printOnDestruction = true)
            : Mapping(mapping)
            , fieldHits{}
            , printOnDestruction(printOnDestruction)
        {
        }

        Trace(const Trace&) = delete;
        auto operator=(const Trace&) -> Trace& = delete;

        Trace(Trace&& other) noexcept
            : Mapping(std::move(static_cast<Mapping&>(other)))
            , printOnDestruction(other.printOnDestruction)
        {
            for(std::size_t i = 0; i < fieldHits.size(); i++)
                fieldHits[i] = other.fieldHits[i].load();
            other.printOnDestruction = false;
        }

        auto operator=(Trace&& other) noexcept -> Trace&
        {
            static_cast<Mapping&>(*this) = std::move(static_cast<Mapping&>(other));
            printOnDestruction = other.printOnDestruction;
            for(std::size_t i = 0; i < fieldHits.size(); i++)
                fieldHits[i] = other.fieldHits[i].load();
            other.printOnDestruction = false;
            return *this;
        }

        ~Trace()
        {
            if(printOnDestruction && !fieldHits.empty())
                print();
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(
            typename Mapping::ArrayIndex ai,
            RecordCoord<RecordCoords...> rc = {}) const -> NrAndOffset
        {
            ++fieldHits[flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>];
            return Mapping::blobNrAndOffset(ai, rc);
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

        mutable std::array<std::atomic<std::size_t>, flatFieldCount<RecordDim>> fieldHits;
        bool printOnDestruction;
    };
} // namespace llama::mapping

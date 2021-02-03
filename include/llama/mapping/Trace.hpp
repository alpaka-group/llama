#pragma once

#include "Common.hpp"

#include <atomic>
#include <boost/core/demangle.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace llama::mapping
{
    namespace internal
    {
        template <typename DatumDomain, std::size_t... Coords>
        auto coordName(DatumCoord<Coords...>) -> std::string
        {
            using Tags = GetTags<DatumDomain, DatumCoord<Coords...>>;

            std::string r;
            boost::mp11::mp_for_each<Tags>([&](auto tag) {
                if (!r.empty())
                    r += '.';
                r += structName(tag);
            });
            return r;
        }
    } // namespace internal

    /// Forwards all calls to the inner mapping. Traces all accesses made
    /// through this mapping and prints a summary on destruction.
    /// \tparam Mapping The type of the inner mapping.
    template <typename Mapping>
    struct Trace
    {
        using ArrayDomain = typename Mapping::ArrayDomain;
        using DatumDomain = typename Mapping::DatumDomain;
        using OriginalDatumDomain = typename Mapping::OriginalDatumDomain;
        static constexpr std::size_t blobCount = Mapping::blobCount;

        constexpr Trace() = default;

        LLAMA_FN_HOST_ACC_INLINE
        Trace(Mapping mapping) : mapping(mapping)
        {
            forEachLeave<DatumDomain>([&](auto coord) { datumHits[internal::coordName<DatumDomain>(coord)] = 0; });
        }

        Trace(const Trace&) = delete;
        auto operator=(const Trace&) -> Trace& = delete;

        Trace(Trace&&) noexcept = default;
        auto operator=(Trace&&) noexcept -> Trace& = default;

        ~Trace()
        {
            if (!datumHits.empty())
            {
                std::cout << "Trace mapping, number of accesses:\n";
                for (const auto& [k, v] : datumHits)
                    std::cout << '\t' << k << ":\t" << v << '\n';
            }
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobSize(std::size_t i) const -> std::size_t
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return mapping.getBlobSize(i);
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            const static auto name = internal::coordName<DatumDomain>(DatumCoord<DatumDomainCoord...>{});
            datumHits.at(name)++;

            LLAMA_FORCE_INLINE_RECURSIVE return mapping.template getBlobNrAndOffset<DatumDomainCoord...>(coord);
        }

        Mapping mapping;
        mutable std::unordered_map<std::string, std::atomic<std::size_t>> datumHits;
    };
} // namespace llama::mapping

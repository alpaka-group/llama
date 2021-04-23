#pragma once

#include "Common.hpp"

#include <array>
#include <atomic>
#include <sstream>
#include <vector>

namespace llama::mapping
{
    /// Forwards all calls to the inner mapping. Counts all accesses made to all bytes, allowing to extract a heatmap.
    /// \tparam Mapping The type of the inner mapping.
    template <typename Mapping, typename CountType = std::size_t>
    struct Heatmap
    {
        using ArrayDims = typename Mapping::ArrayDims;
        using RecordDim = typename Mapping::RecordDim;
        static constexpr std::size_t blobCount = Mapping::blobCount;

        constexpr Heatmap() = default;

        LLAMA_FN_HOST_ACC_INLINE
        Heatmap(Mapping mapping) : mapping(mapping)
        {
            for (auto i = 0; i < blobCount; i++)
                byteHits[i] = std::vector<std::atomic<CountType>>(blobSize(i));
        }

        Heatmap(const Heatmap&) = delete;
        auto operator=(const Heatmap&) -> Heatmap& = delete;

        Heatmap(Heatmap&&) noexcept = default;
        auto operator=(Heatmap&&) noexcept -> Heatmap& = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return mapping.arrayDims();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize(std::size_t i) const -> std::size_t
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return mapping.blobSize(i);
        }

        template <std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayDims coord) const -> NrAndOffset
        {
            const auto nao = mapping.template blobNrAndOffset<RecordCoords...>(coord);
            for (auto i = 0; i < sizeof(GetType<RecordDim, RecordCoord<RecordCoords...>>); i++)
                byteHits[nao.nr][nao.offset + i]++;
            return nao;
        }

        // gnuplot with:
        // set view map
        // set yrange [] reverse
        // splot "./file.dat" matrix with image
        auto toGnuplotDatFile() const -> std::string
        {
            std::stringstream f;
            for (auto i = 0; i < blobCount; i++)
            {
                std::size_t byteCount = 0;
                for (const auto& hits : byteHits[i])
                    f << hits << ((++byteCount % 64 == 0) ? '\n' : ' ');
                while (byteCount++ % 64 != 0)
                    f << "0 ";
                f << '\n';
            }
            return f.str();
        }

        Mapping mapping;
        mutable std::array<std::vector<std::atomic<CountType>>, blobCount> byteHits;
    };
} // namespace llama::mapping

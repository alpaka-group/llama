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
    template<typename Mapping, typename CountType = std::size_t>
    struct Heatmap : Mapping
    {
        constexpr Heatmap() = default;

        LLAMA_FN_HOST_ACC_INLINE
        explicit Heatmap(Mapping mapping) : Mapping(mapping)
        {
            for(std::size_t i = 0; i < Mapping::blobCount; i++)
                byteHits[i] = std::vector<std::atomic<CountType>>(Mapping::blobSize(i));
        }

        Heatmap(const Heatmap&) = delete;
        auto operator=(const Heatmap&) -> Heatmap& = delete;

        Heatmap(Heatmap&&) noexcept = default;
        auto operator=(Heatmap&&) noexcept -> Heatmap& = default;

        ~Heatmap() = default;

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(
            typename Mapping::ArrayIndex ai,
            RecordCoord<RecordCoords...> rc = {}) const -> NrAndOffset<typename Mapping::ArrayExtents::value_type>
        {
            const auto nao = Mapping::blobNrAndOffset(ai, rc);
            using Type = GetType<typename Mapping::RecordDim, RecordCoord<RecordCoords...>>;
            for(std::size_t i = 0; i < sizeof(Type); i++)
                ++byteHits[nao.nr][nao.offset + i];
            return nao;
        }

        auto toGnuplotScript(std::size_t wrapAfterBytes = 64) const -> std::string
        {
            std::stringstream f;
            f << "#!/usr/bin/gnuplot -p\n$data << EOD\n";
            for(std::size_t i = 0; i < Mapping::blobCount; i++)
            {
                std::size_t byteCount = 0;
                for(const auto& hits : byteHits[i])
                    f << hits << ((++byteCount % wrapAfterBytes == 0) ? '\n' : ' ');
                while(byteCount++ % wrapAfterBytes != 0)
                    f << "0 ";
                f << '\n';
            }
            f << R"(EOD
set view map
set xtics format ""
set x2tics autofreq 8
set yrange [] reverse
set link x2; set link y2
set ylabel "Cacheline"
set x2label "Byte"
plot $data matrix with image pixels axes x2y1
)";
            return f.str();
        }

        mutable std::array<std::vector<std::atomic<CountType>>, Mapping::blobCount> byteHits;
    };
} // namespace llama::mapping

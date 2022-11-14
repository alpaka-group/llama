#pragma once

#include "Common.hpp"

#include <array>
#include <atomic>
#include <sstream>
#include <vector>

namespace llama::mapping
{
    /// Forwards all calls to the inner mapping. Counts all accesses made to blocks inside the blobs, allowing to
    /// extract a heatmap.
    /// @tparam Mapping The type of the inner mapping.
    /// @tparam Granularity The granularity in bytes on which to could accesses. A value of 1 counts every byte.
    /// individually. A value of e.g. 64, counts accesses per 64 byte block.
    /// @tparam TCountType Data type used to count the number of accesses. Atomic increments must be supported for this
    /// type.
    template<
        typename Mapping,
        typename Mapping::ArrayExtents::value_type Granularity = 1,
        typename TCountType = std::size_t>
    struct Heatmap : private Mapping
    {
    private:
        using size_type = typename Mapping::ArrayExtents::value_type;

    public:
        using Inner = Mapping;
        inline static constexpr std::size_t granularity = Granularity;
        using CountType = TCountType;
        using typename Mapping::ArrayExtents;
        using typename Mapping::ArrayIndex;
        using typename Mapping::RecordDim;

        // We duplicate every blob of the inner mapping with a shadow blob, where we count the accesses
        inline static constexpr std::size_t blobCount = Mapping::blobCount * 2;

        constexpr Heatmap() = default;

        LLAMA_FN_HOST_ACC_INLINE
        explicit Heatmap(Mapping mapping) : Mapping(std::move(mapping))
        {
        }

        template<typename... Args>
        LLAMA_FN_HOST_ACC_INLINE explicit Heatmap(Args&&... innerArgs) : Mapping(std::forward<Args>(innerArgs)...)
        {
        }

#if defined(__cpp_lib_concepts) && defined(__NVCOMPILER)
        // nvc++ fails to find extents() from the base class when trying to satisfy the Mapping concept
        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> typename Mapping::ArrayExtents
        {
            return static_cast<const Mapping&>(*this).extents();
        }
#else
        using Mapping::extents;
#endif

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type blobIndex) const -> size_type
        {
            if(blobIndex < size_type{Mapping::blobCount})
                return Mapping::blobSize(blobIndex);
            return blockHitsSize(blobIndex) * sizeof(CountType);
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto compute(
            typename Mapping::ArrayIndex ai,
            RecordCoord<RecordCoords...> rc,
            Blobs& blobs) const -> decltype(auto)
        {
            static_assert(
                !std::is_const_v<Blobs>,
                "Cannot access (even just reading) data through Heatmap from const blobs/view, since we need to write "
                "the access counts");

            const auto [nr, offset] = Mapping::blobNrAndOffset(ai, rc);
            using Type = GetType<typename Mapping::RecordDim, RecordCoord<RecordCoords...>>;

            auto* hits = blockHits(nr, blobs);
            for(size_type i = 0; i < divCeil(size_type{sizeof(Type)}, Granularity); i++)
                internal::atomicInc(hits[offset / Granularity + i]);

            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return reinterpret_cast<CopyConst<std::remove_reference_t<decltype(blobs[nr][offset])>, Type>&>(
                blobs[nr][offset]);
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }

        // Returns the size of the block hits buffer for blob forBlobI in block counts.
        LLAMA_FN_HOST_ACC_INLINE auto blockHitsSize(size_type forBlobI) const -> size_type
        {
            return divCeil(Mapping::blobSize(forBlobI), Granularity);
        }
        LLAMA_SUPPRESS_HOST_DEVICE_WARNING

        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto blockHits(size_type forBlobI, const Blobs& blobs) const -> const CountType*
        {
            return reinterpret_cast<const CountType*>(&blobs[size_type{Mapping::blobCount} + forBlobI][0]);
        }

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto blockHits(size_type forBlobI, Blobs& blobs) const -> CountType*
        {
            return reinterpret_cast<CountType*>(&blobs[size_type{Mapping::blobCount} + forBlobI][0]);
        }

    private:
        static auto trimBlobRight(const CountType* bh, std::size_t size)
        {
            while(size > 0 && bh[size - 1] == 0)
                size--;
            return size;
        }

    public:
        /// Writes a data file suitable for gnuplot containing the heatmap data. You can use the script provided by
        /// \ref gnuplotScript to plot this data file.
        /// @param blobs The blobs of the view containing this mapping
        /// @param os The stream to write the data to. Should be some form of std::ostream.
        template<typename Blobs, typename OStream>
        void writeGnuplotDataFileAscii(
            const Blobs& blobs,
            OStream&& os,
            bool trimEnd = true,
            std::size_t wrapAfterBlocks = 64) const
        {
            for(std::size_t i = 0; i < Mapping::blobCount; i++)
            {
                auto* bh = blockHits(i, blobs);
                auto size = blockHitsSize(i);
                if(trimEnd)
                    size = trimBlobRight(bh, size);
                for(size_type j = 0; j < size; j++)
                {
                    if(j > 0)
                        os << (j % wrapAfterBlocks == 0 ? '\n' : ' ');
                    os << bh[j];
                }
                for(size_type j = size; j < roundUpToMultiple(size, wrapAfterBlocks); j++)
                    os << " 0";
                os << '\n';
            }
        }

        template<typename Blobs, typename OStream>
        void writeGnuplotDataFileBinary(
            const Blobs& blobs,
            OStream&& os,
            bool trimEnd = true,
            std::size_t afterBlobRoundUpTo = 64) const
        {
            for(std::size_t i = 0; i < Mapping::blobCount; i++)
            {
                auto* bh = blockHits(i, blobs);
                auto size = blockHitsSize(i);
                if(trimEnd)
                    size = trimBlobRight(bh, size);
                os.write(reinterpret_cast<const char*>(bh), size * sizeof(CountType));

                // round up before starting next blob
                CountType zero = 0;
                for(size_type j = size; j < roundUpToMultiple(size, afterBlobRoundUpTo); j++)
                    os.write(reinterpret_cast<const char*>(&zero), sizeof(CountType));
            }
        }

        /// An example script for plotting the ASCII heatmap data using gnuplot.
        static constexpr std::string_view gnuplotScriptAscii = R"(#!/bin/bash
gnuplot -p <<EOF
file = '${1:-plot.bin}'

set xtics format ""
set x2tics autofreq 32
set ytics autofreq 32
set yrange [] reverse
set link x2; set link y2
set x2label "Byte"
plot file matrix with image pixels axes x2y1
EOF
)";

        /// An example script for plotting the binary heatmap data using gnuplot.
        static constexpr std::string_view gnuplotScriptBinary = R"(#!/bin/bash
gnuplot -p <<EOF
file = '${1:-plot.bin}'
row = '${2:-64}'
format = '${3:-%uint64}'

counts = system('stat -c "%s" ${1:-plot.bin}')/8

set xtics format ""
set x2tics autofreq 32
set ytics autofreq 32
set yrange [] reverse
set link x2; set link y2
set x2label "Byte"
plot file binary array=(row,counts/row) format=format with image pixels axes x2y1
EOF
)";
    };

    template<typename Mapping>
    inline constexpr bool isHeatmap = false;

    template<typename Mapping, typename Mapping::ArrayExtents::value_type Granularity, typename CountType>
    inline constexpr bool isHeatmap<Heatmap<Mapping, Granularity, CountType>> = true;
} // namespace llama::mapping

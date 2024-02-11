// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#if __has_include(<fmt/format.h>)
#    include "ArrayIndexRange.hpp"
#    include "Core.hpp"
#    include "StructName.hpp"
#    include "View.hpp"

#    include <fmt/format.h>
#    include <functional>
#    include <optional>
#    include <string>
#    include <string_view>
#    include <vector>

namespace llama
{
    namespace internal
    {
        inline auto color(std::string_view recordCoordTags) -> std::uint32_t
        {
            auto c = static_cast<uint32_t>(std::hash<std::string_view>{}(recordCoordTags) &std::size_t{0xFFFFFF});
            c |= 0x404040u; // ensure color per channel is at least 0x40.
            return c;
        }

        // from: https://stackoverflow.com/questions/5665231/most-efficient-way-to-escape-xml-html-in-c-string
        inline auto xmlEscape(const std::string& str) -> std::string
        {
            std::string result;
            result.reserve(str.size());
            for(const char c : str)
            {
                switch(c)
                {
                case '&':
                    result.append("&amp;");
                    break;
                case '\"':
                    result.append("&quot;");
                    break;
                case '\'':
                    result.append("&apos;");
                    break;
                case '<':
                    result.append("&lt;");
                    break;
                case '>':
                    result.append("&gt;");
                    break;
                default:
                    result += c;
                    break;
                }
            }
            return result;
        }

        template<typename T, std::size_t Dim>
        auto formatArrayIndex(const ArrayIndex<T, Dim>& ai)
        {
            if constexpr(Dim == 1)
                return std::to_string(ai[0]);
            else
            {
                std::string s = "{";
                for(auto v : ai)
                {
                    if(s.size() >= 2)
                        s += ",";
                    s += std::to_string(v);
                }
                s += "}";
                return s;
            }
        }

        template<typename ArrayIndex>
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
        struct FieldBox
        {
            ArrayIndex arrayIndex;
            std::size_t flatRecordCoord;
            std::string_view recordCoordTags;
            NrAndOffset<std::size_t> nrAndOffset;
            std::size_t size;
        };

        template<typename View>
        void fillBlobsWithPattern(View& view, uint8_t pattern)
        {
            const auto& mapping = view.mapping();
            for(std::size_t i = 0; i < View::Mapping::blobCount; i++)
                std::memset(&view.blobs()[i][0], pattern, mapping.blobSize(i));
        }

        template<typename View, typename RecordCoord>
        void boxesFromComputedField(
            View& view,
            typename View::Mapping::ArrayExtents::Index ai,
            RecordCoord rc,
            std::vector<FieldBox<typename View::Mapping::ArrayExtents::Index>>& infos)
        {
            using Mapping = typename View::Mapping;
            using RecordDim = typename Mapping::RecordDim;

            auto emitInfo = [&](auto nrAndOffset, std::size_t size)
            {
                infos.push_back(
                    {ai,
                     flatRecordCoord<RecordDim, RecordCoord>,
                     prettyRecordCoord<RecordDim>(rc),
                     nrAndOffset,
                     size});
            };

            using Type = GetType<RecordDim, decltype(rc)>;
            // computed values can come from anywhere, so we can only apply heuristics
            auto& blobs = view.blobs();
            auto&& ref = view.mapping().compute(ai, rc, blobs);

            // if we get a reference, try to find the mapped address in one of the blobs
            if constexpr(std::is_lvalue_reference_v<decltype(ref)>)
            {
                auto address = reinterpret_cast<std::intptr_t>(&ref);
                for(std::size_t i = 0; i < blobs.size(); i++)
                {
                    // TODO(bgruber): this is UB, because we are comparing pointers from unrelated
                    // allocations
                    const auto front = reinterpret_cast<std::intptr_t>(&blobs[i][0]);
                    const auto back = reinterpret_cast<std::intptr_t>(&blobs[i][view.mapping().blobSize(i) - 1]);
                    if(front <= address && address <= back)
                    {
                        emitInfo(NrAndOffset{i, static_cast<std::size_t>(address - front)}, sizeof(Type));
                        return; // a mapping can only map to one location in the blobs
                    }
                }
            }

            if constexpr(std::is_default_constructible_v<Type>)
            {
                const auto infosBefore = infos.size();

                // try to observe written bytes
                const auto pattern = std::uint8_t{0xFF};
                fillBlobsWithPattern(view, pattern);
                ref = Type{}; // a broad range of types is default constructible and should write
                              // something zero-ish
                auto wasTouched = [&](auto b) { return static_cast<std::uint8_t>(b) != pattern; };
                for(std::size_t i = 0; i < Mapping::blobCount; i++)
                {
                    const auto blobSize = view.mapping().blobSize(i);
                    const auto* begin = &blobs[i][0];
                    const auto* end = begin + blobSize;

                    auto* searchBegin = begin;
                    while(true)
                    {
                        const auto* touchedBegin = std::find_if(searchBegin, end, wasTouched);
                        if(touchedBegin == end)
                            break;
                        const auto& touchedEnd = std::find_if_not(touchedBegin + 1, end, wasTouched);
                        emitInfo(
                            NrAndOffset{i, static_cast<std::size_t>(touchedBegin - begin)},
                            touchedEnd - touchedBegin);
                        if(touchedEnd == end)
                            break;
                        searchBegin = touchedEnd + 1;
                    }
                }

                if(infosBefore != infos.size())
                    return;
            }

            // if we come here, we could not find out where the value is coming from
            emitInfo(NrAndOffset{Mapping::blobCount, std::size_t{0}}, sizeof(Type));
        }

        template<typename Mapping>
        auto boxesFromMapping(const Mapping& mapping) -> std::vector<FieldBox<typename Mapping::ArrayExtents::Index>>
        {
            std::vector<FieldBox<typename Mapping::ArrayExtents::Index>> infos;

            std::optional<decltype(allocView(mapping))> view;
            if constexpr(hasAnyComputedField<Mapping>)
                view = allocView(mapping);

            using RecordDim = typename Mapping::RecordDim;
            for(auto ai : ArrayIndexRange{mapping.extents()})
                forEachLeafCoord<RecordDim>(
                    [&](auto rc)
                    {
                        using Type = GetType<RecordDim, decltype(rc)>;
                        if constexpr(llama::isComputed<Mapping, decltype(rc)>)
                            boxesFromComputedField(view.value(), ai, rc, infos);
                        else
                        {
                            const auto [nr, off] = mapping.blobNrAndOffset(ai, rc);
                            infos.push_back(
                                {ai,
                                 flatRecordCoord<RecordDim, decltype(rc)>,
                                 prettyRecordCoord<RecordDim>(rc),
                                 {static_cast<std::size_t>(nr), static_cast<std::size_t>(off)},
                                 sizeof(Type)});
                        }
                    });

            return infos;
        }

        template<typename ArrayIndex>
        auto breakBoxes(std::vector<FieldBox<ArrayIndex>> boxes, std::size_t wrapByteCount)
            -> std::vector<FieldBox<ArrayIndex>>
        {
            for(std::size_t i = 0; i < boxes.size(); i++)
            {
                auto& fb = boxes[i];
                if(fb.nrAndOffset.offset / wrapByteCount != (fb.nrAndOffset.offset + fb.size - 1) / wrapByteCount)
                {
                    const auto remainingSpace = wrapByteCount - fb.nrAndOffset.offset % wrapByteCount;
                    auto newFb = fb;
                    newFb.nrAndOffset.offset = fb.nrAndOffset.offset + remainingSpace;
                    newFb.size = fb.size - remainingSpace;
                    fb.size = remainingSpace;
                    boxes.push_back(newFb);
                }
            }
            return boxes;
        }

        inline auto cssClass(std::string tags)
        {
            std::replace(begin(tags), end(tags), '.', '_');
            std::replace(begin(tags), end(tags), '<', '_');
            std::replace(begin(tags), end(tags), '>', '_');
            return tags;
        };
    } // namespace internal

    /// Returns an SVG image visualizing the memory layout created by the given mapping. The created memory blocks are
    /// wrapped after wrapByteCount bytes.
    /// @param palette RGB colors as 0x00RRGGBB assigned cyclically. When empty, colors are assigned by hashing the
    /// record coordinate.
    LLAMA_EXPORT
    template<typename Mapping>
    auto toSvg(
        const Mapping& mapping,
        std::size_t wrapByteCount = 64,
        bool breakBoxes = true,
        const std::vector<std::uint32_t>& palette = {},
        std::string_view textColor = "black") -> std::string
    {
        constexpr auto byteSizeInPixel = 30;
        constexpr auto blobBlockWidth = 60;

        auto infos = internal::boxesFromMapping(mapping);
        if(breakBoxes)
            infos = internal::breakBoxes(std::move(infos), wrapByteCount);
        std::stable_sort(
            begin(infos),
            end(infos),
            [](const auto& a, const auto& b) {
                return std::tie(a.nrAndOffset.nr, a.nrAndOffset.offset)
                    < std::tie(b.nrAndOffset.nr, b.nrAndOffset.offset);
            });

        std::string svg;

        std::array<int, Mapping::blobCount + hasAnyComputedField<Mapping> + 1> blobYOffset{};
        auto writeBlobHeader = [&](std::size_t i, std::size_t size, std::string_view name)
        {
            const auto blobRows = (size + wrapByteCount - 1) / wrapByteCount;
            blobYOffset[i + 1] = blobYOffset[i] + (blobRows + 1) * byteSizeInPixel; // one row gap between blobs
            const auto height = blobRows * byteSizeInPixel;
            svg += fmt::format(
                R"a(<rect x="0" y="{}" width="{}" height="{}" fill="#AAA" stroke="#000"/>
<text x="{}" y="{}" fill="{}" text-anchor="middle">{}</text>
)a",
                blobYOffset[i],
                blobBlockWidth,
                height,
                blobBlockWidth / 2,
                blobYOffset[i] + height / 2,
                textColor,
                name);
        };
        for(std::size_t i = 0; i < Mapping::blobCount; i++)
            writeBlobHeader(i, mapping.blobSize(i), "Blob: " + std::to_string(i));

        svg = fmt::format(
                  R"(<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">
    <style>
        .label {{ font: {}px sans-serif; }}
    </style>
)",
                  blobBlockWidth + wrapByteCount * byteSizeInPixel,
                  blobYOffset.back() == 0 ? 987654321 : blobYOffset.back() - byteSizeInPixel,
                  byteSizeInPixel / 2)
            + svg;

        std::size_t computedSizeSoFar = 0;
        std::size_t lastBlobNr = std::numeric_limits<std::size_t>::max();
        std::size_t usedBytesInBlobSoFar = 0;
        for(const auto& info : infos)
        {
            if(lastBlobNr != info.nrAndOffset.nr)
            {
                usedBytesInBlobSoFar = 0;
                lastBlobNr = info.nrAndOffset.nr;
            }

            const auto blobY = blobYOffset[info.nrAndOffset.nr];
            const auto offset = [&]
            {
                if(info.nrAndOffset.nr < Mapping::blobCount)
                    return info.nrAndOffset.offset;

                const auto offset = computedSizeSoFar;
                computedSizeSoFar += info.size;
                return offset;
            }();
            auto x = (offset % wrapByteCount) * byteSizeInPixel + blobBlockWidth;
            auto y = (offset / wrapByteCount) * byteSizeInPixel + blobY;
            const auto fillColor = [&]
            {
                if(palette.empty())
                    return internal::color(info.recordCoordTags);
                return palette[info.flatRecordCoord % palette.size()];
            }();
            const auto width = byteSizeInPixel * info.size;

            const auto nextOffset = [&]
            {
                if(&info == &infos.back())
                    return std::numeric_limits<std::size_t>::max();
                const auto& nextInfo = (&info)[1];
                if(info.nrAndOffset.nr < Mapping::blobCount && info.nrAndOffset.nr == nextInfo.nrAndOffset.nr)
                    return nextInfo.nrAndOffset.offset;

                return std::numeric_limits<std::size_t>::max();
            }();
            const auto isOverlapped = offset < usedBytesInBlobSoFar || nextOffset < offset + info.size;
            usedBytesInBlobSoFar = offset + info.size;

            constexpr auto cropBoxes = true;
            if(cropBoxes)
            {
                svg += fmt::format(
                    R"(<svg x="{}" y="{}" width="{}" height="{}">
)",
                    x,
                    y,
                    width,
                    byteSizeInPixel);
                x = 0;
                y = 0;
            }
            svg += fmt::format(
                R"(<rect x="{}" y="{}" width="{}" height="{}" fill="#{:06X}" stroke="#000" fill-opacity="{}"/>
)",
                x,
                y,
                width,
                byteSizeInPixel,
                fillColor,
                isOverlapped ? 0.3 : 1.0);
            for(std::size_t i = 1; i < info.size; i++)
            {
                svg += fmt::format(
                    R"(<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#777"/>
)",
                    x + i * byteSizeInPixel,
                    y + byteSizeInPixel * 2 / 3,
                    x + i * byteSizeInPixel,
                    y + byteSizeInPixel);
            }
            svg += fmt::format(
                R"(<text x="{}" y="{}" fill="{}" text-anchor="middle" class="label">{} {}</text>
)",
                x + width / 2,
                y + byteSizeInPixel * 3 / 4,
                textColor,
                internal::formatArrayIndex(info.arrayIndex),
                internal::xmlEscape(std::string{info.recordCoordTags}));
            if(cropBoxes)
                svg += R"(</svg>
)";
        }

        if(hasAnyComputedField<Mapping>)
        {
            if(computedSizeSoFar > 0)
                writeBlobHeader(Mapping::blobCount, computedSizeSoFar, "Comp.");
            else
            {
                const auto blobRows = (wrapByteCount - 1) / wrapByteCount;
                blobYOffset[Mapping::blobCount + 1]
                    = blobYOffset[Mapping::blobCount] + blobRows * byteSizeInPixel; // fix-up, omit gap
            }

            // fix total SVG size
            const auto i = svg.find("987654321");
            assert(i != std::string::npos);
            svg.replace(i, 9, std::to_string(blobYOffset.back() - byteSizeInPixel));
        }

        svg += "</svg>";
        return svg;
    }

    LLAMA_EXPORT
    template<typename Mapping>
    auto toSvg(const Mapping& mapping, const std::vector<std::uint32_t>& palette, std::string_view textColor = "#000")
        -> std::string
    {
        return toSvg(mapping, 64, true, palette, textColor);
    }

    /// Returns an HTML document visualizing the memory layout created by the given mapping. The visualization is
    /// resizeable.
    LLAMA_EXPORT
    template<typename Mapping>
    auto toHtml(const Mapping& mapping) -> std::string
    {
        constexpr auto byteSizeInPixel = 30;
        constexpr auto rulerLengthInBytes = 512;
        constexpr auto rulerByteInterval = 8;

        auto infos = internal::boxesFromMapping(mapping);
        std::stable_sort(
            begin(infos),
            end(infos),
            [](const auto& a, const auto& b) {
                return std::tie(a.nrAndOffset.nr, a.nrAndOffset.offset)
                    < std::tie(b.nrAndOffset.nr, b.nrAndOffset.offset);
            });
        infos.erase(
            std::unique(
                begin(infos),
                end(infos),
                [](const auto& a, const auto& b) { return a.nrAndOffset == b.nrAndOffset; }),
            end(infos));

        std::string html;
        html += fmt::format(
            R"(<!DOCTYPE html>
<html>
<head>
<style>
.box {{
    outline: 1px solid;
    display: inline-block;
    white-space: nowrap;
    height: {}px;
    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #777 29px, #777 30px);
    text-align: center;
    overflow: hidden;
    vertical-align: middle;
}}
#ruler {{
    background: repeating-linear-gradient(90deg, #0000, #0000 29px, #000 29px, #000 30px);
    border-bottom: 1px solid;
    height: 20px;
    margin-bottom: 20px;
}}
#ruler div {{
    position: absolute;
    display: inline-block;
}}
)",
            byteSizeInPixel);
        using RecordDim = typename Mapping::RecordDim;
        forEachLeafCoord<RecordDim>(
            [&](auto rc)
            {
                constexpr int size = sizeof(GetType<RecordDim, decltype(rc)>);

                html += fmt::format(
                    R"(.{} {{
    width: {}px;
    background-color: #{:X};
}}
)",
                    internal::cssClass(std::string{prettyRecordCoord<RecordDim>(rc)}),
                    byteSizeInPixel * size,
                    internal::color(prettyRecordCoord<RecordDim>(rc)));
            });

        html += fmt::format(R"(</style>
</head>
<body>
    <header id="ruler">
)");
        for(auto i = 0; i < rulerLengthInBytes; i += rulerByteInterval)
            html += fmt::format(
                R"(</style>
        <div style="margin-left: {}px;">{}</div>)",
                i * byteSizeInPixel,
                i);
        html += fmt::format(R"(
    </header>
)");

        auto currentBlobNr = std::numeric_limits<std::size_t>::max();
        for(const auto& info : infos)
        {
            if(currentBlobNr != info.nrAndOffset.nr)
            {
                currentBlobNr = info.nrAndOffset.nr;
                html += fmt::format("<h1>Blob: {}</h1>", currentBlobNr);
            }
            html += fmt::format(
                R"(<div class="box {0}" title="{1} {2}">{1} {2}</div>)",
                internal::cssClass(std::string{info.recordCoordTags}),
                internal::formatArrayIndex(info.arrayIndex),
                internal::xmlEscape(std::string{info.recordCoordTags}));
        }
        html += R"(</body>
</html>)";
        return html;
    }
} // namespace llama

#endif

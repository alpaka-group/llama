// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Core.hpp"

#include <boost/container_hash/hash.hpp>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace llama
{
    namespace internal
    {
        template <std::size_t... Coords>
        auto toVec(DatumCoord<Coords...>) -> std::vector<std::size_t>
        {
            return {Coords...};
        }

        template <typename Tag>
        auto tagToString(Tag tag)
        {
            return structName(tag);
        }

        template <std::size_t N>
        auto tagToString(Index<N>)
        {
            return std::to_string(N);
        }

        template <
            typename DatumDomain,
            std::size_t... CoordsBefore,
            std::size_t CoordCurrent,
            std::size_t... CoordsAfter>
        void collectTagsAsStrings(
            std::vector<std::string>& v,
            DatumCoord<CoordsBefore...> before,
            DatumCoord<CoordCurrent, CoordsAfter...> after)
        {
            using Tag = GetTag<DatumDomain, DatumCoord<CoordsBefore..., CoordCurrent>>;
            v.push_back(tagToString(Tag{}));
            if constexpr (sizeof...(CoordsAfter) > 0)
                collectTagsAsStrings<DatumDomain>(
                    v,
                    DatumCoord<CoordsBefore..., CoordCurrent>{},
                    DatumCoord<CoordsAfter...>{});
        }

        template <typename DatumDomain, std::size_t... Coords>
        auto tagsAsStrings(DatumCoord<Coords...>) -> std::vector<std::string>
        {
            std::vector<std::string> v;
            collectTagsAsStrings<DatumDomain>(v, DatumCoord<>{}, DatumCoord<Coords...>{});
            return v;
        }

        template <typename Mapping, typename ArrayDomain, std::size_t... Coords>
        auto mappingBlobNrAndOffset(const Mapping& mapping, const ArrayDomain& udCoord, DatumCoord<Coords...>)
        {
            return mapping.template getBlobNrAndOffset<Coords...>(udCoord);
        }
    } // namespace internal

    /// Returns an SVG image visualizing the memory layout created by the given
    /// mapping. The created memory blocks are wrapped after wrapByteCount
    /// bytes.
    template <typename Mapping>
    auto toSvg(const Mapping& mapping, int wrapByteCount = 64) -> std::string
    {
        using ArrayDomain = typename Mapping::ArrayDomain;
        using DatumDomain = typename Mapping::DatumDomain;

        constexpr auto byteSizeInPixel = 30;

        struct DatumInfo
        {
            ArrayDomain udCoord;
            std::vector<std::size_t> ddIndices;
            std::vector<std::string> ddTags;
            NrAndOffset nrAndOffset;
            std::size_t size;
        };
        std::vector<DatumInfo> infos;

        for (auto udCoord : ArrayDomainIndexRange{mapping.arrayDomainSize})
        {
            forEach<DatumDomain>([&](auto coord) {
                constexpr int size = sizeof(GetType<DatumDomain, decltype(coord)>);
                infos.push_back(DatumInfo{
                    udCoord,
                    internal::toVec(coord),
                    internal::tagsAsStrings<DatumDomain>(coord),
                    internal::mappingBlobNrAndOffset(mapping, udCoord, coord),
                    size});
            });
        }

        auto formatDDTags = [](const std::vector<std::string>& tags) {
            std::string s;
            for (const auto& tag : tags)
            {
                if (!s.empty())
                    s += ".";
                s += tag;
            }
            return s;
        };

        auto formatUdCoord = [](const auto& coord) {
            if constexpr (std::is_same_v<decltype(coord), llama::ArrayDomain<1>>)
                return std::to_string(coord[0]);
            else
            {
                std::string s = "{";
                for (auto v : coord)
                {
                    if (s.size() >= 2)
                        s += ",";
                    s += std::to_string(v);
                }
                s += "}";
                return s;
            }
        };

        std::string svg;
        svg += fmt::format(
            R"(<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg">
    <style>
        .label {{ font: {}px sans-serif; }}
    </style>
)",
            byteSizeInPixel / 2);

        for (const auto& info : infos)
        {
            std::size_t blobY = 0;
            for (auto i = 0; i < info.nrAndOffset.nr; i++)
            {
                auto blobRows = (mapping.getBlobSize(i) + wrapByteCount - 1) / wrapByteCount;
                blobRows++; // one row gap between blobs
                blobY += blobRows * byteSizeInPixel;
            }

            const auto x = (info.nrAndOffset.offset % wrapByteCount) * byteSizeInPixel;
            const auto y = (info.nrAndOffset.offset / wrapByteCount) * byteSizeInPixel + blobY;

            const auto fill = boost::hash_value(info.ddIndices) & 0xFFFFFF;

            const auto width = byteSizeInPixel * info.size;
            svg += fmt::format(
                R"(<rect x="{}" y="{}" width="{}" height="{}" fill="#{:X}" stroke="#000"/>
)",
                x,
                y,
                width,
                byteSizeInPixel,
                fill);
            for (auto i = 1; i < info.size; i++)
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
                R"(<text x="{}" y="{}" fill="#000" text-anchor="middle" class="label">{} {}</text>
)",
                x + width / 2,
                y + byteSizeInPixel * 3 / 4,
                formatUdCoord(info.udCoord),
                formatDDTags(info.ddTags));
        }
        svg += "</svg>";
        return svg;
    }

    /// Returns an HTML document visualizing the memory layout created by the
    /// given mapping. The visualization is resizeable.
    template <typename Mapping>
    auto toHtml(const Mapping& mapping) -> std::string
    {
        using ArrayDomain = typename Mapping::ArrayDomain;
        using DatumDomain = typename Mapping::DatumDomain;

        constexpr auto byteSizeInPixel = 30;
        constexpr auto rulerLengthInBytes = 512;
        constexpr auto rulerByteInterval = 8;

        struct DatumInfo
        {
            ArrayDomain udCoord;
            std::vector<std::size_t> ddIndices;
            std::vector<std::string> ddTags;
            NrAndOffset nrAndOffset;
            std::size_t size;
        };
        std::vector<DatumInfo> infos;

        for (auto udCoord : ArrayDomainIndexRange{mapping.arrayDomainSize})
        {
            forEach<DatumDomain>([&](auto coord) {
                constexpr int size = sizeof(GetType<DatumDomain, decltype(coord)>);
                infos.push_back(DatumInfo{
                    udCoord,
                    internal::toVec(coord),
                    internal::tagsAsStrings<DatumDomain>(coord),
                    internal::mappingBlobNrAndOffset(mapping, udCoord, coord),
                    size});
            });
        }
        std::sort(begin(infos), end(infos), [](const DatumInfo& a, const DatumInfo& b) {
            return std::tie(a.nrAndOffset.nr, a.nrAndOffset.offset) < std::tie(b.nrAndOffset.nr, b.nrAndOffset.offset);
        });

        auto formatDDTags = [](const std::vector<std::string>& tags) {
            std::string s;
            for (const auto& tag : tags)
            {
                if (!s.empty())
                    s += ".";
                s += tag;
            }
            return s;
        };

        auto cssClass = [](const std::vector<std::string>& tags) {
            std::string s;
            for (const auto& tag : tags)
            {
                if (!s.empty())
                    s += "_";
                s += tag;
            }
            return s;
        };

        auto formatUdCoord = [](const auto& coord) {
            if constexpr (std::is_same_v<decltype(coord), llama::ArrayDomain<1>>)
                return std::to_string(coord[0]);
            else
            {
                std::string s = "{";
                for (auto v : coord)
                {
                    if (s.size() >= 2)
                        s += ",";
                    s += std::to_string(v);
                }
                s += "}";
                return s;
            }
        };

        std::string svg;
        svg += fmt::format(
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
        forEach<DatumDomain>([&](auto coord) {
            constexpr int size = sizeof(GetType<DatumDomain, decltype(coord)>);

            svg += fmt::format(
                R"(.{} {{
    width: {}px;
    background-color: #{:X};
}}
)",
                cssClass(internal::tagsAsStrings<DatumDomain>(coord)),
                byteSizeInPixel * size,
                boost::hash_value(internal::toVec(coord)) & 0xFFFFFF);
        });

        svg += fmt::format(R"(</style>
</head>
<body>
    <header id="ruler">
)");
        for (auto i = 0; i < rulerLengthInBytes; i += rulerByteInterval)
            svg += fmt::format(
                R"(</style>
        <div style="margin-left: {}px;">{}</div>)",
                i * byteSizeInPixel,
                i);
        svg += fmt::format(R"(
    </header>
)");

        auto currentBlobNr = std::numeric_limits<std::size_t>::max();
        for (const auto& info : infos)
        {
            if (currentBlobNr != info.nrAndOffset.nr)
            {
                currentBlobNr = info.nrAndOffset.nr;
                svg += fmt::format("<h1>Blob: {}</h1>", currentBlobNr);
            }
            const auto width = byteSizeInPixel * info.size;
            svg += fmt::format(
                R"(<div class="box {0}" title="{1} {2}">{1} {2}</div>)",
                cssClass(info.ddTags),
                formatUdCoord(info.udCoord),
                formatDDTags(info.ddTags));
        }
        svg += R"(</body>
</html>)";
        return svg;
    }
} // namespace llama

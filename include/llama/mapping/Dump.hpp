#pragma once

#include "../ForEach.hpp"
#include "../Functions.hpp"
#include "../UserDomain.hpp"

#include <boost/container_hash/hash.hpp>
#include <fmt/format.h>
#include <string>
#include <vector>

namespace llama::mapping
{
    namespace internal
    {
        template<std::size_t... Coords>
        auto toVec(DatumCoord<Coords...>) -> std::vector<std::size_t>
        {
            return {Coords...};
        }

        template<typename Tag>
        auto tagToString(Tag)
        {
            auto s = boost::core::demangle(typeid(Tag).name());
            if(const auto pos = s.rfind(':'); pos != std::string::npos)
                s = s.substr(pos + 1);
            return s;
        }

        template<std::size_t N>
        auto tagToString(Index<N>)
        {
            return std::to_string(N);
        }

        template<
            typename DatumDomain,
            std::size_t... CoordsBefore,
            std::size_t CoordCurrent,
            std::size_t... CoordsAfter>
        void collectTagsAsStrings(
            std::vector<std::string> & v,
            DatumCoord<CoordsBefore...> before,
            DatumCoord<CoordCurrent, CoordsAfter...> after)
        {
            using Tag = GetUID<
                DatumDomain,
                DatumCoord<CoordsBefore..., CoordCurrent>>;
            v.push_back(tagToString(Tag{}));
            if constexpr(sizeof...(CoordsAfter) > 0)
                collectTagsAsStrings<DatumDomain>(
                    v,
                    DatumCoord<CoordsBefore..., CoordCurrent>{},
                    DatumCoord<CoordsAfter...>{});
        }

        template<typename DatumDomain, std::size_t... Coords>
        auto tagsAsStrings(DatumCoord<Coords...>) -> std::vector<std::string>
        {
            std::vector<std::string> v;
            collectTagsAsStrings<DatumDomain>(
                v, DatumCoord<>{}, DatumCoord<Coords...>{});
            return v;
        }

        template<typename Mapping, typename UserDomain, std::size_t... Coords>
        auto mappingOffset(
            const Mapping & mapping,
            const UserDomain & udCoord,
            DatumCoord<Coords...>)
        {
            return mapping.template getBlobNrAndOffset<Coords...>(udCoord)
                .offset;
        }
    }

    template<typename Mapping>
    auto toSvg(const Mapping & mapping, int wrapByteCount = 64) -> std::string
    {
        using UserDomain = typename Mapping::UserDomain;
        using DatumDomain = typename Mapping::DatumDomain;

        constexpr auto byteSizeInPixel = 30;

        struct DatumInfo
        {
            UserDomain udCoord;
            std::vector<std::size_t> ddIndices;
            std::vector<std::string> ddTags;
            std::size_t offset;
            std::size_t size;
        };
        std::vector<DatumInfo> infos;

        for(auto udCoord : UserDomainCoordRange{mapping.userDomainSize})
        {
            forEach<DatumDomain>([&](auto outer, auto inner) {
                constexpr int size
                    = sizeof(GetType<DatumDomain, decltype(inner)>);
                infos.push_back(DatumInfo{
                    udCoord,
                    internal::toVec(inner),
                    internal::tagsAsStrings<DatumDomain>(inner),
                    internal::mappingOffset(mapping, udCoord, inner),
                    size});
            });
        }

        auto formatDDTags = [](const std::vector<std::string> & tags) {
            std::string s;
            for(const auto & tag : tags)
            {
                if(!s.empty())
                    s += ".";
                s += tag;
            }
            return s;
        };

        auto formatUdCoord = [](const auto & coord) {
            if constexpr(std::is_same_v<decltype(coord), llama::UserDomain<1>>)
                return std::to_string(coord[0]);
            else
            {
                std::string s = "{";
                for(auto v : coord)
                {
                    if(s.size() >= 2)
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

        for(const auto & info : infos)
        {
            const auto x = (info.offset % wrapByteCount) * byteSizeInPixel;
            const auto y = (info.offset / wrapByteCount) * byteSizeInPixel;

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
            for(auto i = 1; i < info.size; i++)
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
}

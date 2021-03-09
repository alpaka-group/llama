// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam LinearizeArrayDomainFunctor Defines how the
    /// user domain should be mapped into linear numbers and how big the linear
    /// domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        typename SeparateBuffers = std::false_type, // TODO: make this a bool. Needs work in Split mapping
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct SoA
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = []() constexpr
        {
            if constexpr (SeparateBuffers::value)
            {
                std::size_t count = 0;
                forEachLeave<DatumDomain>([&](auto) constexpr { count++; });
                return count;
            }
            else
                return 1;
        }
        ();

        constexpr SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr SoA(ArrayDomain size, DatumDomain = {}, SeparateBuffers = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto getBlobSize(std::size_t blobIndex) const -> std::size_t
        {
            if constexpr (SeparateBuffers::value)
            {
                constexpr llama::Array<std::size_t, blobCount> typeSizes = []() constexpr
                {
                    llama::Array<std::size_t, blobCount> r{};
                    std::size_t i = 0;
                    forEachLeave<DatumDomain>([&](auto coord) constexpr {
                        r[i++] = sizeof(GetType<DatumDomain, decltype(coord)>);
                    });
                    return r;
                }
                ();
                return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * typeSizes[blobIndex];
            }
            else
                return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            if constexpr (SeparateBuffers::value)
            {
                using TargetDatumCoord = DatumCoord<DatumDomainCoord...>;
                constexpr auto blob = [&]() constexpr
                {
                    std::size_t index = 0;
                    bool found = false;
                    forEachLeave<DatumDomain>([&](auto c) constexpr {
                        if constexpr (std::is_same_v<decltype(c), TargetDatumCoord>)
                            found = true;
                        else if (!found)
                            index++;
                    });
                    if (!found)
                        return std::numeric_limits<std::size_t>::max();
                    return index;
                }
                ();
                static_assert(
                    blob != std::numeric_limits<std::size_t>::max(),
                    "Passed TargetDatumCoord must be in datum domain");

                LLAMA_FORCE_INLINE_RECURSIVE
                const auto offset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize)
                    * sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>);
                return {blob, offset};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                const auto offset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize)
                        * sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>)
                    + offsetOf<
                          DatumDomain,
                          DatumCoord<DatumDomainCoord...>> * LinearizeArrayDomainFunctor{}.size(arrayDomainSize);
                return {0, offset};
            }
        }

        ArrayDomain arrayDomainSize;
    };

    template <
        typename ArrayDomain,
        typename DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using SingleBlobSoA = SoA<ArrayDomain, DatumDomain, std::false_type, LinearizeArrayDomainFunctor>;

    template <
        typename ArrayDomain,
        typename DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using MultiBlobSoA = SoA<ArrayDomain, DatumDomain, std::true_type, LinearizeArrayDomainFunctor>;

    template <
        typename SeparateBuffers = std::false_type,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct PreconfiguredSoA
    {
        template <typename ArrayDomain, typename DatumDomain>
        using type = SoA<ArrayDomain, DatumDomain, SeparateBuffers, LinearizeArrayDomainFunctor>;
    };
} // namespace llama::mapping

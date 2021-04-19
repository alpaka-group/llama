// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Common.hpp"

#include <limits>

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam SeparateBuffers If true, every element of the datum domain is mapped to its own buffer.
    /// \tparam LinearizeArrayDomainFunctor Defines how the user domain should be mapped into linear numbers and how big
    /// the linear domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        bool SeparateBuffers = false,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct SoA
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = []() constexpr
        {
            if constexpr (SeparateBuffers)
                return boost::mp11::mp_size<llama::FlattenDatumDomain<DatumDomain>>::value;
            else
                return 1;
        }
        ();

        constexpr SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr SoA(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(std::size_t blobIndex) const -> std::size_t
        {
            if constexpr (SeparateBuffers)
            {
                constexpr llama::Array<std::size_t, blobCount> typeSizes = []() constexpr
                {
                    llama::Array<std::size_t, blobCount> r{};
                    std::size_t i = 0;
                    forEachLeaf<DatumDomain>([&](auto coord) constexpr
                                             { r[i++] = sizeof(GetType<DatumDomain, decltype(coord)>); });
                    return r;
                }
                ();
                return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * typeSizes[blobIndex];
            }
            else
                return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            if constexpr (SeparateBuffers)
            {
                using TargetDatumCoord = DatumCoord<DatumDomainCoord...>;
                constexpr auto blob = [&]() constexpr
                {
                    std::size_t index = 0;
                    bool found = false;
                    forEachLeaf<DatumDomain>([&](auto c) constexpr
                                             {
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

                const auto offset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize)
                    * sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>);
                return {blob, offset};
            }
            else
            {
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

    /// Struct of array mapping storing the entire layout in a single blob. See \see SoA.
    template <
        typename ArrayDomain,
        typename DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using SingleBlobSoA = SoA<ArrayDomain, DatumDomain, false, LinearizeArrayDomainFunctor>;

    /// Struct of array mapping storing each attribute of the datum domain in a separate blob. See \see SoA.
    template <
        typename ArrayDomain,
        typename DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    using MultiBlobSoA = SoA<ArrayDomain, DatumDomain, true, LinearizeArrayDomainFunctor>;

    template <bool SeparateBuffers = false, typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct PreconfiguredSoA
    {
        template <typename ArrayDomain, typename DatumDomain>
        using type = SoA<ArrayDomain, DatumDomain, SeparateBuffers, LinearizeArrayDomainFunctor>;
    };
} // namespace llama::mapping

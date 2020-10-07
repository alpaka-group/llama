// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Functions.hpp"
#include "../Types.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Struct of array mapping. Used to create a \ref View via \ref allocView.
    /// \tparam LinearizeArrayDomainFunctor Defines how the
    /// user domain should be mapped into linear numbers and how big the linear
    /// domain gets.
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        typename LinearizeArrayDomainFunctor = LinearizeArrayDomainCpp>
    struct SoA
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        SoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        SoA(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto getBlobSize(std::size_t) const -> std::size_t
        {
            return LinearizeArrayDomainFunctor{}.size(arrayDomainSize) * sizeOf<DatumDomain>;
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto offset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize)
                    * sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>)
                + offsetOf<DatumDomain, DatumDomainCoord...> * LinearizeArrayDomainFunctor{}.size(arrayDomainSize);
            return {0, offset};
        }

        ArrayDomain arrayDomainSize;
    };
} // namespace llama::mapping

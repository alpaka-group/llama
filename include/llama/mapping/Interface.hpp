// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Types.hpp"

// FIXME: what does this do? There is no implementation
// If this is a template for mappings, it should be turned into a concept
namespace llama::mapping
{
    template<typename T_UserDomain, typename T_DatumDomain>
    struct Interface
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;

        static constexpr std::size_t blobCount = 0;

        LLAMA_FN_HOST_ACC_INLINE
        auto getBlobSize(std::size_t const blobNr) const -> std::size_t;

        template<typename T_DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndByte(UserDomain coord) const
            -> NrAndOffset;
    };
}

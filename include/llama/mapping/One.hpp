/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

#include "../Functions.hpp"
#include "../Types.hpp"

namespace llama::mapping
{
    /// Maps all UserDomain coordinates into the same location and layouts
    /// struct members consecutively. This mapping is used for temporary, single
    /// element views.
    template<typename T_UserDomain, typename T_DatumDomain>
    struct One
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;

        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const
            -> std::size_t
        {
            return sizeOf<DatumDomain>;
        }

        template<std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain) const
            -> NrAndOffset
        {
            constexpr auto offset = offsetOf<DatumDomain, DatumDomainCoord...>;
            return {0, offset};
        }
    };
}

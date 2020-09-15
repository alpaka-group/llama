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

#include "../Types.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct mapping. Used to create a \ref View via \ref allocView.
    /// \tparam LinearizeUserDomainAdressFunctor Defines how the
    /// user domain should be mapped into linear numbers.
    /// \tparam ExtentUserDomainAdressFunctor Defines how the total number of
    /// \ref UserDomain indices is calculated.
    template<
        typename T_UserDomain,
        typename T_DatumDomain,
        typename LinearizeUserDomainAdressFunctor = LinearizeUserDomainAdress,
        typename ExtentUserDomainAdressFunctor = ExtentUserDomainAdress>
    struct AoS
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        AoS() = default;

        LLAMA_FN_HOST_ACC_INLINE
        AoS(UserDomain size) : userDomainSize(size) {}

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const
            -> std::size_t
        {
            return ExtentUserDomainAdressFunctor{}(
                userDomainSize)*sizeOf<DatumDomain>;
        }

        template<std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain coord) const
            -> NrAndOffset
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto offset
                = LinearizeUserDomainAdressFunctor{}(coord, userDomainSize)
                    * sizeOf<
                        DatumDomain> + offsetOf<DatumDomain, DatumDomainCoord...>;
            return {0, offset};
        }

        UserDomain userDomainSize;
    };
}

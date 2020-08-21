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
#include "../UserDomain.hpp"

namespace llama::mapping
{
    /** Neither struct of array nor array of struct mapping as only exactly one
     *  element (in the user domain) can be mapped. If more than one element is
     *  tried to be mapped all virtual datums are mapped to the very same
     * memory. This mapping is especially used for temporary views on the stack
     * allocated with \ref stackViewAlloc. \tparam T_UserDomain type of the user
     * domain, expected to have only element, although more are working (but
     * doesn't make sense) \tparam T_DatumDomain type of the datum domain \see
     * stackViewAlloc, OneOnStackFactory, allocator::Stack
     */
    template<typename T_UserDomain, typename T_DatumDomain>
    struct One
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const
            -> std::size_t
        {
            return SizeOf<DatumDomain>::value;
        }

        template<std::size_t... DatumDomainCoord>
        auto getBlobNrAndOffset(UserDomain coord) const -> NrAndOffset
        {
            const auto offset
                = LinearBytePos<DatumDomain, DatumDomainCoord...>::value;
            return {0, offset};
        }
    };
}

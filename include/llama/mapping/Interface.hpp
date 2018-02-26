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

namespace llama
{

namespace mapping
{

template<
    typename T_UserDomain,
    typename T_DateDomain
>
struct Interface
{
    using UserDomain = T_UserDomain;
    using DateDomain = T_DateDomain;
    static constexpr std::size_t blobCount = 0;

    Interface( UserDomain const );

    LLAMA_FN_HOST_ACC_INLINE
    auto
    getBlobSize( std::size_t const blobNr ) const
    -> std::size_t;

    template< typename T_DateDomainCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    getBlobByte(
        UserDomain const coord,
        UserDomain const size
    ) const
    -> std::size_t;

    LLAMA_FN_HOST_ACC_INLINE
    auto
    getBlobNr(
        UserDomain const coord,
        UserDomain const size
    ) const
    -> std::size_t;
};

} // namespace mapping

} // namespace llama

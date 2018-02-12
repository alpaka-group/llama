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
#include "../UserDomain.hpp"

namespace llama
{

namespace mapping
{

template<
    typename T_UserDomain,
    typename T_DateDomain,
    typename T_LinearizeUserDomainAdressFunctor =
        LinearizeUserDomainAdress< T_UserDomain::count >,
    typename T_ExtentUserDomainAdressFunctor =
        ExtentUserDomainAdress< T_UserDomain::count >
>
struct AoS
{
    using UserDomain = T_UserDomain;
    using DateDomain = T_DateDomain;
    static constexpr std::size_t blobCount = 1;

    AoS( UserDomain const size ) :
		userDomainSize( size )
	{ }

    inline
    auto
    getBlobSize( std::size_t const ) const
    -> std::size_t
    {
        return T_ExtentUserDomainAdressFunctor()(userDomainSize)
            * DateDomain::size;
    }

    template< std::size_t... T_dateDomainCoord >
    inline
    auto
    getBlobByte( UserDomain const coord ) const
    -> std::size_t
    {
        return T_LinearizeUserDomainAdressFunctor()(
                coord,
                userDomainSize
            )
            * DateDomain::size
            + DateDomain::template LinearBytePos< T_dateDomainCoord... >::value;
    }

    template< std::size_t... T_dateDomainCoord >
    constexpr
    auto
    getBlobNr( UserDomain const coord ) const
    -> std::size_t
    {
        return 0;
    }
    UserDomain const userDomainSize;
};

} // namespace mapping

} // namespace llama

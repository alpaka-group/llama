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
#include "../GetType.hpp"
#include "../UserDomain.hpp"

namespace llama
{

namespace mapping
{

template<
    typename __UserDomain,
    typename __DateDomain,
    typename LinearizeUserDomainAdressFunctor =
		LinearizeUserDomainAdress< __UserDomain::count >,
    typename ExtentUserDomainAdressFunctor =
		ExtentUserDomainAdress< __UserDomain::count >
>
struct SoA
{
    using UserDomain = __UserDomain;
    using DateDomain = __DateDomain;
    SoA(const UserDomain size) :
        userDomainSize(size),
        extentUserDomainAdress(
			ExtentUserDomainAdressFunctor()( userDomainSize )
		)
    {}
    static constexpr size_t blobCount = 1;
    inline size_t getBlobSize( const size_t ) const
    {
        return extentUserDomainAdress * DateDomain::size;
    }
    template< size_t... dateDomainCoord >
    inline size_t getBlobByte( const UserDomain coord ) const
    {
        return LinearizeUserDomainAdressFunctor()( coord, userDomainSize )
            * sizeof( typename GetType<
				DateDomain,
				dateDomainCoord...
			>::type )
            + DateDomain::template LinearBytePos< dateDomainCoord... >::value
            * extentUserDomainAdress;
    }
    template< size_t... dateDomainCoord >
    constexpr size_t getBlobNr( const UserDomain coord ) const
    {
        return 0;
    }
    const UserDomain userDomainSize;
    const size_t extentUserDomainAdress;
};

} //namespace mapping

} //namespace llama


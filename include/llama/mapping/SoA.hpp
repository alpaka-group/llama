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
    SoA(UserDomain const size) :
        userDomainSize(size),
        extentUserDomainAdress(
			ExtentUserDomainAdressFunctor()( userDomainSize )
		)
    {}
    static constexpr std::size_t blobCount = 1;
    inline std::size_t getBlobSize( std::size_t const ) const
    {
        return extentUserDomainAdress * DateDomain::size;
    }
    template< std::size_t... dateDomainCoord >
    inline std::size_t getBlobByte( UserDomain const coord ) const
    {
        return LinearizeUserDomainAdressFunctor()( coord, userDomainSize )
            * sizeof( typename GetType<
				DateDomain,
				dateDomainCoord...
			>::type )
            + DateDomain::template LinearBytePos< dateDomainCoord... >::value
            * extentUserDomainAdress;
    }
    template< std::size_t... dateDomainCoord >
    constexpr std::size_t getBlobNr( UserDomain const coord ) const
    {
        return 0;
    }
    UserDomain const userDomainSize;
    std::size_t const extentUserDomainAdress;
};

} //namespace mapping

} //namespace llama


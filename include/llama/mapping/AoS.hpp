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

/** Array of struct mapping which can be used for creating a \ref View with a
 *  \ref Factory. For the interface details see \ref Factory.
 * \tparam T_UserDomain type of the user domain
 * \tparam T_DatumDomain type of the datum domain
 * \tparam T_LinearizeUserDomainAdressFunctor Defines how the user domain should
 *  be linearized, e.g. C like with the last dimension being the "fast" one
 *  (\ref LinearizeUserDomainAdress, default) or Fortran like with the first
 *  dimension being the "fast" one (\ref LinearizeUserDomainAdressLikeFortran).
 * \tparam T_ExtentUserDomainAdressFunctor Defines how the size of the view
 *  shall be created. Should fit for `T_LinearizeUserDomainAdressFunctor`. Only
 *  right now implemented and default value is \ref ExtentUserDomainAdress.
 * \see SoA
 */
template<
    typename T_UserDomain,
    typename T_DatumDomain,
    typename T_LinearizeUserDomainAdressFunctor =
        LinearizeUserDomainAdress< T_UserDomain::count >,
    typename T_ExtentUserDomainAdressFunctor =
        ExtentUserDomainAdress< T_UserDomain::count >
>
struct AoS
{
    using UserDomain = T_UserDomain;
    using DatumDomain = T_DatumDomain;
    static constexpr std::size_t blobCount = 1;

    /// \param size size of the user domain
    LLAMA_FN_HOST_ACC_INLINE
    AoS( UserDomain const size ) :
		userDomainSize( size )
	{ }

    AoS() = default;
    AoS( AoS const & ) = default;
    AoS( AoS && ) = default;
    ~AoS( ) = default;

    LLAMA_FN_HOST_ACC_INLINE
    auto
    getBlobSize( std::size_t const ) const
    -> std::size_t
    {
        return T_ExtentUserDomainAdressFunctor()(userDomainSize)
            * SizeOf<DatumDomain>::value;
    }

    template< std::size_t... T_datumDomainCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    getBlobByte( UserDomain const coord ) const
    -> std::size_t
    {
        return T_LinearizeUserDomainAdressFunctor()(
                coord,
                userDomainSize
            )
            * SizeOf<DatumDomain>::value
            + LinearBytePos<
                DatumDomain,
                T_datumDomainCoord...
            >::value;
    }

    template< std::size_t... T_datumDomainCoord >
    LLAMA_FN_HOST_ACC_INLINE
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

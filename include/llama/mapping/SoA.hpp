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
    typename T_UserDomain,
    typename T_DatumDomain,
    typename T_LinearizeUserDomainAdressFunctor =
        LinearizeUserDomainAdress< T_UserDomain::count >,
    typename T_ExtentUserDomainAdressFunctor =
        ExtentUserDomainAdress< T_UserDomain::count >
>
struct SoA
{
    using UserDomain = T_UserDomain;
    using DatumDomain = T_DatumDomain;
    static constexpr std::size_t blobCount = 1;

    LLAMA_FN_HOST_ACC_INLINE
    SoA( UserDomain const size ) :
        userDomainSize( size ),
        extentUserDomainAdress(
            T_ExtentUserDomainAdressFunctor()( userDomainSize )
        )
    {}

    SoA() = default;
    SoA( SoA const & ) = default;
    SoA( SoA && ) = default;
    ~SoA( ) = default;

    LLAMA_FN_HOST_ACC_INLINE
    auto
    getBlobSize( std::size_t const ) const
    -> std::size_t
    {
        return extentUserDomainAdress * DatumDomain::TypeTree::sizeOf;
    }

    template< std::size_t... T_datumDomainCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    getBlobByte( UserDomain const coord ) const
    -> std::size_t
    {
        return T_LinearizeUserDomainAdressFunctor()( coord, userDomainSize )
            * sizeof( typename GetType<
                typename DatumDomain::TypeTree,
                T_datumDomainCoord...
            >::type )
            + DatumDomain::TypeTree::template LinearBytePos< T_datumDomainCoord... >::value
            * extentUserDomainAdress;
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
    std::size_t const extentUserDomainAdress;
};

} // namespace mapping

} // namespace llama


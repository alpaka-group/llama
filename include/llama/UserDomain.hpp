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

#include "Types.hpp"

namespace llama
{

template< std::size_t dim >
struct ExtentUserDomainAdress
{
    inline
    auto
    operator()( UserDomain< dim > const & size ) const
    -> std::size_t
    {
        return ExtentUserDomainAdress< dim - 1 >()( size.pop_front() )
            * size[ 0 ];
    }
};

template< >
struct ExtentUserDomainAdress< 1 >
{
    inline
    auto
    operator()( UserDomain< 1 > const & size ) const
    -> std::size_t
    {
        return size[ 0 ];
    }
};

template<
    size_t dim,
    size_t it = dim
>
struct LinearizeUserDomainAdress
{
    inline
    auto
    operator()(
        UserDomain< dim > const & coord,
        UserDomain< dim > const & size
    ) const
    -> std::size_t
    {
        return coord[ it - 1 ]
            + LinearizeUserDomainAdress<
                dim,
                it-1
            >()(
                coord,
                size
            )
            * size[ it - 1 ];
    }
};

template< std::size_t dim >
struct LinearizeUserDomainAdress<
    dim,
    1
>
{
    inline
    auto
    operator()(
        UserDomain< dim > const & coord,
        UserDomain< dim > const & size
    ) const
    -> std::size_t
    {
        return coord[ 0 ];
    }
};

template< std::size_t dim >
struct LinearizeUserDomainAdressLikeFortran
{
    inline
    auto
    operator()(
        UserDomain< dim > const & coord,
        UserDomain< dim > const & size
    ) const
    -> std::size_t
    {
        return coord[ 0 ]
            + LinearizeUserDomainAdressLikeFortran< dim - 1 >()(
                coord.pop_front(),
                size.pop_front()
            )
            * size[ 0 ];
    }
};

template< >
struct LinearizeUserDomainAdressLikeFortran< 1 >
{
    inline
    auto
    operator()(
        UserDomain< 1 > const & coord,
        UserDomain< 1 > const & size
    ) const
    -> std::size_t
    {
        return coord[ 0 ];
    }
};

} // namespace llama

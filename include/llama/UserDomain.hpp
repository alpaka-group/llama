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

template< std::size_t T_dim >
struct ExtentUserDomainAdress
{
    inline
    auto
    operator()( UserDomain< T_dim > const & size ) const
    -> std::size_t
    {
        return ExtentUserDomainAdress< T_dim - 1 >()( size.pop_front() )
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
    std::size_t T_dim,
    std::size_t T_it = T_dim
>
struct LinearizeUserDomainAdress
{
    inline
    auto
    operator()(
        UserDomain< T_dim > const & coord,
        UserDomain< T_dim > const & size
    ) const
    -> std::size_t
    {
        return coord[ T_it - 1 ]
            + LinearizeUserDomainAdress<
                T_dim,
                T_it - 1
            >()(
                coord,
                size
            )
            * size[ T_it - 1 ];
    }
};

template< std::size_t T_dim >
struct LinearizeUserDomainAdress<
    T_dim,
    1
>
{
    inline
    auto
    operator()(
        UserDomain< T_dim > const & coord,
        UserDomain< T_dim > const & size
    ) const
    -> std::size_t
    {
        return coord[ 0 ];
    }
};

template< std::size_t T_dim >
struct LinearizeUserDomainAdressLikeFortran
{
    inline
    auto
    operator()(
        UserDomain< T_dim > const & coord,
        UserDomain< T_dim > const & size
    ) const
    -> std::size_t
    {
        return coord[ 0 ]
            + LinearizeUserDomainAdressLikeFortran< T_dim - 1 >()(
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

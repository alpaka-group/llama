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

#include "GetType.hpp"
#include "Array.hpp"

namespace llama
{

template<
    typename T_Mapping,
    typename T_BlobType
>
struct View
{
    using BlobType = T_BlobType;
    using Mapping = T_Mapping;
    View(
        Mapping mapping,
        Array<
            BlobType,
            Mapping::blobCount
        > blob
    ) :
        mapping( mapping ),
        blob( blob )
    { }

    template< std::size_t... T_dateDomain >
    auto
    accessor( typename Mapping::UserDomain const userDomain )
    -> typename GetType<
        typename Mapping::DateDomain,
        T_dateDomain...
    >::type &
    {
        auto const nr =
            mapping.template getBlobNr< T_dateDomain... >( userDomain );
        auto const byte =
            mapping.template getBlobByte< T_dateDomain... >( userDomain );
        return *( reinterpret_cast< typename GetType<
                typename Mapping::DateDomain,
                T_dateDomain...
            >::type* > (
                &blob[ nr ][ byte ]
            )
        );
    }

    struct VirtualDate
    {
        template< std::size_t... T_coord >
        auto
        access( DateCoord< T_coord... > && = DateCoord< T_coord... >() )
        -> typename GetType<
            typename Mapping::DateDomain,
            T_coord...
        >::type &
        {
            return view.accessor< T_coord... >( userDomainPos );
        }

        template< std::size_t... T_coord >
        auto
        operator()( DateCoord< T_coord... > && dc= DateCoord< T_coord... >() )
        -> typename GetType<
            typename Mapping::DateDomain,
            T_coord...
        >::type &
        {
            return access< T_coord... >(
                std::forward< DateCoord< T_coord... > >( dc )
            );
        }

        typename Mapping::UserDomain const userDomainPos;
        View<
            Mapping,
            BlobType
        > & view;
    };

    auto
    operator()( typename Mapping::UserDomain const userDomain )
    -> VirtualDate
    {
        return VirtualDate{
                userDomain,
                *this
            };
    }

    template< typename... T_Coord >
    auto
    operator()( T_Coord... coord )
    -> VirtualDate
    {
        return VirtualDate{
                typename Mapping::UserDomain{coord...},
                *this
            };
    }


    Array<
        BlobType,
        Mapping::blobCount
    > blob;
    const Mapping mapping;
};

} // namespace llama

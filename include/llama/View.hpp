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

namespace llama
{

template<
    typename __Mapping,
    typename __BlobType
>
struct View
{
    using BlobType = __BlobType;
    using Mapping = __Mapping;
    View( Mapping mapping ) :
        mapping( mapping )
    { }

    template< std::size_t... dd >
    auto
    accessor( typename Mapping::UserDomain const ud )
    -> typename GetType<
        typename Mapping::DateDomain,
        dd...
    >::type &
    {
        auto const nr = mapping.template getBlobNr< dd... >( ud );
        auto const byte = mapping.template getBlobByte< dd... >( ud );
        return *( reinterpret_cast< typename GetType<
                typename Mapping::DateDomain,
                dd...
            >::type* > (
                &blob[ nr ][ byte ]
            )
        );
    }

    struct VirtualDate
    {
        template< std::size_t... coord >
        auto
        access( DateCoord< coord... > && = DateCoord< coord... >() )
        -> typename GetType<
            typename Mapping::DateDomain,
            coord...
        >::type &
        {
            return view.accessor< coord... >( userDomainPos );
        }

        template< std::size_t... coord >
		auto
        operator()( DateCoord< coord... > && dc= DateCoord< coord... >() )
        -> typename GetType<
            typename Mapping::DateDomain,
            coord...
        >::type &
        {
            return access< coord... >(
                std::forward< DateCoord<coord... > >( dc )
            );
        }

        typename Mapping::UserDomain const userDomainPos;
        View<
            Mapping,
            BlobType
        > & view;
    };

    auto
    operator()( typename Mapping::UserDomain const ud )
    -> VirtualDate
    {
        return VirtualDate{
                ud,
                *this
            };
    };

    template< typename... TCoord >
    auto
    operator()( TCoord... coord )
    -> VirtualDate
    {
        return VirtualDate{
                typename Mapping::UserDomain{coord...},
                *this
            };
    };

    BlobType blob[ Mapping::blobCount ];
    const Mapping mapping;
};

} // namespace llama

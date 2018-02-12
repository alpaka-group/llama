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

#include "View.hpp"
#include "allocator/Vector.hpp"

namespace llama
{


template<
    typename Mapping,
    typename Allocator = allocator::Vector
>
struct Factory
{
    static inline
	auto
    allowView ( Mapping const mapping )
    -> View<
        Mapping,
        typename Allocator::BlobType
    >
    {
        View<
            Mapping,
            typename Allocator::BlobType
        > view( mapping );
        for( std::size_t i = 0; i < Mapping::blobCount; ++i )
            view.blob[ i ] = Allocator::allocate( mapping.getBlobSize( i ) );
        return view;
    }
};

} // namespace llama

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
#include "IntegerSequence.hpp"

namespace llama
{

namespace internal
{

template <
    typename T_Allocator,
    typename T_Mapping,
    std::size_t... Is
>
auto
makeBlobArrayImpl(
    T_Mapping const mapping,
    typename T_Allocator::Parameter allocatorParams,
    IntegerSequence<Is...>
)
-> Array<
        typename T_Allocator::BlobType,
        T_Mapping::blobCount
    >
{
    return Array<
        typename T_Allocator::BlobType,
        sizeof...( Is )
    > { T_Allocator::allocate(
            mapping.getBlobSize( Is ),
            allocatorParams
        )...
    };
}

template <
    typename T_Allocator,
    typename T_Mapping
>
auto
makeBlobArray(
    T_Mapping const mapping,
    typename T_Allocator::Parameter allocatorParams
)
-> Array<
        typename T_Allocator::BlobType,
        T_Mapping::blobCount
    >
{
    return makeBlobArrayImpl<
        T_Allocator,
        T_Mapping
    > (
        mapping,
        allocatorParams,
        MakeIntegerSequence< T_Mapping::blobCount >{ }
    );
}

}; // namespace internal

template<
    typename T_Mapping,
    typename T_Allocator = allocator::Vector
>
struct Factory
{
    static inline
    auto
    allowView(
        T_Mapping const mapping,
        typename T_Allocator::Parameter allocatorParams =
            typename T_Allocator::Parameter()
    )
    -> View<
        T_Mapping,
        typename T_Allocator::BlobType
    >
    {
        View<
            T_Mapping,
            typename T_Allocator::BlobType
        > view(
            mapping,
            internal::makeBlobArray< T_Allocator >(
                mapping,
                allocatorParams
            )
        );

        return view;
    }
};

} // namespace llama

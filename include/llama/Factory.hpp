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

#include "IntegerSequence.hpp"
#include "allocator/Stack.hpp"
#include "allocator/Vector.hpp"
#include "mapping/One.hpp"

namespace llama
{
    template<typename Mapping, typename BlobType>
    struct View;

    /** Creates views with the help of mapping and allocation functors. Should
     * be the preferred way to create a \ref View. \tparam Mapping Mapping
     * type. A mapping binds the user domain and datum domain and needs to
     * expose them as typedefs called `UserDomain` and `DatumDomain`.
     * Furthermore it has to define a `static constexpr` called `blobCount` with
     * the number of needed memory regions to allocate. Furthermore three
     * methods need to be defined (Note: For working with offloading device some
     * further annotations for these methods are needed. Best is to have a look
     * at \ref mapping::AoS, \ref mapping::SoA or \ref mapping::One for the
     * exactly implemenatation details):
     *  - `auto getBlobSize( std::size_t ) -> std::size_t` which returns the
     * needed size in byte per blob
     *  - `template< std::size_t... > auto getBlobByte( UserDomain ) ->
     * std::size_t` which returns the byte position for a given coordinate in
     * the datum domain (template parameter) and user domain (method parameter).
     *  - `template< std::size_t... > auto getBlobNr( UserDomain ) ->
     * std::size_t` which returns the blob in which the byte position given by
     * getBlobByte resides.
     */
    template<typename Mapping>
    class Factory
    {
        template<typename Allocator>
        using AllocatorBlobType
            = decltype(std::declval<Allocator>().allocate(0));

        template<typename Allocator, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE static auto makeBlobArray(
            const Allocator & alloc,
            Mapping mapping,
            std::integer_sequence<std::size_t, Is...>)
            -> Array<AllocatorBlobType<Allocator>, Mapping::blobCount>
        {
            return {alloc.allocate(mapping.getBlobSize(Is))...};
        }

    public:
        template<typename Allocator = allocator::Vector<>>
        LLAMA_FN_HOST_ACC_INLINE static auto
        allocView(Mapping mapping = {}, const Allocator & alloc = {})
            -> View<Mapping, AllocatorBlobType<Allocator>>
        {
            return {
                mapping,
                makeBlobArray<Allocator>(
                    alloc,
                    mapping,
                    std::make_index_sequence<Mapping::blobCount>{})};
        }
    };


    /** Uses the \ref OneOnStackFactory to allocate one (probably temporary)
     * element for a given dimension and datum domain on the stack (no costly
     * allocation). \tparam Dimension dimension of the view \tparam DatumDomain
     * the datum domain for the one element mapping \return the allocated view
     * \see OneOnStackFactory
     */
    template<std::size_t Dim, typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto stackViewAlloc() -> decltype(auto)
    {
        using Mapping = llama::mapping::One<UserDomain<Dim>, DatumDomain>;
        return llama::Factory<Mapping>::allocView(
            Mapping{}, llama::allocator::Stack<SizeOf<DatumDomain>::value>{});
    }
}

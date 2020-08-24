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
    namespace internal
    {
        template<typename Allocator, typename... AllocatorArgs>
        using AllocatorBlobType = decltype(
            Allocator::allocate(0, std::declval<AllocatorArgs>()...));

        LLAMA_NO_HOST_ACC_WARNING
        template<
            typename Allocator,
            typename Mapping,
            std::size_t... Is,
            typename... AllocatorArgs>
        LLAMA_FN_HOST_ACC_INLINE auto makeBlobArray(
            Mapping const mapping,
            std::integer_sequence<std::size_t, Is...>,
            AllocatorArgs &&... allocatorArgs)
            -> Array<
                AllocatorBlobType<Allocator, AllocatorArgs...>,
                Mapping::blobCount>
        {
            return {Allocator::allocate(
                mapping.getBlobSize(Is),
                std::forward<AllocatorArgs>(allocatorArgs)...)...};
        }
    }

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
     * getBlobByte resides. \tparam Allocator Allocator type, at default \ref
     * allocator::Vector. An allocator also needs to define some typedefs,
     * namely `PrimType` which is the raw datatype returned from the allocator
     * (e.g. `unsigned char`), `BlobType` which is the type returned from the
     * allocator (can be a pointer, a `std::shared_ptr`, an own class, you name
     * it!) and `Parameter` which is the (optional) allocation parameter type
     * forwarded from the \ref Factory to the allocator. Be aware that at the
     * moment only an 8 bit `PrimType` is supported. Beside these definitions
     * only the method
     *  - `static inline auto allocate( std::size_t, Parameter ) -> BlobType`
     * needs to be implements which allocates memory and returns its self
     * defined blob type.
     */
    template<typename Mapping, typename Allocator = allocator::Vector<>>
    struct Factory
    {
        LLAMA_NO_HOST_ACC_WARNING
        template<typename... AllocatorArgs>
        static LLAMA_FN_HOST_ACC_INLINE auto
        allocView(Mapping const mapping = {}, AllocatorArgs &&... allocatorArgs)
            -> View<
                Mapping,
                internal::AllocatorBlobType<Allocator, AllocatorArgs...>>
        {
            return {
                mapping,
                internal::makeBlobArray<Allocator>(
                    mapping,
                    std::make_index_sequence<Mapping::blobCount>{},
                    std::forward<AllocatorArgs>(allocatorArgs)...)};
        }
    };

    /** Special factory which predefines some options for getting a \ref View
     * with only one element laying on the stack avoiding costly allocation
     * operations. \tparam dimension dimension of the view \tparam DatumDomain
     * the datum domain for the one element mapping \see stackViewAlloc
     */
    template<std::size_t Dimension, typename DatumDomain>
    using OneOnStackFactory = llama::Factory<
        llama::mapping::One<UserDomain<Dimension>, DatumDomain>,
        llama::allocator::Stack<SizeOf<DatumDomain>::value>>;

    /** Uses the \ref OneOnStackFactory to allocate one (probably temporary)
     * element for a given dimension and datum domain on the stack (no costly
     * allocation). \tparam Dimension dimension of the view \tparam DatumDomain
     * the datum domain for the one element mapping \return the allocated view
     * \see OneOnStackFactory
     */
    template<std::size_t Dimension, typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto stackViewAlloc() -> decltype(auto)
    {
        return OneOnStackFactory<Dimension, DatumDomain>::allocView();
    }
}

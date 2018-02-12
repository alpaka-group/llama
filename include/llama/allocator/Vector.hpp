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

#include <vector>
#include <stdlib.h>
#include <malloc.h>

namespace llama
{

namespace allocator
{

namespace internal
{

    template<typename T, std::size_t N = 16>
    struct AlignmentAllocator
    {
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        using pointer = T *;
        using const_pointer = T const *;

        using reference = T &;
        using const_reference = T const &;

        inline AlignmentAllocator() throw () { }

        template< typename T2 >
        inline
        AlignmentAllocator(
            AlignmentAllocator<
                T2,
                N
            > const &
            ) throw () { }

        inline
        ~AlignmentAllocator() throw () { }

        inline
        auto
        adress( reference r )
        -> pointer
        {
            return &r;
        }

        inline
        auto
        adress( const_reference r ) const
        -> const_pointer
        {
            return &r;
        }

        inline
        auto
        allocate( size_type n )
        -> pointer
        {
#            if defined _MSC_VER
                return reinterpret_cast< pointer >(
                    _aligned_malloc(
                        n * sizeof( value_type ),
                        N
                    )
                );
#            elif defined __linux__
                return reinterpret_cast< pointer >(
                    memalign(
                        N,
                        n * sizeof( value_type )
                    )
                );
#            elif defined __MACH__      // Mac OS X
                return reinterpret_cast< pointer >(
                    malloc(
                        n * sizeof( value_type )
                    )
                ); // malloc is always 16 byte aligned on Mac.
#            else
                return reinterpret_cast< pointer >(
                    malloc(
                        n * sizeof( value_type )
                    )
                ); // other (use valloc for page-aligned memory)
#            endif
        }

        inline
        auto
        deallocate(
            pointer p,
            size_type
        )
        -> void
        {
#            if defined _MSC_VER
                _aligned_free( p );
#            elif defined __linux__
                free( p );
#            elif defined __MACH__
                free( p );
#            else
                free( p );
#            endif
        }

        inline
        auto
        construct(
            pointer p,
            value_type const &
        )
        -> void
        {
            /* commented out for performance reasons
            /* new ( p ) value_type ( value );
             */
        }

        inline
        auto
        destroy( pointer p )
        -> void
        {
            p->~value_type();
        }

        inline
        auto
        max_size() const throw ()
        -> size_type
        {
            return size_type( -1 ) / sizeof( value_type );
        }

        template< typename T2 >
        struct rebind
        {
            using other = AlignmentAllocator<
                T2,
                N
            >;
        };

        auto
        operator!=(
            const AlignmentAllocator<
                T,
                N
            > & other
        ) const
        -> bool
        {
            return !( *this == other );
        }

        /* Returns true if and only if storage allocated from *this
         * can be deallocated from other, and vice versa.
         * Always returns true for stateless allocators.
         */
        auto
        operator==(
            const AlignmentAllocator<
                T,
                N
            > & other
        ) const
        -> bool
        {
            return true;
        }
    };

} // namespace internal

struct Vector
{
    using PrimType = unsigned char;
    using BlobType = std::vector<
        PrimType,
        internal::AlignmentAllocator<
            PrimType,
            64
        >
    >;
    static inline
    auto
    allocate( std::size_t count )
    -> BlobType
    {
        return BlobType( count );
    }
};

} // namespace allocator

} // namespace llama

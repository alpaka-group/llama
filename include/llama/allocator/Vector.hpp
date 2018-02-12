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
        public:
            using value_type = T;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;

            using pointer = T *;
            using const_pointer = const T *;

            using reference = T &;
            using const_reference = const T &;

        public:
            inline AlignmentAllocator() throw () { }

            template< typename T2 >
            inline AlignmentAllocator(
				const AlignmentAllocator<
					T2,
					N
				> &
				) throw () { }

            inline ~AlignmentAllocator() throw () { }

            inline pointer adress( reference r )
            {
                return &r;
            }

            inline const_pointer adress( const_reference r ) const
            {
                return &r;
            }

            inline pointer allocate( size_type n )
            {
                #if defined _MSC_VER
                    return reinterpret_cast< pointer >(
						_aligned_malloc(
							n * sizeof( value_type ),
							N
						)
					);
                #elif defined __linux__
                    return reinterpret_cast< pointer >(
						memalign(
							N,
							n * sizeof( value_type )
						)
					);
                #elif defined __MACH__      // Mac OS X
                    return reinterpret_cast< pointer >(
						malloc(
							n * sizeof( value_type )
						)
					); // malloc is always 16 byte aligned on Mac.
                #else
                    return reinterpret_cast< pointer >(
						malloc(
							n * sizeof( value_type )
						)
					); // other (use valloc for page-aligned memory)
                #endif
            }

            inline void deallocate(
				pointer p,
				size_type
			)
            {
                #if defined _MSC_VER
                    _aligned_free( p );
                #elif defined __linux__
                    free( p );
                #elif defined __MACH__
                    free( p );
                #else
                    free( p );
                #endif
            }

            inline void construct(
				pointer p,
				const value_type &
			)
            {
				/* commented out for performance reasons
                /* new ( p ) value_type ( value );
                 */
            }

            inline void destroy( pointer p )
            {
                p->~value_type();
            }

            inline size_type max_size() const throw ()
            {
                return size_type (-1) / sizeof( value_type );
            }

            template< typename T2 >
            struct rebind
            {
                using other = AlignmentAllocator<
					T2,
					N
				>;
            };

            bool operator!=( const AlignmentAllocator<
					T,
					N
				>& other
			) const
            {
                return !( *this == other );
            }

            // Returns true if and only if storage allocated from *this
            // can be deallocated from other, and vice versa.
            // Always returns true for stateless allocators.
            bool operator==(const AlignmentAllocator<
					T,
					N
				>& other
			) const
            {
                return true;
            }
    };

} //internal

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
    static inline BlobType allocate( size_t count )
    {
        return BlobType( count );
    }
};

} //namespace allocator

} //namespace llama

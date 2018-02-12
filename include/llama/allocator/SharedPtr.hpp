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

#include <memory>

namespace llama
{

namespace allocator
{

namespace internal
{

struct SharedPtrAccessor
{
    using PrimType = unsigned char;
    using BlobType = std::shared_ptr< PrimType >;
    // SharedPtrAccessor( BlobType blob ) : blob( blob ) {}
    template< typename IndexType >
    PrimType& operator[] ( IndexType && idx )
    {
        return blob.get()[ idx ];
    }
    template<typename IndexType>
    const PrimType& operator[] ( IndexType && idx ) const
    {
        return blob.get()[ idx ];
    }
    BlobType blob;
};

} //namespace internal

template<size_t alignment = 64u>
struct SharedPtr
{
    using PrimType = typename internal::SharedPtrAccessor::PrimType;
    using BlobType = internal::SharedPtrAccessor;
    static inline BlobType allocate( std::size_t count )
    {
		#if defined _MSC_VER
			PrimType* raw_pointer = reinterpret_cast< PrimType* >(
				_aligned_malloc(
					count * sizeof( PrimType ),
					alignment
				)
			);
		#elif defined __linux__
			PrimType* raw_pointer = reinterpret_cast< PrimType* >(
				memalign(
					alignment,
					count * sizeof( PrimType )
				)
			);
		#elif defined __MACH__      // Mac OS X
			PrimType* raw_pointer = reinterpret_cast< PrimType* >(
				malloc(
					count * sizeof( PrimType )
				)
			); // malloc is always 16 byte aligned on Mac.
		#else
			PrimType* raw_pointer = reinterpret_cast< PrimType* >(
				malloc(
					count * sizeof( PrimType )
				)
			); // other (use valloc for page-aligned memory)
		#endif
        BlobType accessor;
        accessor.blob = internal::SharedPtrAccessor::BlobType(
			raw_pointer,
			[=]( PrimType* raw_pointer )
            {
                #if defined _MSC_VER
                    _aligned_free( raw_pointer );
                #elif defined __linux__
                    free( raw_pointer );
                #elif defined __MACH__
                    free( raw_pointer );
                #else
                    free( raw_pointer );
                #endif
            }
        );
        return accessor;
    }
};

} //namespace allocator

} //namespace llama

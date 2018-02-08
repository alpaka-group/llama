#pragma once

#include <vector>

namespace llama
{

#include <stdlib.h>
#include <malloc.h>

namespace internal
{

	template <typename T, std::size_t N = 16>
	struct AlignmentAllocator
	{
		public:
			typedef T value_type;
			typedef std::size_t size_type;
			typedef std::ptrdiff_t difference_type;

			typedef T * pointer;
			typedef const T * const_pointer;

			typedef T & reference;
			typedef const T & const_reference;

		public:
			inline AlignmentAllocator () throw () { }

			template <typename T2>
			inline AlignmentAllocator (const AlignmentAllocator<T2, N> &) throw () { }

			inline ~AlignmentAllocator () throw () { }

			inline pointer adress (reference r)
			{
				return &r;
			}

			inline const_pointer adress (const_reference r) const
			{
				return &r;
			}

			inline pointer allocate (size_type n)
			{
				#if defined _MSC_VER
					return (pointer)_aligned_malloc(n*sizeof(value_type), N);
				#elif defined __linux__
					return (pointer)memalign(N, n*sizeof(value_type));
				#elif defined __MACH__      // Mac OS X
					return (pointer)malloc(n*sizeof(value_type));    // malloc is always 16 byte aligned on Mac.
				#else
					return (pointer)valloc(n*sizeof(value_type));    // other (use valloc for page-aligned memory)
				#endif
			}

			inline void deallocate (pointer p, size_type)
			{
				#if defined _MSC_VER
					_aligned_free(p);
				#elif defined __linux__
					free(p);
				#elif defined __MACH__
					free(p);
				#else
					free(p);
				#endif
			}

			inline void construct (pointer p, const value_type &)
			{
				//new (p) value_type (wert);
			}

			inline void destroy (pointer p)
			{
				p->~value_type ();
			}

			inline size_type max_size () const throw ()
			{
				return size_type (-1) / sizeof (value_type);
			}

			template <typename T2>
			struct rebind
			{
				typedef AlignmentAllocator<T2, N> other;
			};

			bool operator!=(const AlignmentAllocator<T,N>& other) const
			{
				return !(*this == other);
			}

			// Returns true if and only if storage allocated from *this
			// can be deallocated from other, and vice versa.
			// Always returns true for stateless allocators.
			bool operator==(const AlignmentAllocator<T,N>& other) const
			{
				return true;
			}
	};

} //internal

struct VectorAllocator
{
	using PrimType = unsigned char;
	using BlobType = std::vector<PrimType, internal::AlignmentAllocator<PrimType,64> >;
	static inline BlobType allocate(size_t count)
	{
		return BlobType(count);
	}
};

} //namespace llama

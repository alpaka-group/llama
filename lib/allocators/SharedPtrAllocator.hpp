#pragma once

#include <memory>

namespace llama
{

struct SharedPtrAccessor
{
	using PrimType = unsigned char;
	using BlobType = std::shared_ptr<PrimType>;
	//SharedPtrAccessor(BlobType blob) : blob(blob) {}
	template <typename IndexType>
	PrimType& operator[] (IndexType&& idx)
	{
		return blob.get()[idx];
	}
	template <typename IndexType>
	const PrimType& operator[] (IndexType&& idx) const
	{
		return blob.get()[idx];
	}
	BlobType blob;
};

template <size_t alignment = 64u>
struct SharedPtrAllocator
{
	using PrimType = typename SharedPtrAccessor::PrimType;
	using BlobType = SharedPtrAccessor;
	static inline BlobType allocate(size_t count)
	{
		#if defined _MSC_VER
			PrimType* raw_pointer = (PrimType*)_aligned_malloc(count*sizeof(PrimType), alignment);
		#elif defined __linux__
			PrimType* raw_pointer = (PrimType*)memalign(alignment, count*sizeof(PrimType));
		#elif defined __MACH__      // Mac OS X
			PrimType* raw_pointer = (PrimType*)malloc(count*sizeof(PrimType));    // malloc is always 16 byte aligned on Mac.
		#else
			PrimType* raw_pointer = (PrimType*)valloc(count*sizeof(PrimType));    // other (use valloc for page-aligned memory)
		#endif
		BlobType accessor;
		accessor.blob = SharedPtrAccessor::BlobType(raw_pointer,[=](PrimType* raw_pointer)
			{
				#if defined _MSC_VER
					_aligned_free(raw_pointer);
				#elif defined __linux__
					free(raw_pointer);
				#elif defined __MACH__
					free(raw_pointer);
				#else
					free(raw_pointer);
				#endif
			}
		);
		return accessor;
	}
};

} //namespace llama

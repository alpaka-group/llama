#pragma once

#include "View.hpp"
#include "allocators/VectorAllocator.hpp"

namespace llama
{


template <
	typename Mapping,
	typename Allocator = VectorAllocator
>
struct Factory
{
	static inline View<Mapping,typename Allocator::BlobType> allowView (
		const Mapping mapping
	)
	{
		View<Mapping,typename Allocator::BlobType> view(mapping);
		for (size_t i = 0; i < Mapping::blobCount; ++i)
			view.blob[i] = Allocator::allocate(mapping.getBlobSize(i));
		return view;
	}
};

} //namespace llama

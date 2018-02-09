#pragma once

#include "GetType.hpp"

namespace llama
{

template <
	typename __Mapping,
	typename __BlobType
>
struct View
{
	using BlobType = __BlobType;
	using Mapping = __Mapping;
	View(Mapping mapping) :
		mapping(mapping)
	{
	}
	template <size_t... dd>
	typename GetType<typename Mapping::DateDomain,dd...>::type&
	accessor( typename Mapping::UserDomain const ud )
	{
		auto const nr = mapping.template getBlobNr<dd...>(ud);
		auto const byte = mapping.template getBlobByte<dd...>(ud);
		return *(
			reinterpret_cast<typename GetType<typename Mapping::DateDomain,dd...>::type*> (
				&blob[nr][byte]
			)
		);
	}

	struct VirtualDate
	{
		template <size_t... coord>
		typename GetType<typename Mapping::DateDomain,coord...>::type&
		access( DateCoord<coord...>&& = DateCoord<coord...>() )
		{
			return view.accessor<coord...>(userDomainPos);
		}

		template <size_t... coord>
		typename GetType<typename Mapping::DateDomain,coord...>::type&
		operator()(DateCoord<coord...>&& dc= DateCoord<coord...>())
		{
			return access<coord...>(std::forward<DateCoord<coord...>>(dc));
		}

		typename Mapping::UserDomain const userDomainPos;
		View<Mapping,BlobType>& view;
	};

	VirtualDate operator()(typename Mapping::UserDomain const ud)
	{
		return VirtualDate{ud,*this};
	};

	template <typename... TCoord>
	VirtualDate operator()(TCoord... coord)
	{
		return VirtualDate{typename Mapping::UserDomain{coord...},*this};
	};

	BlobType blob[Mapping::blobCount];
	const Mapping mapping;
};

} //namespace llama

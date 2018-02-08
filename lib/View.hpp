#pragma once

#include "GetType.hpp"

namespace llama
{

template <
	typename Mapping,
	typename __BlobType
>
struct View
{
	using BlobType = __BlobType;
	View(Mapping mapping) :
		mapping(mapping)
	{
	}
	template <size_t... dd>
	typename GetType<typename Mapping::DateDomain,dd...>::type& accessor(
		typename Mapping::UserDomain const ud
	)
	{
		auto const adress = mapping.template getBlobAdress<dd...>(ud);
		return *(
			reinterpret_cast<typename GetType<typename Mapping::DateDomain,dd...>::type*> (
				&blob[adress.blobNr][adress.bytePos]
			)
		);
	}

	struct VirtualDate
	{
		template <size_t... coord>
		typename GetType<typename Mapping::DateDomain,coord...>::type& access(
			DateCoord<coord...>&& = DateCoord<coord...>()
		)
		{
			return view.accessor<coord...>(userDomainPos);
		}
		template<size_t... coord>
		auto operator()(DateCoord<coord...>&& = DateCoord<coord...>())
		-> decltype(access<coord...>())&
		{
			return access<coord...>();
		}
		typename Mapping::UserDomain const userDomainPos;
		View<Mapping,BlobType>& view;
	};

	VirtualDate operator()(typename Mapping::UserDomain const ud)
	{
		return VirtualDate{ud,*this};
	};

	BlobType blob[Mapping::blobCount];
	const Mapping mapping;
};

} //namespace llama

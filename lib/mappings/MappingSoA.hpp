#pragma once

#include "../Types.hpp"
#include "../GetType.hpp"
#include "../UserDomain.hpp"

namespace llama
{

template<
	typename __UserDomain,
	typename __DateDomain,
	typename LinearizeUserDomainAdressFunctor = LinearizeUserDomainAdress<__UserDomain::count>,
	typename ExtentUserDomainAdressFunctor = ExtentUserDomainAdress<__UserDomain::count>
>
struct MappingSoA
{
	using UserDomain = __UserDomain;
	using DateDomain = __DateDomain;
	MappingSoA(const UserDomain size) : userDomainSize(size) {}
	static constexpr size_t blobCount = 1;
	inline size_t getBlobSize(const size_t) const
	{
		return ExtentUserDomainAdressFunctor()(userDomainSize) * DateDomain::size;
	}
	template <size_t... dateDomainCoord>
	inline BlobAdress getBlobAdress(const UserDomain coord) const
	{
		return BlobAdress
		{
			0,
			LinearizeUserDomainAdressFunctor()(coord,userDomainSize)
			* sizeof(typename GetType<DateDomain,dateDomainCoord...>::type)
			+ DateDomain::template LinearBytePos<dateDomainCoord...>::value
			* ExtentUserDomainAdressFunctor()(userDomainSize)
		};
	}
	const UserDomain userDomainSize;
};

} //namespace llama


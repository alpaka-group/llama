#pragma once

#include "../Types.hpp"
#include "../UserDomain.hpp"

namespace llama
{

template<
	typename __UserDomain,
	typename __DateDomain,
	typename LinearizeUserDomainAdressFunctor = LinearizeUserDomainAdress<__UserDomain::count>,
	typename ExtentUserDomainAdressFunctor = ExtentUserDomainAdress<__UserDomain::count>
>
struct MappingAoS
{
	using UserDomain = __UserDomain;
	using DateDomain = __DateDomain;
	MappingAoS(const UserDomain size) : userDomainSize(size) {}
	static constexpr size_t blobCount = 1;
	inline size_t getBlobSize(const size_t) const
	{
		return ExtentUserDomainAdressFunctor()(userDomainSize) * DateDomain::size;
	}
	template <size_t... dateDomainCoord>
	inline size_t getBlobByte(const UserDomain coord) const
	{
		return LinearizeUserDomainAdressFunctor()(coord,userDomainSize)
			* DateDomain::size
			+ DateDomain::template LinearBytePos<dateDomainCoord...>::value;
	}
	template <size_t... dateDomainCoord>
	constexpr size_t getBlobNr(const UserDomain coord) const
	{
		return 0;
	}
	const UserDomain userDomainSize;
};

} //namespace llama


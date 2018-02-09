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
	MappingSoA(const UserDomain size) :
		userDomainSize(size),
		extentUserDomainAdress(ExtentUserDomainAdressFunctor()(userDomainSize))
	{}
	static constexpr size_t blobCount = 1;
	inline size_t getBlobSize(const size_t) const
	{
		return extentUserDomainAdress * DateDomain::size;
	}
	template <size_t... dateDomainCoord>
	inline size_t getBlobByte(const UserDomain coord) const
	{
		return LinearizeUserDomainAdressFunctor()(coord,userDomainSize) // variabele runtime
			* sizeof(typename GetType<DateDomain,dateDomainCoord...>::type) //constexpr
			+ DateDomain::template LinearBytePos<dateDomainCoord...>::value //constexpr
			* extentUserDomainAdress; //const (runtime)
	}
	template <size_t... dateDomainCoord>
	constexpr size_t getBlobNr(const UserDomain coord) const
	{
		return 0;
	}
	const UserDomain userDomainSize;
	const size_t extentUserDomainAdress;
};

} //namespace llama


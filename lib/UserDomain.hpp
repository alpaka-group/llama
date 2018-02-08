#pragma once

#include "Types.hpp"

namespace llama
{

template< size_t dim >
struct ExtentUserDomainAdress
{
	inline size_t operator()(const UserDomain<dim>& size)
	{
		return ExtentUserDomainAdress<dim-1>()( size.pop_front() ) * size[0];
	}
};

template<>
struct ExtentUserDomainAdress<1>
{
	inline size_t operator()(const UserDomain<1>& size)
	{
		return size[0];
	}
};

template< size_t dim >
struct LinearizeUserDomainAdress
{
	inline size_t operator()(const UserDomain<dim>& coord, const UserDomain<dim>& size) const
	{
		return coord[dim-1] + LinearizeUserDomainAdress<dim-1>()( coord.pop_back(), size.pop_back() ) * size[dim-1];
	}
};

template<>
struct LinearizeUserDomainAdress<1>
{
	inline size_t operator()(const UserDomain<1>& coord,const UserDomain<1>& size)
	{
		return coord[0];
	}
};


template< size_t dim >
struct LinearizeUserDomainAdressLikeFortran
{
	inline size_t operator()(const UserDomain<dim>& coord, const UserDomain<dim>& size) const
	{
		return coord[0] + LinearizeUserDomainAdressLikeFortran<dim-1>()( coord.pop_front(), size.pop_front() ) * size[0];
	}
};

template<>
struct LinearizeUserDomainAdressLikeFortran<1>
{
	inline size_t operator()(const UserDomain<1>& coord,const UserDomain<1>& size)
	{
		return coord[0];
	}
};
} //namespace llama

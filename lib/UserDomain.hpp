#pragma once

#include "Types.hpp"

namespace llama
{

template< size_t dim >
struct ExtentUserDomainAdress
{
	inline size_t operator()(const UserDomain<dim>& size) const
	{
		return ExtentUserDomainAdress<dim-1>()( size.pop_front() ) * size[0];
	}
};

template<>
struct ExtentUserDomainAdress<1>
{
	inline size_t operator()(const UserDomain<1>& size) const
	{
		return size[0];
	}
};

template< size_t dim, size_t it = dim >
struct LinearizeUserDomainAdress
{
	inline size_t operator()(const UserDomain<dim>& coord, const UserDomain<dim>& size) const
	{
		return coord[it-1] + LinearizeUserDomainAdress<dim,it-1>()( coord, size ) * size[it-1];
	}
};

template< size_t dim >
struct LinearizeUserDomainAdress < dim, 1 >
{
	inline size_t operator()(const UserDomain<dim>& coord, const UserDomain<dim>& size) const
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
	inline size_t operator()(const UserDomain<1>& coord,const UserDomain<1>& size) const
	{
		return coord[0];
	}
};
} //namespace llama

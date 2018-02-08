#pragma once

namespace llama
{

template<
	typename Tree,
	size_t firstDateDomainCoord,
	size_t... dateDomainCoords
>
struct GetType
{
	using type = typename GetType<
		typename Tree::template GetBranch<firstDateDomainCoord>::type,
		dateDomainCoords...
	>::type;
};

template<
	typename Tree,
	size_t firstDateDomainCoord
>
struct GetType<Tree,firstDateDomainCoord>
{
	using type = typename Tree::template GetBranch<firstDateDomainCoord>::type;
};

} //namespace llama

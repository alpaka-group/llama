/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

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

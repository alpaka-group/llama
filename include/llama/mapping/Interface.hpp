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

#include "../Types.hpp"

namespace llama
{

namespace mapping
{

template<
	typename __UserDomain,
	typename __DateDomain
>
struct Interface
{
	using UserDomain = __UserDomain;
	using DateDomain = __DateDomain;
	Interface(const UserDomain);
	static constexpr size_t blobCount = 0;
	inline size_t getBlobSize(const size_t blobNr) const;
	template <typename DateDomainCoord>
	inline size_t getBlobByte(const UserDomain coord,const UserDomain size) const;
	inline size_t getBlobNr(const UserDomain coord,const UserDomain size) const;
};

} //namespace mapping

} //namespace llama

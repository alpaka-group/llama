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
    typename T_Tree,
    std::size_t T_firstDatumDomainCoord,
    std::size_t... T_datumDomainCoords
>
struct GetType
{
    using type = typename GetType<
        typename T_Tree::template GetBranch< T_firstDatumDomainCoord >::type,
        T_datumDomainCoords...
    >::type;
};

template<
    typename T_Tree,
    std::size_t T_firstDatumDomainCoord
>
struct GetType<
    T_Tree,
    T_firstDatumDomainCoord
>
{
    using type =
        typename T_Tree::template GetBranch< T_firstDatumDomainCoord >::type;
};

} // namespace llama

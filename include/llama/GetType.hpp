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

#include "Functions.hpp"

#include <boost/mp11.hpp>

namespace llama
{
    namespace internal
    {
        template<typename DatumDomain, typename T_DatumCoord>
        struct GetTypeImpl;

        template<
            typename DatumDomain,
            std::size_t HeadCoord,
            std::size_t... TailCoords>
        struct GetTypeImpl<DatumDomain, DatumCoord<HeadCoord, TailCoords...>>
        {
            using _DateElement = boost::mp11::mp_at_c<DatumDomain, HeadCoord>;
            using type = typename GetTypeImpl<
                GetDatumElementType<_DateElement>,
                DatumCoord<TailCoords...>>::type;
        };

        template<typename T>
        struct GetTypeImpl<T, DatumCoord<>>
        {
            using type = T;
        };
    }

    /** Returns the type of a node in a datum domain tree for a coordinate given
     * as \ref DatumCoord \tparam DatumDomain the datum domain (probably \ref
     * DatumStruct) \tparam T_DatumCoord the coordinate \return type at the
     * specified node
     */
    template<typename DatumDomain, typename T_DatumCoord>
    using GetType =
        typename internal::GetTypeImpl<DatumDomain, T_DatumCoord>::type;
}

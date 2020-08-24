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

#include "DatumCoord.hpp"
#include "Functions.hpp"

#include <type_traits>

namespace llama
{
    namespace internal
    {
        template<
            typename DatumDomain,
            typename BaseDatumCoord,
            std::size_t... InnerCoords,
            typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(
            DatumDomain,
            BaseDatumCoord base,
            DatumCoord<InnerCoords...> inner,
            Functor && functor)
        {
            using InnerDatumCoord = decltype(inner);
            if constexpr(
                InnerDatumCoord::size >= BaseDatumCoord::size
                && DatumCoordIsSame<InnerDatumCoord, BaseDatumCoord>::value)
                functor(
                    BaseDatumCoord{},
                    DatumCoordFromList<boost::mp11::mp_drop_c<
                        typename InnerDatumCoord::List,
                        BaseDatumCoord::size>>{});
        };

        template<
            typename... Leaves,
            typename BaseDatumCoord,
            std::size_t... InnerCoords,
            typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(
            DatumStruct<Leaves...>,
            BaseDatumCoord base,
            DatumCoord<InnerCoords...> inner,
            Functor && functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            boost::mp11::mp_for_each<
                boost::mp11::mp_iota_c<sizeof...(Leaves)>>([&](auto i) {
                constexpr auto leafIndex = decltype(i)::value;
                using Leaf
                    = boost::mp11::mp_at_c<DatumStruct<Leaves...>, leafIndex>;

                LLAMA_FORCE_INLINE_RECURSIVE
                applyFunctorForEachLeaf(
                    GetDatumElementType<Leaf>{},
                    base,
                    llama::DatumCoord<InnerCoords..., leafIndex>{},
                    std::forward<Functor>(functor));
            });
        }
    }

    /** Can be used to access a given functor for every leaf in a datum domain
     * given as \ref DatumStruct. Basically a helper function to iterate over a
     * datum domain at compile time without the need to recursively iterate
     * yourself. The given functor needs to implement the operator() with two
     * template parameters for the outer and the inner coordinate in the datum
     * domain tree. These coordinates are both a \ref DatumCoord , which can be
     * concatenated to one coordinate with \ref DatumCoord::Cat and used to
     * access the data. \tparam DatumDomain the datum domain (\ref
     * DatumStruct) to iterate over \tparam DatumCoordOrFirstUID DatumCoord or
     * a UID to address the start node inside the datum domain tree. Will be
     * given to the functor as \ref DatumCoord as first template parameter.
     * \tparam RestUID... optional further UIDs for addressing the start node
     */
    template<
        typename DatumDomain,
        typename DatumCoordOrFirstUID = DatumCoord<>,
        typename... RestUID>
    struct ForEach :
            ForEach<
                DatumDomain,
                GetCoordFromUID<DatumDomain, DatumCoordOrFirstUID, RestUID...>>
    {};

    template<typename DatumDomain, std::size_t... Coords>
    struct ForEach<DatumDomain, DatumCoord<Coords...>>
    {
        template<typename Functor>
        LLAMA_FN_HOST_ACC_INLINE static void apply(Functor && functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            internal::applyFunctorForEachLeaf(
                DatumDomain{},
                llama::DatumCoord<Coords...>{},
                DatumCoord<>{},
                std::forward<Functor>(functor));
        }
    };
}

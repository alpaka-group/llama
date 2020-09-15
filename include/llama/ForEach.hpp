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
            typename T,
            typename BaseDatumCoord,
            std::size_t... InnerCoords,
            typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(
            T,
            BaseDatumCoord base,
            DatumCoord<InnerCoords...> inner,
            Functor && functor)
        {
            using InnerDatumCoord = decltype(inner);
            if constexpr(
                InnerDatumCoord::size >= BaseDatumCoord::size
                && DatumCoordIsSame<InnerDatumCoord, BaseDatumCoord>)
                functor(
                    base,
                    DatumCoordFromList<boost::mp11::mp_drop_c<
                        typename InnerDatumCoord::List,
                        BaseDatumCoord::size>>{});
        };

        template<
            typename... Children,
            typename BaseDatumCoord,
            std::size_t... InnerCoords,
            typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(
            DatumStruct<Children...>,
            BaseDatumCoord base,
            DatumCoord<InnerCoords...> inner,
            Functor && functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            boost::mp11::mp_for_each<
                boost::mp11::mp_iota_c<sizeof...(Children)>>([&](auto i) {
                constexpr auto childIndex = decltype(i)::value;
                using DatumElement = boost::mp11::
                    mp_at_c<DatumStruct<Children...>, childIndex>;

                LLAMA_FORCE_INLINE_RECURSIVE
                applyFunctorForEachLeaf(
                    GetDatumElementType<DatumElement>{},
                    base,
                    llama::DatumCoord<InnerCoords..., childIndex>{},
                    std::forward<Functor>(functor));
            });
        }
    }

    /// Iterates over the datum domain tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have
    /// `operator()` with two template parameters for the base and the inner
    /// coordinate in the datum domain tree, both will be a spezialization of
    /// \ref DatumCoord.
    /// \param base \ref DatumCoord at which the iteration should be started.
    /// The functor is called on elements beneath this coordinate.
    template<typename DatumDomain, typename Functor, std::size_t... Coords>
    LLAMA_FN_HOST_ACC_INLINE void
    forEach(Functor && functor, DatumCoord<Coords...> base)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        internal::applyFunctorForEachLeaf(
            DatumDomain{},
            base,
            DatumCoord<>{},
            std::forward<Functor>(functor));
    }

    /// Iterates over the datum domain tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have
    /// `operator()` with two template parameters for the base and the inner
    /// coordinate in the datum domain tree, both will be a spezialization of
    /// \ref DatumCoord.
    /// \param baseTags Tags used to define where the iteration should be
    /// started. The functor is called on elements beneath this coordinate.
    template<typename DatumDomain, typename Functor, typename... Tags>
    LLAMA_FN_HOST_ACC_INLINE void forEach(Functor && functor, Tags... baseTags)
    {
        forEach<DatumDomain>(
            std::forward<Functor>(functor),
            GetCoordFromTags<DatumDomain, Tags...>{});
    }
}

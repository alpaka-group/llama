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
            typename Coord,
            typename Pos,
            typename Functor,
            typename... Leaves>
        struct ApplyFunctorForEachLeafImpl;

        template<typename Coord, typename Pos, typename Functor, typename Leaf>
        struct ApplyFunctorForDatumDomainImpl
        {
            LLAMA_FN_HOST_ACC_INLINE void operator()(Functor && functor)
            {
                if constexpr(
                    Pos::size >= Coord::size
                    && DatumCoordIsSame<Pos, Coord>::value)
                    functor(
                        Coord{},
                        typename Pos::template Back<Pos::size - Coord::size>());
            };
        };

        template<
            typename Coord,
            typename Pos,
            typename Functor,
            typename... Leaves>
        struct ApplyFunctorForDatumDomainImpl<
            Coord,
            Pos,
            Functor,
            DatumStruct<Leaves...>>
        {
            LLAMA_FN_HOST_ACC_INLINE void operator()(Functor && functor)
            {
                ApplyFunctorForEachLeafImpl<
                    Coord,
                    typename Pos::template PushBack<0>,
                    Functor,
                    Leaves...>{}(std::forward<Functor>(functor));
            }
        };

        template<
            typename Coord,
            typename Pos,
            typename Functor,
            typename Leaf,
            typename... Leaves>
        struct ApplyFunctorForEachLeafImpl<Coord, Pos, Functor, Leaf, Leaves...>
        {
            LLAMA_FN_HOST_ACC_INLINE void operator()(Functor && functor)
            {
                ApplyFunctorForDatumDomainImpl<
                    Coord,
                    Pos,
                    Functor,
                    GetDatumElementType<Leaf>>{}(
                    std::forward<Functor>(functor));
                if constexpr(sizeof...(Leaves) > 0)
                    ApplyFunctorForEachLeafImpl<
                        Coord,
                        typename Pos::IncBack,
                        Functor,
                        Leaves...>{}(std::forward<Functor>(functor));
            }
        };

        template<typename DatumDomain, typename DatumCoord, typename Functor>
        struct ApplyFunctorForEachLeaf;

        template<typename DatumCoord, typename Functor, typename... Leaves>
        struct ApplyFunctorForEachLeaf<
            DatumStruct<Leaves...>,
            DatumCoord,
            Functor>
        {
            LLAMA_FN_HOST_ACC_INLINE static void apply(Functor && functor)
            {
                ApplyFunctorForEachLeafImpl<
                    DatumCoord,
                    llama::DatumCoord<0>,
                    Functor,
                    Leaves...>{}(std::forward<Functor>(functor));
            }
        };
    }

    /** Can be used to access a given functor for every leaf in a datum domain
     * given as \ref DatumStruct. Basically a helper function to iterate over a
     * datum domain at compile time without the need to recursively iterate
     * yourself. The given functor needs to implement the operator() with two
     * template parameters for the outer and the inner coordinate in the datum
     * domain tree. These coordinates are both a \ref DatumCoord , which can be
     * concatenated to one coordinate with \ref DatumCoord::Cat and used to
     * access the data. \tparamDatumDomain the datum domain (\ref
     * DatumStruct) to iterate over \tparamDatumCoordOrFirstUID DatumCoord or
     * a UID to address the start node inside the datum domain tree. Will be
     * given to the functor as \ref DatumCoord as first template parameter.
     * \tparamRestUID... optional further UIDs for addressing the start node
     */
    template<
        typename DatumDomain,
        typename DatumCoordOrFirstUID = DatumCoord<>,
        typename... RestUID>
    struct ForEach
    {
        /** Applies the given functor to the given (part of the) datum domain.
         * \tparamFunctor type of the functor
         * \param functor the perfectly forwarded functor
         */
        template<typename Functor>
        LLAMA_FN_HOST_ACC_INLINE static void apply(Functor && functor)
        {
            using DatumCoord = GetCoordFromUID<
                DatumDomain,
                DatumCoordOrFirstUID,
                RestUID...>;
            internal::ApplyFunctorForEachLeaf<
                DatumDomain,
                DatumCoord,
                Functor>::apply(std::forward<Functor>(functor));
        }
    };

    template<typename DatumDomain, std::size_t... coords>
    struct ForEach<DatumDomain, DatumCoord<coords...>>
    {
        template<typename Functor>
        LLAMA_FN_HOST_ACC_INLINE static void apply(Functor && functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            internal::ApplyFunctorForEachLeaf<
                DatumDomain,
                llama::DatumCoord<coords...>,
                Functor>::apply(std::forward<Functor>(functor));
        }
    };
}

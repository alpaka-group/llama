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
#include "GetCoordFromUID.hpp"

#include <type_traits>

namespace llama
{
    namespace internal
    {
        template<typename T_Pos, typename T_Coord, typename SFINAE = void>
        struct CoordIsIncluded;

        template<typename T_Pos, typename T_Coord>
        struct CoordIsIncluded<
            T_Pos,
            T_Coord,
            std::enable_if_t<(T_Pos::size < T_Coord::size)>>
        {
            static constexpr bool value = false;
        };

        template<typename T_Pos, typename T_Coord>
        struct CoordIsIncluded<
            T_Pos,
            T_Coord,
            std::enable_if_t<
                (T_Pos::size >= T_Coord::size)
                && DatumCoordIsSame<T_Pos, T_Coord>::value>>
        {
            static constexpr bool value = true;
        };

        template<
            typename T_Coord,
            typename T_Pos,
            typename T_Functor,
            typename SFINAE = void>
        struct ApplyFunctorIfCoordIsIncluded
        {
            LLAMA_FN_HOST_ACC_INLINE void
            operator()(T_Functor const & functor){};
        };

        template<typename T_Coord, typename T_Pos, typename T_Functor>
        struct ApplyFunctorIfCoordIsIncluded<
            T_Coord,
            T_Pos,
            T_Functor,
            typename std::enable_if<
                CoordIsIncluded<T_Pos, T_Coord>::value>::type>
        {
            LLAMA_FN_HOST_ACC_INLINE void operator()(T_Functor & functor)
            {
                functor(
                    T_Coord(),
                    typename T_Pos::template Back<
                        T_Pos::size - T_Coord::size>());
            };
        };

        template<
            typename T_Coord,
            typename T_Pos,
            typename T_Functor,
            typename... T_Leaves>
        struct ApplyFunctorForEachLeafImpl;

        template<
            typename T_Coord,
            typename T_Pos,
            typename T_Functor,
            typename T_Leaf>
        struct ApplyFunctorForDatumDomainImpl
        {
            LLAMA_FN_HOST_ACC_INLINE void operator()(T_Functor && functor)
            {
                ApplyFunctorIfCoordIsIncluded<T_Coord, T_Pos, T_Functor>{}(
                    std::forward<T_Functor>(functor));
            };
        };

        template<
            typename T_Coord,
            typename T_Pos,
            typename T_Functor,
            typename... T_Leaves>
        struct ApplyFunctorForDatumDomainImpl<
            T_Coord,
            T_Pos,
            T_Functor,
            DatumStruct<T_Leaves...>>
        {
            LLAMA_FN_HOST_ACC_INLINE void operator()(T_Functor && functor)
            {
                ApplyFunctorForEachLeafImpl<
                    T_Coord,
                    typename T_Pos::template PushBack<0>,
                    T_Functor,
                    T_Leaves...>{}(std::forward<T_Functor>(functor));
            }
        };

        template<
            typename T_Coord,
            typename T_Pos,
            typename T_Functor,
            typename T_Leaf,
            typename... T_Leaves>
        struct ApplyFunctorForEachLeafImpl<
            T_Coord,
            T_Pos,
            T_Functor,
            T_Leaf,
            T_Leaves...>
        {
            LLAMA_FN_HOST_ACC_INLINE auto operator()(T_Functor && functor)
                -> void
            {
                ApplyFunctorForDatumDomainImpl<
                    T_Coord,
                    T_Pos,
                    T_Functor,
                    GetDatumElementType<T_Leaf>>{}(
                    std::forward<T_Functor>(functor));
                ApplyFunctorForEachLeafImpl<
                    T_Coord,
                    typename T_Pos::IncBack,
                    T_Functor,
                    T_Leaves...>{}(std::forward<T_Functor>(functor));
            }
        };

        template<typename T_Coord, typename T_Pos, typename T_Functor>
        struct ApplyFunctorForEachLeafImpl<T_Coord, T_Pos, T_Functor>
        {
            LLAMA_FN_HOST_ACC_INLINE void operator()(T_Functor && functor) {}
        };

        template<
            typename T_DatumDomain,
            typename T_DatumCoord,
            typename T_Functor>
        struct ApplyFunctorForEachLeaf;

        template<
            typename T_DatumCoord,
            typename T_Functor,
            typename... T_Leaves>
        struct ApplyFunctorForEachLeaf<
            DatumStruct<T_Leaves...>,
            T_DatumCoord,
            T_Functor>
        {
            LLAMA_FN_HOST_ACC_INLINE static void apply(T_Functor && functor)
            {
                ApplyFunctorForEachLeafImpl<
                    T_DatumCoord,
                    DatumCoord<0>,
                    T_Functor,
                    T_Leaves...>{}(std::forward<T_Functor>(functor));
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
     * access the data. \tparam T_DatumDomain the datum domain (\ref
     * DatumStruct) to iterate over \tparam T_DatumCoordOrFirstUID DatumCoord or
     * a UID to address the start node inside the datum domain tree. Will be
     * given to the functor as \ref DatumCoord as first template parameter.
     * \tparam T_RestUID... optional further UIDs for addressing the start node
     */
    template<
        typename T_DatumDomain,
        typename T_DatumCoordOrFirstUID = DatumCoord<>,
        typename... T_RestUID>
    struct ForEach
    {
        using T_DatumCoord = GetCoordFromUID<
            T_DatumDomain,
            T_DatumCoordOrFirstUID,
            T_RestUID...>;
        /** Applies the given functor to the given (part of the) datum domain.
         * \tparam T_Functor type of the functor
         * \param functor the perfectly forwarded functor
         */
        template<typename T_Functor>
        LLAMA_FN_HOST_ACC_INLINE static void apply(T_Functor && functor)
        {
            internal::ApplyFunctorForEachLeaf<
                T_DatumDomain,
                T_DatumCoord,
                T_Functor>::apply(std::forward<T_Functor>(functor));
        }
    };

    template<typename T_DatumDomain, std::size_t... T_coords>
    struct ForEach<T_DatumDomain, DatumCoord<T_coords...>>
    {
        template<typename T_Functor>
        LLAMA_FN_HOST_ACC_INLINE static void apply(T_Functor && functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            internal::ApplyFunctorForEachLeaf<
                T_DatumDomain,
                DatumCoord<T_coords...>,
                T_Functor>::apply(std::forward<T_Functor>(functor));
        }
    };
}

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

#include "TreeCoord.hpp"
#include "TreeElement.hpp"

#include <cstddef>
#include <type_traits>

namespace llama::mapping::tree
{
    namespace internal
    {
        // General case (leaf)
        template<typename T_DatumDomain>
        struct ReplaceDEwithTEinDD
        {
            using type = T_DatumDomain;
        };

        template<typename T_DatumElement>
        struct ReplaceDEwithTEinDE;

        template<typename T_Identifier, typename T_Type>
        struct ReplaceDEwithTEinDE<DatumElement<T_Identifier, T_Type>>
        {
            using type = TreeElementConst<
                T_Identifier,
                typename ReplaceDEwithTEinDD<T_Type>::type>;
        };

        // DatumStruct case (node)
        template<typename T_FirstDatumElement, typename... T_DatumElements>
        struct ReplaceDEwithTEinDD<
            DatumStruct<T_FirstDatumElement, T_DatumElements...>>
        {
            using type = boost::mp11::mp_push_front<
                typename ReplaceDEwithTEinDD<
                    DatumStruct<T_DatumElements...>>::type,
                typename ReplaceDEwithTEinDE<T_FirstDatumElement>::type>;
        };

        // empty DatumStruct case ( recursive loop head )
        template<>
        struct ReplaceDEwithTEinDD<DatumStruct<>>
        {
            using type = Tuple<>;
        };

        template<typename T_Leaf, std::size_t T_count>
        struct CreateTreeChain
        {
            using type = TreeElement<
                NoName,
                Tuple<typename CreateTreeChain<T_Leaf, T_count - 1>::type>>;
        };

        template<typename T_Leaf>
        struct CreateTreeChain<T_Leaf, 0>
        {
            using type = T_Leaf;
        };

        template<typename T_DatumDomain>
        struct TreeFromDatumDomainImpl
        {
            using type = TreeElement<
                NoName,
                typename ReplaceDEwithTEinDD<T_DatumDomain>::type>;
        };

        template<typename T_UserDomain, typename T_DatumDomain>
        struct TreeFromDomainsImpl
        {
            using type = typename CreateTreeChain<
                typename TreeFromDatumDomainImpl<T_DatumDomain>::type,
                T_UserDomain::count - 1>::type;
        };
    }

    template<typename T_DatumDomain>
    using TreeFromDatumDomain =
        typename internal::TreeFromDatumDomainImpl<T_DatumDomain>::type;

    template<typename T_UserDomain, typename T_DatumDomain>
    using TreeFromDomains = typename internal::
        TreeFromDomainsImpl<T_UserDomain, T_DatumDomain>::type;

    namespace internal
    {
        template<
            typename T_DatumDomain,
            typename T_UserDomain,
            std::size_t T_pos = 0>
        struct SetUserDomainInTreeImpl
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(T_UserDomain const & size) const
            {
                if constexpr(T_pos == T_UserDomain::count - 1)
                {
                    return TreeFromDatumDomain<T_DatumDomain>(
                        size[T_UserDomain::count - 1]);
                }
                else
                {
                    Tuple inner{SetUserDomainInTreeImpl<
                        T_DatumDomain,
                        T_UserDomain,
                        T_pos + 1>()(size)};
                    return TreeElement<NoName, decltype(inner)>(
                        size[T_pos], inner);
                }
            }
        };
    }

    template<typename T_DatumDomain, typename T_UserDomain>
    LLAMA_FN_HOST_ACC_INLINE auto setUserDomainInTree(T_UserDomain const & size)
    {
        return internal::SetUserDomainInTreeImpl<T_DatumDomain, T_UserDomain>()(
            size);
    };

    namespace internal
    {
        template<
            typename T_UserDomain,
            std::size_t T_firstDatumDomain,
            std::size_t T_pos = 0>
        struct UserDomainToTreeCoord
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(T_UserDomain const & coord) const
            {
                if constexpr(T_pos == T_UserDomain::count - 1)
                {
                    return Tuple{TreeCoordElement<T_firstDatumDomain>(
                        coord[T_UserDomain::count - 1])};
                }
                else
                {
                    return tupleCat(
                        Tuple{TreeCoordElement<0>(coord[T_pos])},
                        UserDomainToTreeCoord<
                            T_UserDomain,
                            T_firstDatumDomain,
                            T_pos + 1>()(coord));
                }
            }
        };

        template<typename T_DatumCoord>
        struct DatumCoordToTreeCoord
        {
            using TailCoord = PopFront<T_DatumCoord>;
            using type = decltype(tupleCat(
                Tuple{TreeCoordElementConst<TailCoord::front>{}},
                typename DatumCoordToTreeCoord<TailCoord>::type()));
        };

        template<std::size_t T_lastCoord>
        struct DatumCoordToTreeCoord<DatumCoord<T_lastCoord>>
        {
            using type = Tuple<TreeCoordElementConst<>>;
        };
    }

    template<typename T_DatumCoord, typename T_UserDomain>
    LLAMA_FN_HOST_ACC_INLINE auto
    getBasicTreeCoordFromDomains(T_UserDomain const & coord)
    {
        return tupleCat(
            internal::
                UserDomainToTreeCoord<T_UserDomain, T_DatumCoord::front>()(
                    coord),
            typename internal::DatumCoordToTreeCoord<T_DatumCoord>::type());
    }
}

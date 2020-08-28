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
        template<typename DatumDomain>
        struct ReplaceDEwithTEinDD
        {
            using type = DatumDomain;
        };

        template<typename DatumElement>
        struct ReplaceDEwithTEinDE;

        template<typename Tag, typename Type>
        struct ReplaceDEwithTEinDE<DatumElement<Tag, Type>>
        {
            using type = TreeElementConst<
                Tag,
                typename ReplaceDEwithTEinDD<Type>::type>;
        };

        template<typename... DatumElements>
        struct ReplaceDEwithTEinDD<DatumStruct<DatumElements...>>
        {
            using type
                = Tuple<typename ReplaceDEwithTEinDE<DatumElements>::type...>;
        };

        template<typename Leaf, std::size_t Count>
        struct WrapInNTreeElements
        {
            using type = TreeElement<
                NoName,
                Tuple<typename WrapInNTreeElements<Leaf, Count - 1>::type>>;
        };

        template<typename Leaf>
        struct WrapInNTreeElements<Leaf, 0>
        {
            using type = Leaf;
        };

        template<typename DatumDomain>
        using TreeFromDatumDomainImpl = TreeElement<
            NoName,
            typename ReplaceDEwithTEinDD<DatumDomain>::type>;
    }

    template<typename DatumDomain>
    using TreeFromDatumDomain = internal::TreeFromDatumDomainImpl<DatumDomain>;

    template<typename UserDomain, typename DatumDomain>
    using TreeFromDomains = typename internal::WrapInNTreeElements<
        internal::TreeFromDatumDomainImpl<DatumDomain>,
        UserDomain::count - 1>::type;

    template<typename DatumDomain, typename UserDomain, std::size_t Pos = 0>
    LLAMA_FN_HOST_ACC_INLINE auto createTree(const UserDomain & size)
    {
        if constexpr(Pos == UserDomain::count - 1)
        {
            return TreeFromDatumDomain<DatumDomain>(
                size[UserDomain::count - 1]);
        }
        else
        {
            Tuple inner{createTree<DatumDomain, UserDomain, Pos + 1>(size)};
            return TreeElement<NoName, decltype(inner)>(size[Pos], inner);
        }
    };

    namespace internal
    {
        template<
            typename UserDomain,
            std::size_t... UDIndices,
            std::size_t FirstDatumDomainCoord,
            std::size_t... DatumDomainCoords>
        LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(
            const UserDomain & coord,
            std::index_sequence<UDIndices...>,
            DatumCoord<FirstDatumDomainCoord, DatumDomainCoords...>)
        {
            return Tuple{
                TreeCoordElement<(
                    UDIndices == UserDomain::count - 1 ? FirstDatumDomainCoord
                                                       : 0)>(
                    coord[UDIndices])...,
                TreeCoordElement<
                    DatumDomainCoords,
                    boost::mp11::mp_size_t<0>>{}...,
                TreeCoordElement<0, boost::mp11::mp_size_t<0>>{}};
        }
    }

    template<typename DatumCoord, typename UserDomain>
    LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(const UserDomain & coord)
    {
        return internal::createTreeCoord(
            coord, std::make_index_sequence<UserDomain::count>{}, DatumCoord{});
    }
}

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

#include "../../Tuple.hpp"

#include <cstddef>
#include <type_traits>

namespace llama::mapping::tree
{
    template<typename T>
    inline constexpr auto one = 1;

    template<>
    inline constexpr auto
        one<boost::mp11::mp_size_t<1>> = boost::mp11::mp_size_t<1>{};

    template<
        typename T_Identifier,
        typename T_Type,
        typename CountType = std::size_t>
    struct TreeElement
    {
        using Identifier = T_Identifier;
        using Type = T_Type;

        LLAMA_FN_HOST_ACC_INLINE
        TreeElement() = default;

        LLAMA_FN_HOST_ACC_INLINE
        TreeElement(CountType count) : count(count) {}

        const CountType count = one<CountType>;
    };

    template<typename T_Identifier, typename CountType, typename... Children>
    struct TreeElement<T_Identifier, Tuple<Children...>, CountType>
    {
        using Identifier = T_Identifier;
        using Type = Tuple<Children...>;

        LLAMA_FN_HOST_ACC_INLINE
        TreeElement() = default;

        LLAMA_FN_HOST_ACC_INLINE
        TreeElement(CountType count, Type childs) : count(count), childs(childs)
        {}

        LLAMA_FN_HOST_ACC_INLINE
        TreeElement(CountType count) : count(count), childs() {}

        const CountType count = one<CountType>;
        const Type childs = {};
    };

    template<typename TreeElement, typename = void>
    inline constexpr auto HasChildren = false;

    template<typename TreeElement>
    inline constexpr auto HasChildren<
        TreeElement,
        std::void_t<decltype(std::declval<TreeElement>().childs)>> = true;

    template<typename Identifier, typename Type, std::size_t Count = 1>
    using TreeElementConst
        = TreeElement<Identifier, Type, boost::mp11::mp_size_t<Count>>;

    template<typename Tree>
    struct TreePopFrontChild
    {
        using ResultType = TreeElement<
            typename Tree::Identifier,
            typename Tree::Type::RestTuple,
            decltype(Tree::count)>;

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(const Tree & tree) -> ResultType
        {
            return {tree.count, tree.childs.rest};
        }
    };

    template<std::size_t Compiletime = 0, typename RuntimeType = std::size_t>
    struct TreeCoordElement
    {
        static constexpr boost::mp11::mp_size_t<Compiletime> compiletime = {};
        const RuntimeType runtime = {};

        LLAMA_FN_HOST_ACC_INLINE
        TreeCoordElement() = default;

        LLAMA_FN_HOST_ACC_INLINE
        TreeCoordElement(RuntimeType runtime) : runtime(runtime) {}
    };

    template<std::size_t... Coords>
    using TreeCoord
        = Tuple<TreeCoordElement<Coords, boost::mp11::mp_size_t<0>>...>;

    namespace internal
    {
        template<typename... Coords, std::size_t... Is>
        auto treeCoordToString(
            Tuple<Coords...> treeCoord,
            std::index_sequence<Is...>) -> std::string
        {
            auto s
                = ((std::to_string(getTupleElement<Is>(treeCoord).runtime) + ":"
                    + std::to_string(getTupleElement<Is>(treeCoord).compiletime)
                    + ", ")
                   + ...);
            s.resize(s.length() - 2);
            return s;
        }
    }

    template<typename TreeCoord>
    auto treeCoordToString(TreeCoord treeCoord) -> std::string
    {
        return std::string("[ ")
            + internal::treeCoordToString(
                   treeCoord,
                   std::make_index_sequence<SizeOfTuple<TreeCoord>>{})
            + std::string(" ]");
    }

    namespace internal
    {
        template<
            typename Tag,
            typename DatumDomain,
            template<typename, typename> typename TE = TreeElementConst>
        struct CreateTreeElement
        {
            using type = TE<Tag, DatumDomain>;
        };

        template<
            typename Tag,
            typename... DatumElements,
            template<typename, typename>
            typename TE>
        struct CreateTreeElement<Tag, DatumStruct<DatumElements...>, TE>
        {
            using type = TE<
                Tag,
                Tuple<typename CreateTreeElement<
                    GetDatumElementUID<DatumElements>,
                    GetDatumElementType<DatumElements>>::type...>>;
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
        using TreeFromDatumDomainImpl =
            typename CreateTreeElement<NoName, DatumDomain, TreeElement>::type;
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

// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "../../Tuple.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

namespace llama::mapping::tree
{
    template <typename T>
    inline constexpr auto one = 1;

    template <>
    inline constexpr auto one<boost::mp11::mp_size_t<1>> = boost::mp11::mp_size_t<1>{};

    template <typename T_Identifier, typename T_Type, typename CountType = std::size_t>
    struct Leaf
    {
        using Identifier = T_Identifier;
        using Type = T_Type;

        const CountType count = one<CountType>;
    };

    template <typename T_Identifier, typename T_ChildrenTuple, typename CountType = std::size_t>
    struct Node
    {
        using Identifier = T_Identifier;
        using ChildrenTuple = T_ChildrenTuple;

        const CountType count = one<CountType>;
        const ChildrenTuple childs = {};
    };

    template <std::size_t ChildIndex = 0, typename ArrayIndexType = std::size_t>
    struct TreeCoordElement
    {
        static constexpr boost::mp11::mp_size_t<ChildIndex> childIndex = {};
        const ArrayIndexType arrayIndex = {};
    };

    template <std::size_t... Coords>
    using TreeCoord = Tuple<TreeCoordElement<Coords, boost::mp11::mp_size_t<0>>...>;

    namespace internal
    {
        template <typename... Coords, std::size_t... Is>
        auto treeCoordToString(Tuple<Coords...> treeCoord, std::index_sequence<Is...>) -> std::string
        {
            auto s
                = ((std::to_string(get<Is>(treeCoord).arrayIndex) + ":" + std::to_string(get<Is>(treeCoord).childIndex)
                    + ", ")
                   + ...);
            s.resize(s.length() - 2);
            return s;
        }
    } // namespace internal

    template <typename TreeCoord>
    auto treeCoordToString(TreeCoord treeCoord) -> std::string
    {
        return std::string("[ ")
            + internal::treeCoordToString(treeCoord, std::make_index_sequence<tupleSize<TreeCoord>>{})
            + std::string(" ]");
    }

    namespace internal
    {
        template <typename Tag, typename RecordDim, typename CountType>
        struct CreateTreeElement
        {
            using type = Leaf<Tag, RecordDim, boost::mp11::mp_size_t<1>>;
        };

        template <typename Tag, typename... Fields, typename CountType>
        struct CreateTreeElement<Tag, Record<Fields...>, CountType>
        {
            using type = Node<
                Tag,
                Tuple<typename CreateTreeElement<GetFieldTag<Fields>, GetFieldType<Fields>, boost::mp11::mp_size_t<1>>::
                          type...>,
                CountType>;
        };

        template <typename Leaf, std::size_t Count>
        struct WrapInNNodes
        {
            using type = Node<NoName, Tuple<typename WrapInNNodes<Leaf, Count - 1>::type>>;
        };

        template <typename Leaf>
        struct WrapInNNodes<Leaf, 0>
        {
            using type = Leaf;
        };

        template <typename RecordDim>
        using TreeFromRecordDimImpl = typename CreateTreeElement<NoName, RecordDim, std::size_t>::type;
    } // namespace internal

    template <typename RecordDim>
    using TreeFromRecordDim = internal::TreeFromRecordDimImpl<RecordDim>;

    template <typename ArrayDomain, typename RecordDim>
    using TreeFromDomains =
        typename internal::WrapInNNodes<internal::TreeFromRecordDimImpl<RecordDim>, ArrayDomain::rank - 1>::type;

    template <typename RecordDim, typename ArrayDomain, std::size_t Pos = 0>
    LLAMA_FN_HOST_ACC_INLINE auto createTree(const ArrayDomain& size)
    {
        if constexpr (Pos == ArrayDomain::rank - 1)
            return TreeFromRecordDim<RecordDim>{size[ArrayDomain::rank - 1]};
        else
        {
            Tuple inner{createTree<RecordDim, ArrayDomain, Pos + 1>(size)};
            return Node<NoName, decltype(inner)>{size[Pos], inner};
        }
    };

    namespace internal
    {
        template <
            typename ArrayDomain,
            std::size_t... ADIndices,
            std::size_t FirstRecordCoord,
            std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(
            const ArrayDomain& coord,
            std::index_sequence<ADIndices...>,
            RecordCoord<FirstRecordCoord, RecordCoords...>)
        {
            return Tuple{
                TreeCoordElement<(ADIndices == ArrayDomain::rank - 1 ? FirstRecordCoord : 0)>{coord[ADIndices]}...,
                TreeCoordElement<RecordCoords, boost::mp11::mp_size_t<0>>{}...,
                TreeCoordElement<0, boost::mp11::mp_size_t<0>>{}};
        }
    } // namespace internal

    template <typename RecordCoord, typename ArrayDomain>
    LLAMA_FN_HOST_ACC_INLINE auto createTreeCoord(const ArrayDomain& coord)
    {
        return internal::createTreeCoord(coord, std::make_index_sequence<ArrayDomain::rank>{}, RecordCoord{});
    }
} // namespace llama::mapping::tree

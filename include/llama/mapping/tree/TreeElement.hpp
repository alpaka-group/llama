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
    struct HasChildren : std::false_type
    {};

    template<typename TreeElement>
    struct HasChildren<
        TreeElement,
        std::void_t<decltype(std::declval<TreeElement>().childs)>> :
            std::true_type
    {};

    template<typename Identifier, typename Type, std::size_t Count = 1>
    using TreeElementConst
        = TreeElement<Identifier, Type, boost::mp11::mp_size_t<Count>>;

    template<typename T_Tree>
    struct TreePopFrontChild
    {
        using ResultType = TreeElement<
            typename T_Tree::Identifier,
            typename T_Tree::Type::RestTuple,
            decltype(T_Tree::count)
        >;

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(const T_Tree & tree) -> ResultType
        {
            return {tree.count, tree.childs.rest};
        }
    };
}

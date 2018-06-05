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

#include <cstddef>
#include <type_traits>

#include "../../Tuple.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

template<
    typename T_Identifier,
    typename T_Type
>
struct TreeElement
{
	using IsTreeElementWithoutChilds = void;
    using Identifier = T_Identifier;
    using Type = T_Type;

    TreeElement() : count(1) {}

    TreeElement( const std::size_t count ) : count(count) {}

    const std::size_t count;
};

template<
    typename T_Identifier,
    typename... T_Childs
>
struct TreeElement<
    T_Identifier,
    Tuple< T_Childs... >
>
{
	using IsTreeElementWithChilds = void;
    using Identifier = T_Identifier;
    using Type = Tuple< T_Childs... >;

    TreeElement() : count(1) {}

    TreeElement(
        const std::size_t count,
        const Type childs = Type()
    ) :
        count(count),
        childs(childs)
    {}

    const std::size_t count;
    const Type childs;
};

template<
    typename T_Identifier,
    typename T_Type,
    std::size_t T_count = 1
>
struct TreeElementConst
{
	using IsTreeElementWithoutChilds = void;
    using Identifier = T_Identifier;
    using Type = T_Type;
    static std::integral_constant< std::size_t, T_count > count;
};

template<
    typename T_Identifier,
    std::size_t T_count,
    typename... T_Childs
>
struct TreeElementConst<
    T_Identifier,
    Tuple< T_Childs... >,
    T_count
>
{
	using IsTreeElementWithChilds = void;
    using Identifier = T_Identifier;
    using Type = Tuple< T_Childs... >;
    static std::integral_constant< std::size_t, T_count > count;

    TreeElementConst(
        const Type childs = Type()
    ) :
        childs(childs)
    {}

    const Type childs;
};

template< typename T_Tree >
struct TreePopFrontChild
{
    using ResultType = TreeElement<
        typename T_Tree::Identifier,
        typename T_Tree::Type::RestTuple
    >;
    auto
    operator()( T_Tree const & tree)
    -> ResultType
    {
        return ResultType(
            tree.count,
            tree.childs.rest
        );
    }
};

template<
    typename T_Identifier,
    typename T_Type,
    std::size_t T_count
>
struct TreePopFrontChild<
    TreeElementConst<
        T_Identifier,
        T_Type,
        T_count
    >
>
{
    using Tree = TreeElementConst<
        T_Identifier,
        T_Type,
        T_count
    >;
    using ResultType = TreeElementConst<
        typename Tree::Identifier,
        typename Tree::Type::RestTuple,
        Tree::count
    >;
    auto
    operator()( Tree const & tree )
    -> ResultType
    {
        return ResultType( tree.childs.rest );
    }
};

template<
    typename T_Childs,
    typename T_CountType
>
struct TreeOptimalType
{
    using type = TreeElement<
        NoName,
        T_Childs
    >;
};

template<
    typename T_Childs,
    std::size_t T_count
>
struct TreeOptimalType<
    T_Childs,
    std::integral_constant<
        std::size_t,
        T_count
    >
>
{
    using type = TreeElementConst<
        NoName,
        T_Childs,
        T_count
    >;
};

} // namespace tree

} // namespace mapping

} // namespace llama


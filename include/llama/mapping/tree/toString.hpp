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

#include <string>
#include <typeinfo>
#include <boost/core/demangle.hpp>
#include <boost/algorithm/string/replace.hpp>

#include "../../Tuple.hpp"

namespace llama
{
namespace mapping
{

namespace tree
{

template<
    typename T,
    typename T_SFINAE = void
>
struct ToString
{
    auto
    operator()( const T )
    -> std::string
    {
        return "Unknown";
    }
};


template< >
struct ToString< NoName >
{
    auto
    operator()( const NoName )
    -> std::string
    {
        return "";
    }
};

template<
    typename T_First,
    typename T_Second,
    typename... T_Rest
>
struct ToString<
    Tuple<
        T_First,
        T_Second,
        T_Rest...
    >
>
{
    using Tree = Tuple<
        T_First,
        T_Second,
        T_Rest...
    >;
    auto
    operator()( const Tree tree )
    -> std::string
    {
        return ToString< T_First >()( tree.first )
            + " , "
            + ToString<
                Tuple<
                    T_Second,
                    T_Rest...
                >
            >()( tree.rest );
    }
};

template< typename T_First >
struct ToString< Tuple< T_First > >
{
    using Tree = Tuple< T_First >;
    auto
    operator()( const Tree tree )
    -> std::string
    {
        return ToString< T_First >()( tree.first );
    }
};

template< typename T_Tree >
struct ToString<
    T_Tree,
    typename T_Tree::IsTreeElementWithoutChilds
>
{
    using Identifier = typename T_Tree::Identifier;
    auto
    operator()( const T_Tree tree )
    -> std::string
    {
        auto raw = boost::core::demangle(typeid(typename T_Tree::Type()).name());
#ifdef _MSC_VER
        boost::replace_all(raw, " __cdecl(void)", "");
#endif
#ifdef __GNUG__
        boost::replace_all(raw, " ()", "");
#endif
        return std::to_string( tree.count )
            + " * "
            + ToString< Identifier >()( Identifier() )
            + "("
            + raw
            + ")";
    }
};

template< typename T_Tree >
struct ToString<
    T_Tree,
    typename T_Tree::IsTreeElementWithChilds
>
{
    using Identifier = typename T_Tree::Identifier;
    auto
    operator()( const T_Tree tree )
    -> std::string
    {
        return std::to_string( tree.count )
            + " * "
            + ToString< Identifier >()( Identifier() )
            + "[ "
            + ToString< typename T_Tree::Type >()( tree.childs )
            + " ]";
    }
};

template< typename T_Tree >
auto
toString( T_Tree tree )
-> std::string
{
    return ToString< T_Tree >()( tree );
};

} // namespace tree

} // namespace mapping

} // namespace llama

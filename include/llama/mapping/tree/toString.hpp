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
#include <boost/algorithm/string/replace.hpp>

#include "../../Tuple.hpp"

#ifdef __GNUG__
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#endif

namespace llama
{

#ifdef __GNUG__
std::string demangleType(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
}
#else

// does nothing if not g++
std::string demangleType(const char* name) {
    return name;
}

#endif

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
        return std::to_string( tree.count )
            + " * "
            + ToString< Identifier >()( Identifier() )
            + "("
            + boost::replace_all_copy(
                demangleType( typeid( typename T_Tree::Type() ).name() ),
                " ()",
                ""
            )
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

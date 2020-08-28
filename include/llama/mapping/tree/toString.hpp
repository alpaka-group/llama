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
#include "TreeElement.hpp"

#include <boost/algorithm/string/replace.hpp>
#include <boost/core/demangle.hpp>
#include <string>
#include <typeinfo>

namespace llama::mapping::tree
{
    template<typename T>
    auto toString(T) -> std::string
    {
        return "Unknown";
    }

    inline auto toString(NoName) -> std::string
    {
        return "";
    }

    template<typename... Elements>
    auto toString(Tuple<Elements...> tree) -> std::string
    {
        if constexpr(sizeof...(Elements) > 1)
            return toString(tree.first) + " , " + toString(tree.rest);
        else
            return toString(tree.first);
    }

    template<typename Identifier, typename Type, typename CountType>
    auto toString(TreeElement<Identifier, Type, CountType> tree)
        -> std::string
    {
        auto r = std::to_string(tree.count) + " * " + toString(Identifier{});
        if constexpr(HasChildren<decltype(tree)>::value)
            r += "[ " + toString(tree.childs) + " ]";
        else
        {
            auto raw = boost::core::demangle(typeid(Type()).name());
#ifdef _MSC_VER
            boost::replace_all(raw, " __cdecl(void)", "");
#endif
#ifdef __GNUG__
            boost::replace_all(raw, " ()", "");
#endif
            r += "(" + raw + ")";
        }
        return r;
    }
}

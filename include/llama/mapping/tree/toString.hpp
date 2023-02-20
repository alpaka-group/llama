// Copyright 2020 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "TreeFromDimensions.hpp"

#include <string>

namespace llama::mapping::tree
{
    template<typename T>
    auto toString(T) -> std::string
    {
        return "Unknown";
    }

    // handles array indices
    template<std::size_t I>
    inline auto toString(RecordCoord<I>) -> std::string
    {
        return "";
    }

    inline auto toString(NoName) -> std::string
    {
        return "";
    }

    template<typename... Elements>
    auto toString(Tuple<Elements...> tree) -> std::string
    {
        if constexpr(sizeof...(Elements) > 1)
            return toString(tree.first()) + " , " + toString(tree.rest());
        else
            return toString(tree.first());
    }

    namespace internal
    {
        inline void replaceAll(std::string& str, const std::string& search, const std::string& replace)
        {
            std::string::size_type i = 0;
            while((i = str.find(search, i)) != std::string::npos)
            {
                str.replace(i, search.length(), replace);
                i += replace.length();
            }
        }

        template<typename NodeOrLeaf>
        auto countAndIdentToString(const NodeOrLeaf& nodeOrLeaf) -> std::string
        {
            auto r = std::to_string(nodeOrLeaf.count);
            if constexpr(std::is_same_v<std::decay_t<decltype(nodeOrLeaf.count)>, std::size_t>)
                r += "R"; // runtime
            else
                r += "C"; // compile time
            r += std::string{" * "} + toString(typename NodeOrLeaf::Identifier{});
            return r;
        }
    } // namespace internal

    template<typename Identifier, typename Type, typename CountType>
    auto toString(const Node<Identifier, Type, CountType>& node) -> std::string
    {
        return internal::countAndIdentToString(node) + "[ " + toString(node.childs) + " ]";
    }

    template<typename Identifier, typename Type, typename CountType>
    auto toString(const Leaf<Identifier, Type, CountType>& leaf) -> std::string
    {
        auto raw = std::string{llama::structName<Type>()};
#ifdef _MSC_VER
        internal::replaceAll(raw, " __cdecl(void)", "");
#endif
#ifdef __GNUG__
        internal::replaceAll(raw, " ()", "");
#endif
        return internal::countAndIdentToString(leaf) + "(" + raw + ")";
    }
} // namespace llama::mapping::tree

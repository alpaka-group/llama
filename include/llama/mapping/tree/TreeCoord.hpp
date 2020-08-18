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

#include <cstddef>
#include <string>
#include <type_traits>

namespace llama::mapping::tree
{
    template<
        std::size_t T_compiletime = 0,
        typename T_RuntimeType = std::size_t>
    struct TreeCoordElement
    {
        using CompileType = std::integral_constant<std::size_t, T_compiletime>;

        static constexpr CompileType compiletime = {};
        const T_RuntimeType runtime = 0;

        LLAMA_FN_HOST_ACC_INLINE
        TreeCoordElement(const T_RuntimeType runtime = 0) : runtime(runtime) {}
    };

    template<
        std::size_t T_compiletime,
        typename T_RuntimeType,
        T_RuntimeType T_runtime>
    struct TreeCoordElement<
        T_compiletime,
        std::integral_constant<T_RuntimeType, T_runtime>>
    {
        using RuntimeType = std::integral_constant<T_RuntimeType, T_runtime>;
        using CompileType = std::integral_constant<std::size_t, T_compiletime>;

        static constexpr CompileType compiletime = {};
        static constexpr RuntimeType runtime = {};

        LLAMA_FN_HOST_ACC_INLINE
        TreeCoordElement() = default;

        LLAMA_FN_HOST_ACC_INLINE
        TreeCoordElement(RuntimeType const) {}
    };

    template<std::size_t T_compiletime = 0, std::size_t T_runtime = 0>
    using TreeCoordElementConst = TreeCoordElement<
        T_compiletime,
        std::integral_constant<std::size_t, T_runtime>>;

    template<std::size_t... T_coords>
    using TreeCoord = Tuple<TreeCoordElementConst<T_coords>...>;

    namespace internal
    {
        template<typename... Coords>
        auto treeCoordToString(Tuple<Coords...> treeCoord) -> std::string
        {
            return std::to_string(treeCoord.first.runtime) + ":"
                + std::to_string(treeCoord.first.compiletime) + ", "
                + treeCoordToString(treeCoord.rest);
        }

        template<typename Coord>
        auto treeCoordToString(Tuple<Coord> treeCoord) -> std::string
        {
            return std::to_string(treeCoord.first.runtime) + std::string(":")
                + std::to_string(treeCoord.first.compiletime);
        }
    }

    template<typename T_TreeCoord>
    auto treeCoordToString(const T_TreeCoord treeCoord) -> std::string
    {
        return std::string("[ ") + internal::treeCoordToString(treeCoord)
            + std::string(" ]");
    }
}

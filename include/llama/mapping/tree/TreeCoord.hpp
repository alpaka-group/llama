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
}

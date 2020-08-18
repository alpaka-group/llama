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

#include "../TreeElement.hpp"
#include "../operations/GetNode.hpp"

namespace llama::mapping::tree::functor
{
    /// Functor for \ref tree::Mapping. Moves all run time parts to the leaves,
    /// so in fact another struct of array implementation -- but with the
    /// possibility to add further finetuning of the mapping in the future. \see
    /// tree::Mapping
    struct LeafOnlyRT
    {
        template<typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(T_Tree tree) const
        {
            return basicToResultImpl(tree);
        }

        template<typename T_Tree, typename T_BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            T_BasicCoord const & basicCoord,
            T_Tree const & tree) const
        {
            return basicCoordToResultCoordImpl(basicCoord, tree);
        }

        template<typename T_Tree, typename T_ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            T_ResultCoord const & resultCoord,
            T_Tree const & tree) const -> T_ResultCoord
        {
            return resultCoord;
        }

    private:
        template<typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE static auto
        basicToResultImpl(T_Tree tree, std::size_t runtime = 1)
        {
            if constexpr(HasChildren<T_Tree>::value)
            {
                auto children = tupleTransform(
                    tree.childs,
                    [runtime
                     = runtime * LLAMA_DEREFERENCE(tree.count)](auto element) {
                        return basicToResultImpl(element, runtime);
                    });
                return TreeElementConst<
                    typename T_Tree::Identifier,
                    decltype(children),
                    1>{children};
            }
            else
                return TreeElement<
                    typename T_Tree::Identifier,
                    typename T_Tree::Type>{
                    LLAMA_DEREFERENCE(tree.count) * runtime};
        }

        template<typename T_BasicCoord, typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE static auto basicCoordToResultCoordImpl(
            T_BasicCoord const & basicCoord,
            T_Tree const & tree,
            std::size_t const runtime = 0)
        {
            if constexpr(SizeOfTuple<T_BasicCoord>::value == 1)
                return Tuple{TreeCoordElement<decltype(
                    T_BasicCoord::FirstElement::compiletime)::value>(
                    runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))};
            else
            {
                const auto & branch = getTupleElementRef<
                    T_BasicCoord::FirstElement::compiletime>(tree.childs);
                auto first = TreeCoordElementConst<
                    decltype(T_BasicCoord::FirstElement::compiletime)::value,
                    0>{};

                return tupleCat(
                    Tuple{first},
                    basicCoordToResultCoordImpl<
                        typename T_BasicCoord::RestTuple,
                        GetTupleType<
                            typename T_Tree::Type,
                            decltype(T_BasicCoord::FirstElement::compiletime)::
                                value>>(
                        basicCoord.rest,
                        branch,
                        (runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))
                            * LLAMA_DEREFERENCE(branch.count)));
            }
        }
    };
}

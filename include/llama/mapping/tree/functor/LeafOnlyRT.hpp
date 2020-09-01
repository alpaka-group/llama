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

namespace llama::mapping::tree::functor
{
    /// Functor for \ref tree::Mapping. Moves all run time parts to the leaves,
    /// so in fact another struct of array implementation -- but with the
    /// possibility to add further finetuning of the mapping in the future. \see
    /// tree::Mapping
    struct LeafOnlyRT
    {
        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(Tree tree) const
        {
            return basicToResultImpl(tree);
        }

        template<typename Tree, typename BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            const BasicCoord & basicCoord,
            const Tree & tree) const
        {
            return basicCoordToResultCoordImpl(basicCoord, tree);
        }

        template<typename Tree, typename ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            const ResultCoord & resultCoord,
            const Tree & tree) const -> ResultCoord
        {
            return resultCoord;
        }

    private:
        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE static auto
        basicToResultImpl(Tree tree, std::size_t runtime = 1)
        {
            if constexpr(HasChildren<Tree>)
            {
                auto children = tupleTransform(tree.childs, [&](auto element) {
                    return basicToResultImpl(
                        element, runtime * LLAMA_DEREFERENCE(tree.count));
                });
                return TreeElement<
                    typename Tree::Identifier,
                    decltype(children),
                    boost::mp11::mp_size_t<1>>{{}, children};
            }
            else
                return TreeElement<
                    typename Tree::Identifier,
                    typename Tree::Type>{
                    LLAMA_DEREFERENCE(tree.count) * runtime};
        }

        template<typename BasicCoord, typename Tree>
        LLAMA_FN_HOST_ACC_INLINE static auto basicCoordToResultCoordImpl(
            const BasicCoord & basicCoord,
            const Tree & tree,
            std::size_t runtime = 0)
        {
            if constexpr(SizeOfTuple<BasicCoord> == 1)
                return Tuple{
                    TreeCoordElement<BasicCoord::FirstElement::compiletime>(
                        runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))};
            else
            {
                const auto & branch
                    = getTupleElementRef<BasicCoord::FirstElement::compiletime>(
                        tree.childs);
                auto first = TreeCoordElement<
                    BasicCoord::FirstElement::compiletime,
                    boost::mp11::mp_size_t<0>>{};

                return tupleCat(
                    Tuple{first},
                    basicCoordToResultCoordImpl<
                        typename BasicCoord::RestTuple,
                        GetTupleType<
                            typename Tree::Type,
                            BasicCoord::FirstElement::compiletime>>(
                        basicCoord.rest,
                        branch,
                        (runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))
                            * LLAMA_DEREFERENCE(branch.count)));
            }
        }
    };
}

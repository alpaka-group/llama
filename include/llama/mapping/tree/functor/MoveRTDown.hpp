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
#include "../operations/ChangeNodeRuntime.hpp"
#include "../operations/GetNode.hpp"

namespace llama::mapping::tree::functor
{
    /// Functor for \ref tree::Mapping. Move the run time part of a node one
    /// level down in direction of the leaves. \warning Broken at the moment
    /// \tparam T_TreeCoord tree coordinate in the mapping tree which's run time
    /// part shall be moved down one level \see tree::Mapping
    template<typename T_TreeCoord>
    struct MoveRTDown
    {
        const std::size_t amount = 0;

        template<typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(T_Tree const & tree) const
        {
            return operations::
                changeNodeChildsRuntime<T_TreeCoord, Multiplication>(
                    operations::changeNodeRuntime<T_TreeCoord>(
                        tree,
                        (operations::getNode<T_TreeCoord>(tree).count + amount
                         - 1)
                            / amount),
                    amount);
        }

        template<typename T_Tree, typename T_BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            T_BasicCoord const & basicCoord,
            T_Tree const & tree) const
        {
            return basicCoordToResultCoordImpl<T_TreeCoord>(
                basicCoord, tree, amount);
        }

        template<typename T_Tree, typename T_ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            T_ResultCoord const & resultCoord,
            T_Tree const &) const -> T_ResultCoord
        {
            return resultCoord;
        }

    private:
        template<
            typename T_InternalTreeCoord,
            typename T_BasicCoord,
            typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoordImpl(
            const T_BasicCoord & basicCoord,
            const T_Tree & tree,
            std::size_t amount) const
        {
            if constexpr(std::is_same_v<T_InternalTreeCoord, Tuple<>>)
            {
                if constexpr(std::is_same_v<T_BasicCoord, Tuple<>>)
                    return Tuple<>{};
                else
                {
                    const auto & childTree = getTupleElementRef<decltype(
                        T_BasicCoord::FirstElement::compiletime)::value>(
                        tree.childs);
                    const auto rt1 = basicCoord.first.runtime / amount;
                    const auto rt2
                        = basicCoord.first.runtime % amount * childTree.count
                        + basicCoord.rest.first.runtime;
                    auto rt1Child = TreeCoordElement<decltype(
                        T_BasicCoord::FirstElement::compiletime)::value>(rt1);
                    auto rt2Child = TreeCoordElement<decltype(
                        T_BasicCoord::RestTuple::FirstElement::compiletime)::
                                                         value>(rt2);
                    return tupleCat(
                        Tuple{rt1Child},
                        tupleCat(Tuple{rt2Child}, tupleRest(basicCoord.rest)));
                }
            }
            else
            {
                if constexpr(
                    T_InternalTreeCoord::FirstElement::compiletime
                    != T_BasicCoord::FirstElement::compiletime)
                    return basicCoord;
                else
                {
                    auto rest = basicCoordToResultCoordImpl<
                        typename T_InternalTreeCoord::RestTuple>(
                        tupleRest(basicCoord),
                        getTupleElementRef<
                            T_BasicCoord::FirstElement::compiletime>(
                            tree.childs),
                        amount);
                    return tupleCat(Tuple{basicCoord.first}, rest);
                }
            }
        }
    };

    template<typename T_TreeCoord, std::size_t T_amount>
    struct MoveRTDownFixed
    {
        static constexpr std::size_t amount = T_amount;

        template<typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(T_Tree const & tree) const
        {
            return operations::
                changeNodeChildsRuntime<T_TreeCoord, Multiplication>(
                    operations::changeNodeRuntime<T_TreeCoord>(
                        tree,
                        (operations::getNode<T_TreeCoord>(tree).count + amount
                         - 1)
                            / amount),
                    amount);
        }

        template<typename T_Tree, typename T_BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            T_BasicCoord const & basicCoord,
            T_Tree const & tree) const -> decltype(auto)
        {
            return basicCoordToResultCoordImpl<T_TreeCoord>(basicCoord, tree);
        }

        template<typename T_Tree, typename T_ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            T_ResultCoord const & resultCoord,
            T_Tree const &) const -> T_ResultCoord
        {
            return resultCoord;
        }

    private:
        template<
            typename T_InternalTreeCoord,
            typename T_BasicCoord,
            typename T_Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoordImpl(
            const T_BasicCoord & basicCoord,
            const T_Tree & tree) const
        {
            if constexpr(std::is_same_v<T_InternalTreeCoord, Tuple<>>)
            {
                if constexpr(std::is_same_v<T_BasicCoord, Tuple<>>)
                    return Tuple<>{};
                else
                {
                    const auto & childTree = getTupleElementRef<decltype(
                        T_BasicCoord::FirstElement::compiletime)::value>(
                        tree.childs);
                    const auto rt1 = basicCoord.first.runtime / amount;
                    const auto rt2
                        = basicCoord.first.runtime % amount * childTree.count
                        + basicCoord.rest.first.runtime;
                    auto rt1Child = TreeCoordElement<decltype(
                        T_BasicCoord::FirstElement::compiletime)::value>(rt1);
                    auto rt2Child = TreeCoordElement<decltype(
                        T_BasicCoord::RestTuple::FirstElement::compiletime)::
                                                         value>(rt2);
                    return tupleCat(
                        Tuple{rt1Child},
                        tupleCat(Tuple{rt2Child}, tupleRest(basicCoord.rest)));
                }
            }
            else
            {
                if constexpr(
                    T_InternalTreeCoord::FirstElement::compiletime
                    != T_BasicCoord::FirstElement::compiletime)
                    return basicCoord;
                else
                {
                    auto rest = basicCoordToResultCoordImpl<
                        typename T_InternalTreeCoord::RestTuple>(
                        tupleRest(basicCoord),
                        getTupleElementRef<
                            T_BasicCoord::FirstElement::compiletime>(
                            tree.childs));
                    return tupleCat(Tuple{basicCoord.first}, rest);
                }
            }
        }
    };
}

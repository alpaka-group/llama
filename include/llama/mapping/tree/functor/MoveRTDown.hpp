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

namespace llama
{
    namespace mapping
    {
        namespace tree
        {
            namespace functor
            {
                /** Functor for \ref tree::Mapping. Move the run time part of a
                 * node one level down in direction of the leaves. \warning
                 * Broken at the moment \tparam T_TreeCoord tree coordinate in
                 * the mapping tree which's run time part shall be moved down
                 * one level \see tree::Mapping
                 */
                template<typename T_TreeCoord>
                struct MoveRTDown
                {
                    template<
                        typename T_Tree,
                        typename T_InternalTreeCoord,
                        typename T_BasicCoord,
                        typename T_SFINAE = void>
                    struct BasicCoordToResultCoordImpl;

                    template<
                        typename T_Tree,
                        typename T_InternalTreeCoord,
                        typename T_BasicCoord>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        T_InternalTreeCoord,
                        T_BasicCoord,
                        typename std::enable_if<
                            T_InternalTreeCoord::FirstElement::compiletime
                            != T_BasicCoord::FirstElement::compiletime>::type>
                    {
                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_BasicCoord const & basicCoord,
                            T_Tree const & tree,
                            std::size_t const amount) const -> T_BasicCoord
                        {
                            return basicCoord;
                        }
                    };

                    template<
                        typename T_Tree,
                        typename T_InternalTreeCoord,
                        typename T_BasicCoord>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        T_InternalTreeCoord,
                        T_BasicCoord,
                        typename std::enable_if<
                            T_InternalTreeCoord::FirstElement::compiletime
                            == T_BasicCoord::FirstElement::compiletime>::type>
                    {
                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_BasicCoord const & basicCoord,
                            T_Tree const & tree,
                            std::size_t const amount) const -> decltype(auto)
                        {
                            return tupleCat(
                                makeTuple(basicCoord.first),
                                BasicCoordToResultCoordImpl<
                                    GetTupleType<
                                        typename T_Tree::Type,
                                        T_BasicCoord::FirstElement::
                                            compiletime>,
                                    decltype(tupleRest(T_InternalTreeCoord())),
                                    decltype(tupleRest(basicCoord))>()(
                                    tupleRest(basicCoord),
                                    getTupleElementRef<
                                        T_BasicCoord::FirstElement::
                                            compiletime>(tree.childs),
                                    amount));
                        }
                    };

                    template<typename T_Tree, typename T_BasicCoord>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        Tuple<>,
                        T_BasicCoord,
                        void>
                    {
                        using ResultCoord = decltype(tupleCat(
                            makeTuple(
                                TreeCoordElement<
                                    T_BasicCoord::FirstElement::compiletime>()),
                            tupleCat(
                                makeTuple(TreeCoordElement<decltype(tupleRest(
                                              T_BasicCoord()))::FirstElement::
                                                               compiletime>()),
                                tupleRest(tupleRest(T_BasicCoord())))));

                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_BasicCoord const & basicCoord,
                            T_Tree const & tree,
                            std::size_t const amount) const -> ResultCoord
                        {
                            auto const & childTree
                                = getTupleElementRef<decltype(
                                    T_BasicCoord::FirstElement::compiletime)::
                                                         value>(tree.childs);
                            auto const rt1 = basicCoord.first.runtime / amount;
                            auto const rt2 = basicCoord.first.runtime % amount
                                    * childTree.count
                                + basicCoord.rest.first.runtime;
                            return ResultCoord(
                                TreeCoordElement<decltype(
                                    T_BasicCoord::FirstElement::compiletime)::
                                                     value>(rt1),
                                typename ResultCoord::RestTuple(
                                    TreeCoordElement<decltype(
                                        T_BasicCoord::RestTuple::FirstElement::
                                            compiletime)::value>(rt2),
                                    tupleRest(basicCoord.rest)));
                        }
                    };

                    template<typename T_Tree>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        Tuple<>,
                        Tuple<>,
                        void>
                    {
                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            Tuple<> const &,
                            T_Tree const &,
                            std::size_t const) const -> Tuple<>
                        {
                            return {};
                        }
                    };

                    std::size_t const amount;

                    MoveRTDown(std::size_t const amount = 0) : amount(amount) {}

                    template<typename T_Tree>
                    using Result = decltype(operations::changeNodeChildsRuntime<
                                            T_TreeCoord,
                                            Multiplication>(
                        operations::changeNodeRuntime<T_TreeCoord>(
                            T_Tree(),
                            (operations::getNode<T_TreeCoord>(T_Tree()).count
                             + amount - 1)
                                / amount),
                        amount));

                    template<typename T_Tree>
                    LLAMA_FN_HOST_ACC_INLINE auto
                    basicToResult(T_Tree const & tree) const -> Result<T_Tree>
                    {
                        return operations::changeNodeChildsRuntime<
                            T_TreeCoord,
                            Multiplication>(
                            operations::changeNodeRuntime<T_TreeCoord>(
                                tree,
                                (operations::getNode<T_TreeCoord>(tree).count
                                 + amount - 1)
                                    / amount),
                            amount);
                    }

                    template<typename T_Tree, typename T_BasicCoord>
                    LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
                        T_BasicCoord const & basicCoord,
                        T_Tree const & tree) const -> decltype(auto)
                    {
                        return BasicCoordToResultCoordImpl<
                            T_Tree,
                            T_TreeCoord,
                            T_BasicCoord>()(basicCoord, tree, amount);
                    }

                    template<typename T_Tree, typename T_ResultCoord>
                    LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
                        T_ResultCoord const & resultCoord,
                        T_Tree const &) const -> T_ResultCoord
                    {
                        return resultCoord;
                    }
                };

                template<typename T_TreeCoord, std::size_t T_amount>
                struct MoveRTDownFixed
                {
                    static constexpr std::size_t amount = T_amount;

                    template<
                        typename T_Tree,
                        typename T_InternalTreeCoord,
                        typename T_BasicCoord,
                        typename T_SFINAE = void>
                    struct BasicCoordToResultCoordImpl;

                    template<
                        typename T_Tree,
                        typename T_InternalTreeCoord,
                        typename T_BasicCoord>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        T_InternalTreeCoord,
                        T_BasicCoord,
                        typename std::enable_if<
                            T_InternalTreeCoord::FirstElement::compiletime
                            != T_BasicCoord::FirstElement::compiletime>::type>
                    {
                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_BasicCoord const & basicCoord,
                            T_Tree const & tree) const -> T_BasicCoord
                        {
                            return basicCoord;
                        }
                    };

                    template<
                        typename T_Tree,
                        typename T_InternalTreeCoord,
                        typename T_BasicCoord>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        T_InternalTreeCoord,
                        T_BasicCoord,
                        typename std::enable_if<
                            T_InternalTreeCoord::FirstElement::compiletime
                            == T_BasicCoord::FirstElement::compiletime>::type>
                    {
                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_BasicCoord const & basicCoord,
                            T_Tree const & tree) const -> decltype(auto)
                        {
                            return tupleCat(
                                makeTuple(basicCoord.first),
                                BasicCoordToResultCoordImpl<
                                    GetTupleType<
                                        typename T_Tree::Type,
                                        T_BasicCoord::FirstElement::
                                            compiletime>,
                                    decltype(tupleRest(T_InternalTreeCoord())),
                                    decltype(tupleRest(basicCoord))>()(
                                    tupleRest(basicCoord),
                                    getTupleElementRef<
                                        T_BasicCoord::FirstElement::
                                            compiletime>(tree.childs)));
                        }
                    };

                    template<typename T_Tree, typename T_BasicCoord>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        Tuple<>,
                        T_BasicCoord,
                        void>
                    {
                        using ResultCoord = decltype(tupleCat(
                            makeTuple(
                                TreeCoordElement<
                                    T_BasicCoord::FirstElement::compiletime>()),
                            tupleCat(
                                makeTuple(TreeCoordElement<decltype(tupleRest(
                                              T_BasicCoord()))::FirstElement::
                                                               compiletime>()),
                                tupleRest(tupleRest(T_BasicCoord())))));

                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            T_BasicCoord const & basicCoord,
                            T_Tree const & tree) const -> ResultCoord
                        {
                            auto const & childTree
                                = getTupleElementRef<decltype(
                                    T_BasicCoord::FirstElement::compiletime)::
                                                         value>(tree.childs);
                            auto const rt1 = basicCoord.first.runtime / amount;
                            auto const rt2 = basicCoord.first.runtime % amount
                                    * childTree.count
                                + basicCoord.rest.first.runtime;
                            return ResultCoord(
                                TreeCoordElement<decltype(
                                    T_BasicCoord::FirstElement::compiletime)::
                                                     value>(rt1),
                                typename ResultCoord::RestTuple(
                                    TreeCoordElement<decltype(
                                        T_BasicCoord::RestTuple::FirstElement::
                                            compiletime)::value>(rt2),
                                    tupleRest(basicCoord.rest)));
                        }
                    };

                    template<typename T_Tree>
                    struct BasicCoordToResultCoordImpl<
                        T_Tree,
                        Tuple<>,
                        Tuple<>,
                        void>
                    {
                        LLAMA_FN_HOST_ACC_INLINE
                        auto operator()(
                            Tuple<> const &,
                            T_Tree const &,
                            std::size_t const) const -> Tuple<>
                        {
                            return {};
                        }
                    };

                    template<typename T_Tree>
                    using Result = decltype(operations::changeNodeChildsRuntime<
                                            T_TreeCoord,
                                            Multiplication>(
                        operations::changeNodeRuntime<T_TreeCoord>(
                            T_Tree(),
                            (operations::getNode<T_TreeCoord>(T_Tree()).count
                             + amount - 1)
                                / amount),
                        amount));

                    template<typename T_Tree>
                    LLAMA_FN_HOST_ACC_INLINE auto
                    basicToResult(T_Tree const & tree) const -> Result<T_Tree>
                    {
                        return operations::changeNodeChildsRuntime<
                            T_TreeCoord,
                            Multiplication>(
                            operations::changeNodeRuntime<T_TreeCoord>(
                                tree,
                                (operations::getNode<T_TreeCoord>(tree).count
                                 + amount - 1)
                                    / amount),
                            amount);
                    }

                    template<typename T_Tree, typename T_BasicCoord>
                    LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
                        T_BasicCoord const & basicCoord,
                        T_Tree const & tree) const -> decltype(auto)
                    {
                        return BasicCoordToResultCoordImpl<
                            T_Tree,
                            T_TreeCoord,
                            T_BasicCoord>()(basicCoord, tree);
                    }

                    template<typename T_Tree, typename T_ResultCoord>
                    LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
                        T_ResultCoord const & resultCoord,
                        T_Tree const &) const -> T_ResultCoord
                    {
                        return resultCoord;
                    }
                };

            } // namespace functor

        } // namespace tree

    } // namespace mapping

} // namespace llama

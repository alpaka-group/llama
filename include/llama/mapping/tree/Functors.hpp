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

#include "TreeFromDomains.hpp"

namespace llama::mapping::tree::functor
{
    /// Functor for \ref tree::Mapping. Does nothing with the mapping tree at
    /// all (basically implemented for testing purposes). \see tree::Mapping
    struct Idem
    {
        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree & tree) const
            -> Tree
        {
            return tree;
        }

        template<typename Tree, typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            const TreeCoord & basicCoord,
            const Tree &) const -> TreeCoord
        {
            return basicCoord;
        }

        template<typename Tree, typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            const TreeCoord & resultCoord,
            const Tree &) const -> TreeCoord
        {
            return resultCoord;
        }
    };

    /// Functor for \ref tree::Mapping. Moves all run time parts to the leaves,
    /// so in fact another struct of array implementation -- but with the
    /// possibility to add further finetuning of the mapping in the future. \see
    /// tree::Mapping
    struct LeafOnlyRT
    {
        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(Tree tree) const
        {
            return basicToResultImpl(tree, 1);
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
        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
            const Node<Identifier, Type, CountType> & node,
            std::size_t runtime)
        {
            auto children = tupleTransform(node.childs, [&](auto element) {
                return basicToResultImpl(
                    element, LLAMA_DEREFERENCE(node.count) * runtime);
            });
            return Node<
                Identifier,
                decltype(children),
                boost::mp11::mp_size_t<1>>{{}, children};
        }

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
            const Leaf<Identifier, Type, CountType> & leaf,
            std::size_t runtime)
        {
            return Leaf<Identifier, Type, std::size_t>{
                LLAMA_DEREFERENCE(leaf.count) * runtime};
        }

        template<typename BasicCoord, typename NodeOrLeaf>
        LLAMA_FN_HOST_ACC_INLINE static auto basicCoordToResultCoordImpl(
            const BasicCoord & basicCoord,
            const NodeOrLeaf & nodeOrLeaf,
            std::size_t runtime = 0)
        {
            if constexpr(tupleSize<BasicCoord> == 1)
                return Tuple{
                    TreeCoordElement<BasicCoord::FirstElement::compiletime>(
                        runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))};
            else
            {
                const auto & branch
                    = get<BasicCoord::FirstElement::compiletime>(
                        nodeOrLeaf.childs);
                auto first = TreeCoordElement<
                    BasicCoord::FirstElement::compiletime,
                    boost::mp11::mp_size_t<0>>{};

                return tupleCat(
                    Tuple{first},
                    basicCoordToResultCoordImpl<
                        typename BasicCoord::RestTuple,
                        TupleElement<
                            typename NodeOrLeaf::ChildrenTuple,
                            BasicCoord::FirstElement::compiletime>>(
                        basicCoord.rest,
                        branch,
                        (runtime + LLAMA_DEREFERENCE(basicCoord.first.runtime))
                            * LLAMA_DEREFERENCE(branch.count)));
            }
        }
    };

    namespace internal
    {
        template<typename TreeCoord, typename Node>
        LLAMA_FN_HOST_ACC_INLINE auto getNode(const Node & node)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
                return node;
            else
                return getNode<typename TreeCoord::RestTuple>(
                    get<TreeCoord::FirstElement::compiletime>(node.childs));
        }

        template<
            typename TreeCoord,
            typename Identifier,
            typename Type,
            typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
            const Node<Identifier, Type, CountType> & tree,
            std::size_t newValue)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
                return Node<Identifier, Type>{newValue, tree.childs};
            else
            {
                auto current
                    = get<TreeCoord::FirstElement::compiletime>(tree.childs);
                auto replacement
                    = changeNodeRuntime<typename TreeCoord::RestTuple>(
                        current, newValue);
                auto children
                    = tupleReplace<TreeCoord::FirstElement::compiletime>(
                        tree.childs, replacement);
                return Node<Identifier, decltype(children)>{
                    tree.count, children};
            }
        }

        template<
            typename TreeCoord,
            typename Identifier,
            typename Type,
            typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
            const Leaf<Identifier, Type, CountType> & tree,
            std::size_t newValue)
        {
            return Leaf<Identifier, Type, std::size_t>{newValue};
        }

        template<typename Operation>
        struct ChangeNodeChildsRuntimeFunctor
        {
            const std::size_t newValue;

            template<typename Identifier, typename Type, typename CountType>
            LLAMA_FN_HOST_ACC_INLINE auto
            operator()(const Node<Identifier, Type, CountType> & element) const
            {
                return Node<Identifier, Type, std::size_t>{
                    Operation{}(element.count, newValue), element.childs};
            }

            template<typename Identifier, typename Type, typename CountType>
            LLAMA_FN_HOST_ACC_INLINE auto
            operator()(const Leaf<Identifier, Type, CountType> & element) const
            {
                return Leaf<Identifier, Type, std::size_t>{
                    Operation{}(element.count, newValue)};
            }
        };

        template<
            typename TreeCoord,
            typename Operation,
            typename Identifier,
            typename Type,
            typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
            const Node<Identifier, Type, CountType> & tree,
            std::size_t newValue)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
            {
                auto children = tupleTransform(
                    tree.childs,
                    ChangeNodeChildsRuntimeFunctor<Operation>{newValue});
                return Node<Identifier, decltype(children)>{
                    tree.count, children};
            }
            else
            {
                auto current
                    = get<TreeCoord::FirstElement::compiletime>(tree.childs);
                auto replacement = changeNodeChildsRuntime<
                    typename TreeCoord::RestTuple,
                    Operation>(current, newValue);
                auto children
                    = tupleReplace<TreeCoord::FirstElement::compiletime>(
                        tree.childs, replacement);
                return Node<Identifier, decltype(children)>{
                    tree.count, children};
            }
        }

        template<
            typename TreeCoord,
            typename Operation,
            typename Identifier,
            typename Type,
            typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
            const Leaf<Identifier, Type, CountType> & tree,
            std::size_t newValue)
        {
            return tree;
        }
    }

    /// Functor for \ref tree::Mapping. Move the run time part of a node one
    /// level down in direction of the leaves. \warning Broken at the moment
    /// \tparam T_TreeCoord tree coordinate in the mapping tree which's run time
    /// part shall be moved down one level \see tree::Mapping
    template<typename TreeCoord, typename Amount = std::size_t>
    struct MoveRTDown
    {
        const Amount amount = {};

        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree & tree) const
        {
            return internal::
                changeNodeChildsRuntime<TreeCoord, std::multiplies<>>(
                    internal::changeNodeRuntime<TreeCoord>(
                        tree,
                        (internal::getNode<TreeCoord>(tree).count + amount - 1)
                            / amount),
                    amount);
        }

        template<typename Tree, typename BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
            const BasicCoord & basicCoord,
            const Tree & tree) const
        {
            return basicCoordToResultCoordImpl<TreeCoord>(basicCoord, tree);
        }

        template<typename Tree, typename ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
            const ResultCoord & resultCoord,
            const Tree &) const -> ResultCoord
        {
            return resultCoord;
        }

    private:
        template<typename InternalTreeCoord, typename BasicCoord, typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoordImpl(
            const BasicCoord & basicCoord,
            const Tree & tree) const
        {
            if constexpr(std::is_same_v<InternalTreeCoord, Tuple<>>)
            {
                if constexpr(std::is_same_v<BasicCoord, Tuple<>>)
                    return Tuple<>{};
                else
                {
                    const auto & childTree
                        = get<BasicCoord::FirstElement::compiletime>(
                            tree.childs);
                    const auto rt1 = basicCoord.first.runtime / amount;
                    const auto rt2
                        = basicCoord.first.runtime % amount * childTree.count
                        + basicCoord.rest.first.runtime;
                    auto rt1Child = TreeCoordElement<
                        BasicCoord::FirstElement::compiletime>(rt1);
                    auto rt2Child = TreeCoordElement<
                        BasicCoord::RestTuple::FirstElement::compiletime>(rt2);
                    return tupleCat(
                        Tuple{rt1Child},
                        tupleCat(
                            Tuple{rt2Child},
                            tupleWithoutFirst(basicCoord.rest)));
                }
            }
            else
            {
                if constexpr(
                    InternalTreeCoord::FirstElement::compiletime
                    != BasicCoord::FirstElement::compiletime)
                    return basicCoord;
                else
                {
                    auto rest = basicCoordToResultCoordImpl<
                        typename InternalTreeCoord::RestTuple>(
                        tupleWithoutFirst(basicCoord),
                        get<BasicCoord::FirstElement::compiletime>(
                            tree.childs));
                    return tupleCat(Tuple{basicCoord.first}, rest);
                }
            }
        }
    };

    template<typename TreeCoord, std::size_t Amount>
    using MoveRTDownFixed
        = MoveRTDown<TreeCoord, boost::mp11::mp_size_t<Amount>>;
}

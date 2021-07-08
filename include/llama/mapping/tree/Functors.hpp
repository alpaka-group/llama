// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "TreeFromDimensions.hpp"

namespace llama::mapping::tree::functor
{
    /// Functor for \ref tree::Mapping. Does nothing with the mapping tree. Is used for testing.
    struct Idem
    {
        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree& tree) const -> Tree
        {
            return tree;
        }

        template<typename Tree, typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree&) const
            -> TreeCoord
        {
            return basicCoord;
        }

        template<typename Tree, typename TreeCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree&) const
            -> TreeCoord
        {
            return resultCoord;
        }
    };

    /// Functor for \ref tree::Mapping. Moves all run time parts to the leaves, creating a SoA layout.
    struct LeafOnlyRT
    {
        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(Tree tree) const
        {
            return basicToResultImpl(tree, 1);
        }

        template<typename Tree, typename BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const BasicCoord& basicCoord, const Tree& tree) const
        {
            return basicCoordToResultCoordImpl(basicCoord, tree);
        }

        template<typename Tree, typename ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const ResultCoord& resultCoord, const Tree& tree) const
            -> ResultCoord
        {
            return resultCoord;
        }

    private:
        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
            const Node<Identifier, Type, CountType>& node,
            std::size_t arraySize)
        {
            auto children = tupleTransform(
                node.childs,
                [&](auto element) { return basicToResultImpl(element, LLAMA_COPY(node.count) * arraySize); });
            return Node<Identifier, decltype(children), boost::mp11::mp_size_t<1>>{{}, children};
        }

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE static auto basicToResultImpl(
            const Leaf<Identifier, Type, CountType>& leaf,
            std::size_t arraySize)
        {
            return Leaf<Identifier, Type, std::size_t>{LLAMA_COPY(leaf.count) * arraySize};
        }

        template<typename BasicCoord, typename NodeOrLeaf>
        LLAMA_FN_HOST_ACC_INLINE static auto basicCoordToResultCoordImpl(
            const BasicCoord& basicCoord,
            const NodeOrLeaf& nodeOrLeaf,
            std::size_t arraySize = 0)
        {
            if constexpr(std::tuple_size_v<BasicCoord> == 1)
                return Tuple{TreeCoordElement<BasicCoord::FirstElement::childIndex>{
                    arraySize + LLAMA_COPY(basicCoord.first.arrayIndex)}};
            else
            {
                const auto& branch = get<BasicCoord::FirstElement::childIndex>(nodeOrLeaf.childs);
                auto first = TreeCoordElement<BasicCoord::FirstElement::childIndex, boost::mp11::mp_size_t<0>>{};

                return tupleCat(
                    Tuple{first},
                    basicCoordToResultCoordImpl(
                        basicCoord.rest,
                        branch,
                        (arraySize + LLAMA_COPY(basicCoord.first.arrayIndex)) * LLAMA_COPY(branch.count)));
            }
        }
    };

    namespace internal
    {
        template<typename TreeCoord, typename Node>
        LLAMA_FN_HOST_ACC_INLINE auto getNode(const Node& node)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
                return node;
            else
                return getNode<typename TreeCoord::RestTuple>(get<TreeCoord::FirstElement::childIndex>(node.childs));
        }

        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
            const Node<Identifier, Type, CountType>& tree,
            std::size_t newValue)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
                return Node<Identifier, Type>{newValue, tree.childs};
            else
            {
                auto current = get<TreeCoord::FirstElement::childIndex>(tree.childs);
                auto replacement = changeNodeRuntime<typename TreeCoord::RestTuple>(current, newValue);
                auto children = tupleReplace<TreeCoord::FirstElement::childIndex>(tree.childs, replacement);
                return Node<Identifier, decltype(children)>{tree.count, children};
            }
        }

        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeRuntime(
            const Leaf<Identifier, Type, CountType>& tree,
            std::size_t newValue)
        {
            return Leaf<Identifier, Type, std::size_t>{newValue};
        }

        struct ChangeNodeChildsRuntimeFunctor
        {
            const std::size_t newValue;

            template<typename Identifier, typename Type, typename CountType>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(const Node<Identifier, Type, CountType>& element) const
            {
                return Node<Identifier, Type, std::size_t>{element.count * newValue, element.childs};
            }

            template<typename Identifier, typename Type, typename CountType>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(const Leaf<Identifier, Type, CountType>& element) const
            {
                return Leaf<Identifier, Type, std::size_t>{element.count * newValue};
            }
        };

        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
            const Node<Identifier, Type, CountType>& tree,
            std::size_t newValue)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
            {
                auto children = tupleTransform(tree.childs, ChangeNodeChildsRuntimeFunctor{newValue});
                return Node<Identifier, decltype(children)>{tree.count, children};
            }
            else
            {
                auto current = get<TreeCoord::FirstElement::childIndex>(tree.childs);
                auto replacement = changeNodeChildsRuntime<typename TreeCoord::RestTuple>(current, newValue);
                auto children = tupleReplace<TreeCoord::FirstElement::childIndex>(tree.childs, replacement);
                return Node<Identifier, decltype(children)>{tree.count, children};
            }
        }

        template<typename TreeCoord, typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto changeNodeChildsRuntime(
            const Leaf<Identifier, Type, CountType>& tree,
            std::size_t newValue)
        {
            return tree;
        }
    } // namespace internal

    /// Functor for \ref tree::Mapping. Move the run time part of a node one level down in direction of the leaves by
    /// the given amount (runtime or compile time value).
    /// \tparam TreeCoord tree coordinate in the mapping tree which's run time part shall be moved down one level
    /// \see tree::Mapping
    template<typename TreeCoord, typename Amount = std::size_t>
    struct MoveRTDown
    {
        const Amount amount = {};

        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicToResult(const Tree& tree) const
        {
            return internal::changeNodeChildsRuntime<TreeCoord>(
                internal::changeNodeRuntime<TreeCoord>(
                    tree,
                    // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
                    (internal::getNode<TreeCoord>(tree).count + amount - 1) / amount),
                amount);
        }

        template<typename Tree, typename BasicCoord>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const BasicCoord& basicCoord, const Tree& tree) const
        {
            return basicCoordToResultCoordImpl<TreeCoord>(basicCoord, tree);
        }

        template<typename Tree, typename ResultCoord>
        LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const ResultCoord& resultCoord, const Tree&) const
            -> ResultCoord
        {
            return resultCoord;
        }

    private:
        template<typename InternalTreeCoord, typename BasicCoord, typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoordImpl(const BasicCoord& basicCoord, const Tree& tree) const
        {
            if constexpr(std::is_same_v<InternalTreeCoord, Tuple<>>)
            {
                if constexpr(std::is_same_v<BasicCoord, Tuple<>>)
                    return Tuple{};
                else
                {
                    const auto& childTree = get<BasicCoord::FirstElement::childIndex>(tree.childs);
                    const auto rt1 = basicCoord.first.arrayIndex / amount;
                    const auto rt2
                        = basicCoord.first.arrayIndex % amount * childTree.count + basicCoord.rest.first.arrayIndex;
                    auto rt1Child = TreeCoordElement<BasicCoord::FirstElement::childIndex>{rt1};
                    auto rt2Child = TreeCoordElement<BasicCoord::RestTuple::FirstElement::childIndex>{rt2};
                    return tupleCat(Tuple{rt1Child}, tupleCat(Tuple{rt2Child}, pop_front(basicCoord.rest)));
                }
            }
            else
            {
                if constexpr(InternalTreeCoord::FirstElement::childIndex != BasicCoord::FirstElement::childIndex)
                    return basicCoord;
                else
                {
                    auto rest = basicCoordToResultCoordImpl<typename InternalTreeCoord::RestTuple>(
                        pop_front(basicCoord),
                        get<BasicCoord::FirstElement::childIndex>(tree.childs));
                    return tupleCat(Tuple{basicCoord.first}, rest);
                }
            }
        }
    };

    template<typename TreeCoord, std::size_t Amount>
    using MoveRTDownFixed = MoveRTDown<TreeCoord, boost::mp11::mp_size_t<Amount>>;
} // namespace llama::mapping::tree::functor

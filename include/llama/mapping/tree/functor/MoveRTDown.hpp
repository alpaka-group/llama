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
    namespace internal
    {
        template<typename TreeCoord, typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto getNode(const Tree & tree)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
                return tree;
            else
                return getNode<typename TreeCoord::RestTuple>(
                    getTupleElement<decltype(
                        TreeCoord::FirstElement::compiletime)::value>(
                        tree.childs));
        }

        template<typename TreeCoord, typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto
        changeNodeRuntime(const Tree & tree, std::size_t newValue)
        {
            if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
            {
                if constexpr(HasChildren<Tree>::value)
                    return TreeElement<
                        typename Tree::Identifier,
                        typename Tree::Type>{newValue, tree.childs};
                else
                    return Tree{newValue};
            }
            else
            {
                auto current
                    = getTupleElement<TreeCoord::FirstElement::compiletime>(
                        tree.childs);
                auto replacement
                    = changeNodeRuntime<typename TreeCoord::RestTuple>(
                        current, newValue);
                auto children
                    = tupleReplace<TreeCoord::FirstElement::compiletime>(
                        tree.childs, replacement);
                return TreeElement<
                    typename Tree::Identifier,
                    decltype(children)>(tree.count, children);
            }
        }

        template<typename Operation>
        struct ChangeNodeChildsRuntimeFunctor
        {
            const std::size_t newValue;

            template<typename Element>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(Element element) const
            {
                if constexpr(HasChildren<Element>::value)
                {
                    return TreeElement<
                        typename Element::Identifier,
                        typename Element::Type>(
                        Operation{}(element.count, newValue),
                        element.childs);
                }
                else
                {
                    const auto newCount = Operation{}(element.count, newValue);
                    return TreeElement<
                        typename Element::Identifier,
                        typename Element::Type>{newCount};
                }
            }
        };

        template<typename TreeCoord, typename Operation, typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto
        changeNodeChildsRuntime(Tree const & tree, std::size_t const newValue)
        {
            if constexpr(HasChildren<Tree>::value)
            {
                if constexpr(std::is_same_v<TreeCoord, Tuple<>>)
                {
                    auto children = tupleTransform(
                        tree.childs,
                        ChangeNodeChildsRuntimeFunctor<Operation>{newValue});
                    return TreeElement<
                        typename Tree::Identifier,
                        decltype(children)>(tree.count, children);
                }
                else
                {
                    auto current
                        = getTupleElement<TreeCoord::FirstElement::compiletime>(
                            tree.childs);
                    auto replacement = changeNodeChildsRuntime<
                        typename TreeCoord::RestTuple,
                        Operation>(current, newValue);
                    auto children
                        = tupleReplace<TreeCoord::FirstElement::compiletime>(
                            tree.childs, replacement);
                    return TreeElement<
                        typename Tree::Identifier,
                        decltype(children)>(tree.count, children);
                }
            }
            else
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
            return internal::changeNodeChildsRuntime<TreeCoord, std::multiplies<>>(
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
                    const auto & childTree = getTupleElementRef<
                        BasicCoord::FirstElement::compiletime>(tree.childs);
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
                        tupleCat(Tuple{rt2Child}, tupleRest(basicCoord.rest)));
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
                        tupleRest(basicCoord),
                        getTupleElementRef<
                            BasicCoord::FirstElement::compiletime>(
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

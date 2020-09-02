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
#include "../../Types.hpp"
#include "../../UserDomain.hpp"
#include "Functors.hpp"
#include "TreeFromDomains.hpp"
#include "toString.hpp"

#include <type_traits>

namespace llama::mapping::tree
{
    namespace internal
    {
        template<typename Tree, typename TreeOperationList>
        struct MergeFunctors
        {};

        template<typename Tree, typename... Operations>
        struct MergeFunctors<Tree, Tuple<Operations...>>
        {
            boost::mp11::mp_first<Tuple<Operations...>> operation = {};
            using ResultTree = decltype(operation.basicToResult(Tree()));
            ResultTree treeAfterOp;
            MergeFunctors<
                ResultTree,
                boost::mp11::mp_drop_c<Tuple<Operations...>, 1>>
                next = {};

            MergeFunctors() = default;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctors(
                const Tree & tree,
                const Tuple<Operations...> & treeOperationList) :
                    operation(treeOperationList.first),
                    treeAfterOp(operation.basicToResult(tree)),
                    next(treeAfterOp, tupleRest(treeOperationList))
            {}

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const Tree & tree) const
            {
                if constexpr(sizeof...(Operations) > 1)
                    return next.basicToResult(treeAfterOp);
                else if constexpr(sizeof...(Operations) == 1)
                    return operation.basicToResult(tree);
                else
                    return tree;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
                const TreeCoord & basicCoord,
                const Tree & tree) const
            {
                if constexpr(sizeof...(Operations) >= 1)
                    return next.basicCoordToResultCoord(
                        operation.basicCoordToResultCoord(basicCoord, tree),
                        treeAfterOp);
                else
                    return basicCoord;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
                const TreeCoord & resultCoord,
                const Tree & tree) const
            {
                if constexpr(sizeof...(Operations) >= 1)
                    return next.resultCoordToBasicCoord(
                        operation.resultCoordToBasicCoord(resultCoord, tree),
                        operation.basicToResult(tree));
                else
                    return resultCoord;
            }
        };

        template<typename Tree>
        struct MergeFunctors<Tree, Tuple<>>
        {
            MergeFunctors() = default;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctors(const Tree &, const Tuple<> & treeOperationList) {}

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const Tree & tree) const
            {
                return tree;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(
                TreeCoord const & basicCoord,
                Tree const & tree) const -> TreeCoord
            {
                return basicCoord;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(
                TreeCoord const & resultCoord,
                Tree const & tree) const -> TreeCoord
            {
                return resultCoord;
            }
        };

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobSize(const Node<Identifier, Type, CountType> & node)
            -> std::size_t;

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobSize(const Leaf<Identifier, Type, CountType> & leaf)
            -> std::size_t;

        template<typename... Children, std::size_t... Is, typename Count>
        LLAMA_FN_HOST_ACC_INLINE auto getChildrenBlobSize(
            const Tuple<Children...> & childs,
            std::index_sequence<Is...> ii,
            const Count & count) -> std::size_t
        {
            return count * (getTreeBlobSize(getTupleElement<Is>(childs)) + ...);
        }

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobSize(const Node<Identifier, Type, CountType> & node)
            -> std::size_t
        {
            constexpr std::size_t childCount = boost::mp11::mp_size<
                std::decay_t<decltype(node.childs)>>::value;
            return getChildrenBlobSize(
                node.childs,
                std::make_index_sequence<childCount>{},
                LLAMA_DEREFERENCE(node.count));
        }

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobSize(const Leaf<Identifier, Type, CountType> & leaf)
            -> std::size_t
        {
            return leaf.count * sizeof(Type);
        }

        template<typename Childs, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobSize(const Childs & childs, const CountType & count)
            -> std::size_t
        {
            return getTreeBlobSize(
                Node<NoName, Childs, CountType>{count, childs});
        }

        namespace internal
        {
            template<
                std::size_t MaxPos,
                typename Identifier,
                typename Type,
                typename CountType,
                std::size_t... Is>
            LLAMA_FN_HOST_ACC_INLINE auto sumChildrenSmallerThan(
                const Node<Identifier, Type, CountType> & node,
                std::index_sequence<Is...>) -> std::size_t
            {
                return (
                    (getTreeBlobSize(getTupleElementRef<Is>(node.childs))
                     * (Is < MaxPos))
                    + ...);
            }
        }

        template<typename Tree, typename... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobByte(const Tree & tree, const Tuple<Coords...> & treeCoord)
            -> std::size_t
        {
            if constexpr(sizeof...(Coords) > 1)
                return getTreeBlobSize(
                           tree.childs,
                           LLAMA_DEREFERENCE(treeCoord.first.runtime))
                    + internal::sumChildrenSmallerThan<
                           treeCoord.first.compiletime>(
                           tree,
                           std::make_index_sequence<
                               SizeOfTuple<typename Tree::ChildrenTuple>>{})
                    + getTreeBlobByte(
                           getTupleElementRef<treeCoord.first.compiletime>(
                               tree.childs),
                           treeCoord.rest);
            else
                return sizeof(typename Tree::Type) * treeCoord.first.runtime;
        }
    }

    /** Free describable mapping which can be used for creating a \ref View with
     * a \ref Factory. For the interface details see \ref Factory. \tparam
     * T_UserDomain type of the user domain \tparam T_DatumDomain type of the
     * datum domain \tparam TreeOperationList (\ref Tuple) of operations to
     * define the tree mapping
     */
    template<
        typename T_UserDomain,
        typename T_DatumDomain,
        typename TreeOperationList>
    struct Mapping
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        using BasicTree = TreeFromDomains<UserDomain, DatumDomain>;
        // TODO, support more than one blob
        static constexpr std::size_t blobCount = 1;

        using MergedFunctors
            = internal::MergeFunctors<BasicTree, TreeOperationList>;

        UserDomain userDomainSize = {};
        BasicTree basicTree;
        MergedFunctors mergedFunctors;

        using ResultTree = decltype(mergedFunctors.basicToResult(basicTree));
        ResultTree resultTree;

        Mapping() = default;

        /** The initalization of this mapping needs a \ref Tuple of operations
         *  which describe the mapping in detail. Please have a look at the user
         *  documenation for more information.
         * \param size the size of the user domain
         * \param treeOperationList list of operations to define the mapping,
         * e.g. \ref functor::Idem, \ref functor::LeafOnlyRT, \ref
         * functor::MoveRTDown.
         */
        LLAMA_FN_HOST_ACC_INLINE
        Mapping(UserDomain size, TreeOperationList treeOperationList) :
                userDomainSize(size),
                basicTree(createTree<DatumDomain>(size)),
                mergedFunctors(basicTree, treeOperationList),
                resultTree(mergedFunctors.basicToResult(basicTree))
        {}

        LLAMA_FN_HOST_ACC_INLINE
        auto getBlobSize(std::size_t const) const -> std::size_t
        {
            return internal::getTreeBlobSize(resultTree);
        }

        template<std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain coord) const
            -> NrAndOffset
        {
            auto const basicTreeCoord
                = createTreeCoord<DatumCoord<DatumDomainCoord...>>(coord);
            auto const resultTreeCoord = mergedFunctors.basicCoordToResultCoord(
                basicTreeCoord, basicTree);
            const auto offset
                = internal::getTreeBlobByte(resultTree, resultTreeCoord);
            return {0, offset};
        }
    };
}

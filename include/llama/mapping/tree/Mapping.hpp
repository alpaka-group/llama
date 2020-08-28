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

#include "../../Types.hpp"
#include "../../UserDomain.hpp"
#include "MergeFunctors.hpp"
#include "Reduce.hpp"
#include "TreeElement.hpp"
#include "TreeFromDomains.hpp"
#include "functor/Idem.hpp"
#include "functor/LeafOnlyRT.hpp"
#include "functor/MoveRTDown.hpp"
#include "toString.hpp"

#include <type_traits>

namespace llama::mapping::tree
{
    namespace internal
    {
        template<typename Type, typename CountType>
        struct SizeOfFunctor
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(const CountType &) const -> std::size_t
            {
                return sizeof(Type);
            }
        };

        template<typename Childs, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto
        getTreeBlobSize(const Childs & childs, const CountType & count)
            -> std::size_t
        {
            return Reduce<
                TreeElement<NoName, Childs, CountType>,
                std::plus<>,
                std::multiplies<>,
                SizeOfFunctor>()(childs, count);
        }

        template<typename Tree>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Tree & tree)
            -> std::size_t
        {
            return Reduce<
                Tree,
                std::plus<>,
                std::multiplies<>,
                SizeOfFunctor>()(tree);
        }

        template<typename Tree, std::size_t Pos>
        LLAMA_FN_HOST_ACC_INLINE auto summarizeTreeSmallerPos(
            typename Tree::Type const & childs,
            decltype(Tree::count) const & count) -> std::size_t
        {
            if constexpr(Pos == 0)
                return 0;
            else
                return getTreeBlobSize(childs.first)
                    + summarizeTreeSmallerPos<
                           typename TreePopFrontChild<Tree>::ResultType,
                           Pos - 1>(childs.rest, count);
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
                    + summarizeTreeSmallerPos<
                           Tree,
                           treeCoord.first.compiletime>(
                           tree.childs, LLAMA_DEREFERENCE(tree.count))
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

        using MergedFunctors = MergeFunctors<BasicTree, TreeOperationList>;

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

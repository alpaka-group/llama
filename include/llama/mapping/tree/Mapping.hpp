// Copyright 2020 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "../Common.hpp"
#include "Functors.hpp"
#include "TreeFromDimensions.hpp"
#include "toString.hpp"

#include <type_traits>

namespace llama::mapping::tree
{
    namespace internal
    {
        template<typename Tree, typename TreeOperationList>
        struct MergeFunctors
        {
        };

        template<typename Tree, typename... Operations>
        struct MergeFunctors<Tree, Tuple<Operations...>>
        {
            mp_first<Tuple<Operations...>> operation = {};
            using ResultTree = decltype(operation.basicToResult(Tree()));
            ResultTree treeAfterOp;
            MergeFunctors<ResultTree, mp_drop_c<Tuple<Operations...>, 1>> next = {};

            MergeFunctors() = default;

            LLAMA_FN_HOST_ACC_INLINE
            MergeFunctors(const Tree& tree, const Tuple<Operations...>& treeOperationList)
                : operation(treeOperationList.first())
                , treeAfterOp(operation.basicToResult(tree))
                , next(treeAfterOp, popFront(treeOperationList))
            {
            }

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const Tree& tree) const
            {
                if constexpr(sizeof...(Operations) > 1)
                    return next.basicToResult(treeAfterOp);
                else if constexpr(sizeof...(Operations) == 1)
                    return operation.basicToResult(tree);
                else
                    return tree;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree& tree) const
            {
                if constexpr(sizeof...(Operations) >= 1)
                    return next.basicCoordToResultCoord(
                        operation.basicCoordToResultCoord(basicCoord, tree),
                        treeAfterOp);
                else
                    return basicCoord;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree& tree) const
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
            MergeFunctors(const Tree&, const Tuple<>&)
            {
            }

            LLAMA_FN_HOST_ACC_INLINE
            auto basicToResult(const Tree& tree) const
            {
                return tree;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto basicCoordToResultCoord(const TreeCoord& basicCoord, const Tree& /*tree*/)
                const -> TreeCoord
            {
                return basicCoord;
            }

            template<typename TreeCoord>
            LLAMA_FN_HOST_ACC_INLINE auto resultCoordToBasicCoord(const TreeCoord& resultCoord, const Tree& /*tree*/)
                const -> TreeCoord
            {
                return resultCoord;
            }
        };

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t;

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t;

        template<typename... Children, std::size_t... Is, typename Count>
        LLAMA_FN_HOST_ACC_INLINE auto getChildrenBlobSize(
            const Tuple<Children...>& childs,
            std::index_sequence<Is...> /*ii*/,
            const Count& count) -> std::size_t
        {
            return count * (getTreeBlobSize(get<Is>(childs)) + ...);
        }

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Node<Identifier, Type, CountType>& node) -> std::size_t
        {
            constexpr std::size_t childCount = mp_size<std::decay_t<decltype(node.childs)>>::value;
            return getChildrenBlobSize(node.childs, std::make_index_sequence<childCount>{}, LLAMA_COPY(node.count));
        }

        template<typename Identifier, typename Type, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Leaf<Identifier, Type, CountType>& leaf) -> std::size_t
        {
            return leaf.count * sizeof(Type);
        }

        template<typename Childs, typename CountType>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobSize(const Childs& childs, const CountType& count) -> std::size_t
        {
            return getTreeBlobSize(Node<NoName, Childs, CountType>{count, childs});
        }

        template<std::size_t MaxPos, typename Identifier, typename Type, typename CountType, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto sumChildrenSmallerThan(
            const Node<Identifier, Type, CountType>& node,
            std::index_sequence<Is...>) -> std::size_t
        {
            return ((getTreeBlobSize(get<Is>(node.childs)) * (Is < MaxPos)) + ...);
        }

        template<typename Tree, typename... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto getTreeBlobByte(const Tree& tree, const Tuple<Coords...>& treeCoord)
            -> std::size_t
        {
            const auto firstArrayIndex = treeCoord.first().arrayIndex;
            if constexpr(sizeof...(Coords) > 1)
            {
                constexpr auto firstChildIndex = decltype(treeCoord.first().childIndex)::value;
                return getTreeBlobSize(tree.childs, firstArrayIndex)
                    + sumChildrenSmallerThan<firstChildIndex>(
                           tree,
                           std::make_index_sequence<std::tuple_size_v<typename Tree::ChildrenTuple>>{})
                    + getTreeBlobByte(get<firstChildIndex>(tree.childs), treeCoord.rest());
            }
            else
                return sizeof(typename Tree::Type) * firstArrayIndex;
        }
    } // namespace internal

    /// An experimental attempt to provide a general purpose description of a mapping. \ref Array and record
    /// dimensions are represented by a compile time tree data structure. This tree is mapped into memory by means of a
    /// breadth-first tree traversal. By specifying additional tree operations, the tree can be modified at compile
    /// time before being mapped to memory.
    template<typename TArrayExtents, typename TRecordDim, typename TreeOperationList>
    struct Mapping : private TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;

        // TODO(bgruber): , support more than one blob
        static constexpr std::size_t blobCount = 1;

    private:
        using size_type = typename ArrayExtents::value_type;

    public:
        using BasicTree = TreeFromDimensions<ArrayExtents, RecordDim>;
        using MergedFunctors = internal::MergeFunctors<BasicTree, TreeOperationList>;
        BasicTree basicTree;
        MergedFunctors mergedFunctors;

        using ResultTree = decltype(mergedFunctors.basicToResult(basicTree));
        ResultTree resultTree;

        Mapping() = default;

        LLAMA_FN_HOST_ACC_INLINE
        Mapping(ArrayExtents extents, TreeOperationList treeOperationList, RecordDim = {})
            : ArrayExtents(extents)
            , basicTree(createTree<RecordDim>(extents.toArray()))
            , mergedFunctors(basicTree, treeOperationList)
            , resultTree(mergedFunctors.basicToResult(basicTree))
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto extents() const -> ArrayExtents
        {
            return static_cast<const ArrayExtents&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto blobSize(size_type const) const -> size_type
        {
            // TODO(bgruber): propagate use of size_type
            return internal::getTreeBlobSize(resultTree);
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> = {}) const
            -> NrAndOffset<size_type>
        {
            // TODO(bgruber): propagate use of size_type
            const auto basicTreeCoord = createTreeCoord<RecordCoord<RecordCoords...>>(ai);
            const auto resultTreeCoord = mergedFunctors.basicCoordToResultCoord(basicTreeCoord, basicTree);
            const auto offset = static_cast<size_type>(internal::getTreeBlobByte(
                resultTree,
                resultTreeCoord)); // FIXME(bgruber): size_type should be propagated through getTreeBlobByte
            return {0, offset};
        }
    };
} // namespace llama::mapping::tree

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

#include "TreeElement.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

namespace internal
{

template<
    typename T_Tree,
    typename T_TreeOperationList
>
struct MergeFunctorsImpl
{
    using FirstOp = typename T_TreeOperationList::FirstElement;
    using SubMergeFunctorsImpl = MergeFunctorsImpl<
    typename FirstOp::template Result< T_Tree >,
    typename T_TreeOperationList::RestTuple
    >;
    using Result = typename SubMergeFunctorsImpl::Result;

    FirstOp const operation;
    decltype( operation.template basicToResult( T_Tree() ) ) const treeAfterOp;
    SubMergeFunctorsImpl const subMergeFunctorsImpl;

    MergeFunctorsImpl(
        T_Tree const & tree,
        T_TreeOperationList const & treeOperationList
    ) :
        operation( treeOperationList.first ),
        treeAfterOp( operation.template basicToResult( tree ) ),
        subMergeFunctorsImpl(
            treeAfterOp,
            treeOperationList.rest
        )
    { }

    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicToResult( T_Tree const & ) const
    -> Result
    {
        return subMergeFunctorsImpl.basicToResult(
            treeAfterOp
        );
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicCoordToResultCoord(
        T_TreeCoord const & basicCoord,
        T_Tree const & tree
    ) const
    -> decltype(
        subMergeFunctorsImpl.basicCoordToResultCoord(
            operation.template basicCoordToResultCoord(
                basicCoord,
                tree
            ),
            treeAfterOp
        )
    )
    {
        return subMergeFunctorsImpl.basicCoordToResultCoord(
            operation.template basicCoordToResultCoord(
            basicCoord,
            tree
            ),
            treeAfterOp
        );
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    resultCoordToBasicCoord(
        T_TreeCoord const & resultCoord,
        T_Tree const & tree
    ) const
    -> decltype(
        subMergeFunctorsImpl.resultCoordToBasicCoord(
            operation.template resultCoordToBasicCoord(
                resultCoord,
                tree
            ),
            operation.template basicToResult( tree )
        )
    )
    {
        return subMergeFunctorsImpl.resultCoordToBasicCoord(
            operation.template resultCoordToBasicCoord(
                resultCoord,
                tree
            ),
            operation.template basicToResult( tree )
        );
    }
};

template<
    typename T_Tree,
    typename T_LastOperation
>
struct MergeFunctorsImpl<
    T_Tree,
    Tuple< T_LastOperation >
>
{
    using LastOp = T_LastOperation;
    using Result = typename LastOp::template Result< T_Tree >;
    using TreeOperationList = Tuple< T_LastOperation >;

    LastOp const operation;

    MergeFunctorsImpl(
        T_Tree const &,
        TreeOperationList const & treeOperationList
    ) :
        operation( treeOperationList.first )
    { }

    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicToResult( const T_Tree & tree ) const
    -> Result
    {
        return operation.template basicToResult( tree );
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicCoordToResultCoord(
        T_TreeCoord const & basicCoord,
        T_Tree const & tree
    ) const
    -> decltype(
        operation.template basicCoordToResultCoord(
            basicCoord,
            tree
        )
    )
    {
        return operation.template basicCoordToResultCoord(
            basicCoord,
            tree
        );
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    resultCoordToBasicCoord(
        T_TreeCoord const & resultCoord,
        T_Tree const & tree
    ) const
    -> decltype(
        operation.template resultCoordToBasicCoord(
            resultCoord,
            tree
        )
    )
    {
        return operation.template resultCoordToBasicCoord(
            resultCoord,
            tree
        );
    }
};

template< typename T_Tree >
struct MergeFunctorsImpl<
    T_Tree,
    Tuple< >
>
{
    using Result = T_Tree;
    using TreeOperationList = Tuple< >;

    MergeFunctorsImpl(
        T_Tree const &,
        TreeOperationList const & treeOperationList
    ) { }

    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicToResult( const T_Tree & tree ) const
    -> Result
    {
        return tree;
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicCoordToResultCoord(
        T_TreeCoord const & basicCoord,
        T_Tree const & tree
    ) const
    -> T_TreeCoord
    {
        return basicCoord;
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    resultCoordToBasicCoord(
        T_TreeCoord const & resultCoord,
        T_Tree const & tree
    ) const
    -> T_TreeCoord
    {
        return resultCoord;
    }
};

} //namespace internal

template<
    typename T_Tree,
    typename T_TreeOperationList
>
struct MergeFunctors
{
    using MergeFunctorsImpl = typename internal::MergeFunctorsImpl<
        T_Tree,
        T_TreeOperationList
    >;
    using Result = typename MergeFunctorsImpl::Result;

    MergeFunctorsImpl const mergeFunctorsImpl;

    MergeFunctors(
        T_Tree const & tree,
        T_TreeOperationList const & treeOperationList
    ) :
        mergeFunctorsImpl(
            tree,
            treeOperationList
        )
    { }

    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicToResult( T_Tree const & tree ) const
    -> Result
    {
        return mergeFunctorsImpl.basicToResult( tree );
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    basicCoordToResultCoord(
        T_TreeCoord const & basicCoord,
        T_Tree const & tree
    ) const
    -> decltype(
        mergeFunctorsImpl.basicCoordToResultCoord(
            basicCoord,
            tree
        )
    )
    {
        return mergeFunctorsImpl.basicCoordToResultCoord(
            basicCoord,
            tree
        );
    }

    template< typename T_TreeCoord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    resultCoordToBasicCoord(
        T_TreeCoord const & resultCoord,
        T_Tree const & tree
    ) const
    -> decltype(
        mergeFunctorsImpl.resultCoordToBasicCoord(
            resultCoord,
            tree
        )
    )
    {
        return mergeFunctorsImpl.resultCoordToBasicCoord(
            resultCoord,
            tree
        );
    }
};

} // namespace tree

} // namespace mapping

} // namespace llama

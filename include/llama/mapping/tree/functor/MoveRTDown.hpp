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

template< typename T_TreeCoord >
struct MoveRTDown
{
    template<
        typename T_Tree,
        typename T_InternalTreeCoord,
        typename T_BasicCoord,
        typename T_SFINAE = void
    >
    struct BasicCoordToResultCoordImpl;

    template<
        typename T_Tree,
        typename T_InternalTreeCoord,
        typename T_BasicCoord
    >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        T_InternalTreeCoord,
        T_BasicCoord,
        typename std::enable_if<
            T_InternalTreeCoord::FirstElement::compiletime !=
            T_BasicCoord::FirstElement::compiletime
        >::type
    >
    {
        auto
        operator()(
            T_BasicCoord const basicCoord,
            T_Tree const tree,
            std::size_t const amount
        ) const
        -> T_BasicCoord
        {
            return basicCoord;
        }
    };

    template<
        typename T_Tree,
        typename T_InternalTreeCoord,
        typename T_BasicCoord
    >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        T_InternalTreeCoord,
        T_BasicCoord,
        typename std::enable_if<
            T_InternalTreeCoord::FirstElement::compiletime ==
            T_BasicCoord::FirstElement::compiletime
        >::type
    >
    {
        auto
        operator()(
            T_BasicCoord const basicCoord,
            T_Tree const tree,
            std::size_t const amount
        ) const
        -> T_BasicCoord
        {
            return T_BasicCoord(
                basicCoord.first,
                BasicCoordToResultCoordImpl<
                    GetTupleType<
                        typename T_Tree::Type,
                        T_BasicCoord::FirstElement::compiletime
                    >,
                    decltype( tupleRest( T_InternalTreeCoord() ) ),
                    decltype( tupleRest( basicCoord ) )
                >()(
                    tupleRest( basicCoord ),
                    getTupleElement<
                        T_BasicCoord::FirstElement::compiletime
                    >( tree.childs ),
                    amount
                )
            );
        }
    };

    template<
        typename T_Tree,
        typename T_BasicCoord
    >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        Tuple< >,
        T_BasicCoord,
        void
    >
    {
        auto
        operator()(
            T_BasicCoord const basicCoord,
            T_Tree const tree,
            std::size_t const amount
        ) const
        -> T_BasicCoord
        {
            auto const childTree = getTupleElement<
                basicCoord.first.compiletime
            >( tree.childs );
            auto const realAmount = amount ? amount : tree.count;
            auto const rt1 = basicCoord.first.runtime / realAmount;
            auto const rt2 =
                basicCoord.first.runtime % realAmount * childTree.count +
                basicCoord.rest.first.runtime;
            return T_BasicCoord(
                TreeCoordElement< basicCoord.first.compiletime >( rt1 ),
                typename T_BasicCoord::RestTuple(
                    TreeCoordElement< basicCoord.rest.first.compiletime >( rt2),
                    tupleRest( basicCoord.rest )
                )
            );
        }
    };

    template< typename T_Tree >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        Tuple< >,
        Tuple< >,
        void
    >
    {
        auto
        operator()(
            Tuple< > const basicCoord,
            T_Tree const tree,
            std::size_t const amount
        ) const
        -> Tuple< >
        {
            return Tuple< >();
        }
    };


    std::size_t const amount;

    MoveRTDown( std::size_t const amount = 0 ) : amount( amount ) {}

    template< typename T_Tree >
    using Result = decltype(
        operations::changeNodeChildsRuntime<
            T_TreeCoord,
            Multiplication
        >(
            operations::changeNodeRuntime< T_TreeCoord >(
                T_Tree(),
                amount ?
                    ( (operations::getNode< T_TreeCoord >( T_Tree() ).count
                        + amount - 1 ) / amount ) :
                    1
            ),
            amount ? amount : operations::getNode< T_TreeCoord >( T_Tree() ).count
        )
    );

    template< typename T_Tree >
    auto
    basicToResult( T_Tree const tree ) const
    -> Result< T_Tree >
    {
        return operations::changeNodeChildsRuntime<
            T_TreeCoord,
            Multiplication
        >(
            operations::changeNodeRuntime< T_TreeCoord >(
                tree,
                amount ?
                    ( (operations::getNode< T_TreeCoord >( tree ).count
                        + amount - 1 ) / amount ) :
                    1
            ),
            amount ? amount : operations::getNode< T_TreeCoord >( tree ).count
        );
    }

    template<
        typename T_Tree,
        typename T_BasicCoord
    >
    auto
    basicCoordToResultCoord(
        T_BasicCoord const basicCoord,
        T_Tree const tree
    ) const
    -> T_BasicCoord
    {
        return BasicCoordToResultCoordImpl<
            T_Tree,
            T_TreeCoord,
            T_BasicCoord
        >()(
            basicCoord,
            tree,
            amount
        );
    }

    template<
        typename T_Tree,
        typename T_ResultCoord
    >
    auto
    resultCoordToBasicCoord(
        T_ResultCoord const resultCoord,
        T_Tree const tree
    ) const
    -> T_ResultCoord
    {
        return resultCoord;
    }
};

template<
    typename T_TreeCoord,
    std::size_t T_amount
>
struct MoveRTDownFixed
{
    static constexpr std::size_t amount = T_amount;
    template<
        typename T_Tree,
        typename T_InternalTreeCoord,
        typename T_BasicCoord,
        typename T_SFINAE = void
    >
    struct BasicCoordToResultCoordImpl;

    template<
        typename T_Tree,
        typename T_InternalTreeCoord,
        typename T_BasicCoord
    >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        T_InternalTreeCoord,
        T_BasicCoord,
        typename std::enable_if<
            T_InternalTreeCoord::FirstElement::compiletime !=
            T_BasicCoord::FirstElement::compiletime
        >::type
    >
    {
        auto
        operator()(
            T_BasicCoord const basicCoord,
            T_Tree const tree
        ) const
        -> T_BasicCoord
        {
            return basicCoord;
        }
    };

    template<
        typename T_Tree,
        typename T_InternalTreeCoord,
        typename T_BasicCoord
    >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        T_InternalTreeCoord,
        T_BasicCoord,
        typename std::enable_if<
            T_InternalTreeCoord::FirstElement::compiletime ==
            T_BasicCoord::FirstElement::compiletime
        >::type
    >
    {
        auto
        operator()(
            T_BasicCoord const basicCoord,
            T_Tree const tree
        ) const
        -> T_BasicCoord
        {
            return T_BasicCoord(
                basicCoord.first,
                BasicCoordToResultCoordImpl<
                    GetTupleType<
                        typename T_Tree::Type,
                        T_BasicCoord::FirstElement::compiletime
                    >,
                    decltype( tupleRest( T_InternalTreeCoord() ) ),
                    decltype( tupleRest( basicCoord ) )
                >()(
                    tupleRest( basicCoord ),
                    getTupleElement<
                        T_BasicCoord::FirstElement::compiletime
                    >( tree.childs )
                )
            );
        }
    };

    template<
        typename T_Tree,
        typename T_BasicCoord
    >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        Tuple< >,
        T_BasicCoord,
        void
    >
    {
        auto
        operator()(
            T_BasicCoord const basicCoord,
            T_Tree const tree
        ) const
        -> T_BasicCoord
        {
            auto const childTree = getTupleElement<
                basicCoord.first.compiletime
            >( tree.childs );
            auto const rt1 = basicCoord.first.runtime / amount;
            auto const rt2 =
                basicCoord.first.runtime % amount * childTree.count +
                basicCoord.rest.first.runtime;
            return T_BasicCoord(
                TreeCoordElement< basicCoord.first.compiletime >( rt1 ),
                typename T_BasicCoord::RestTuple(
                    TreeCoordElement< basicCoord.rest.first.compiletime >( rt2),
                    tupleRest( basicCoord.rest )
                )
            );
        }
    };

    template< typename T_Tree >
    struct BasicCoordToResultCoordImpl<
        T_Tree,
        Tuple< >,
        Tuple< >,
        void
    >
    {
        auto
        operator()(
            Tuple< > const basicCoord,
            T_Tree const tree
        ) const
        -> Tuple< >
        {
            return Tuple< >();
        }
    };

    template< typename T_Tree >
    using Result = decltype( operations::changeNodeChildsRuntime<
            T_TreeCoord,
            Multiplication
        >(
            operations::changeNodeRuntime< T_TreeCoord >(
                T_Tree(),
                ( operations::getNode< T_TreeCoord >( T_Tree() ).count
                    + amount - 1 ) / amount
            ),
            amount
        )
    );

    template< typename T_Tree >
    auto
    basicToResult( T_Tree const tree ) const
    -> Result< T_Tree >
    {
        return operations::changeNodeChildsRuntime<
            T_TreeCoord,
            Multiplication
        >(
            operations::changeNodeRuntime< T_TreeCoord >(
                tree,
                ( operations::getNode< T_TreeCoord >( tree ).count
                    + amount - 1 ) / amount
            ),
            amount
        );
    }

    template<
        typename T_Tree,
        typename T_BasicCoord
    >
    auto
    basicCoordToResultCoord(
        T_BasicCoord const basicCoord,
        T_Tree const tree
    ) const
    -> T_BasicCoord
    {
        return BasicCoordToResultCoordImpl<
            T_Tree,
            T_TreeCoord,
            T_BasicCoord
        >()(
            basicCoord,
            tree
        );
    }

    template<
        typename T_Tree,
        typename T_ResultCoord
    >
    auto
    resultCoordToBasicCoord(
        T_ResultCoord const resultCoord,
        T_Tree const tree
    ) const
    -> T_ResultCoord
    {
        return resultCoord;
    }
};

} // namespace functor

} // namespace tree

} // namespace mapping

} // namespace llama

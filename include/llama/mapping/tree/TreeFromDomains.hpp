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

#include <cstddef>
#include <type_traits>

#include "TreeElement.hpp"
#include "TreeCoord.hpp"

namespace llama
{

namespace mapping
{

namespace tree
{

namespace internal
{

// General case (leave)
template< typename T_DatumDomain >
struct ReplaceDEwithTEinDD
{
    using type = T_DatumDomain;
};

template< typename T_DatumElement >
struct ReplaceDEwithTEinDE;

template<
    typename T_Identifier,
    typename T_Type
>
struct ReplaceDEwithTEinDE<
    DatumElement<
        T_Identifier,
        T_Type
    >
>
{
    using type = TreeElementConst<
        T_Identifier,
        typename ReplaceDEwithTEinDD< T_Type >::type
    >;
};

// DatumStruct case (node)
template<
    typename T_FirstDatumElement,
    typename... T_DatumElements
>
struct ReplaceDEwithTEinDD<
    DatumStruct<
        T_FirstDatumElement,
        T_DatumElements...
    >
>
{
    using type = boost::mp11::mp_push_front<
        typename ReplaceDEwithTEinDD< DatumStruct< T_DatumElements... > >::type,
        typename ReplaceDEwithTEinDE< T_FirstDatumElement >::type
    >;
};

// empty DatumStruct case ( recursive loop head )
template< >
struct ReplaceDEwithTEinDD< DatumStruct< > >
{
    using type = Tuple< >;
};

template<
    typename T_Leave,
    std::size_t T_count
>
struct CreateTreeChain
{
    using type = TreeElement<
        NoName,
        Tuple< typename CreateTreeChain<
            T_Leave,
            T_count - 1
        >::type >
    >;
};

template< typename T_Leave >
struct CreateTreeChain<
    T_Leave,
    0
>
{
    using type = T_Leave;
};

template< typename T_DatumDomain >
struct TreeFromDatumDomainImpl
{
    using type = TreeElement<
        NoName,
        typename ReplaceDEwithTEinDD< T_DatumDomain >::type
    >;
};

template<
    typename T_UserDomain,
    typename T_DatumDomain
>
struct TreeFromDomainsImpl
{
    using type = typename CreateTreeChain<
        typename TreeFromDatumDomainImpl< T_DatumDomain >::type,
        T_UserDomain::count - 1
    >::type;
};

} // internal

template< typename T_DatumDomain >
using TreeFromDatumDomain = typename internal::TreeFromDatumDomainImpl<
    T_DatumDomain
>::type;

template<
    typename T_UserDomain,
    typename T_DatumDomain
>
using TreeFromDomains = typename internal::TreeFromDomainsImpl<
    T_UserDomain,
    T_DatumDomain
>::type;

namespace internal
{

template<
    typename T_DatumDomain,
    typename T_UserDomain,
    typename T_pos = std::integral_constant<
        std::size_t,
        0
    >
>
struct SetUserDomainInTreeImpl
{
    using InnerTuple = Tuple<
        decltype(
            SetUserDomainInTreeImpl<
                T_DatumDomain,
                T_UserDomain,
                std::integral_constant<
                    std::size_t,
                    T_pos::value + 1
                >
            >()( T_UserDomain() )
        )
    >;

    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_UserDomain const & size ) const
    -> TreeElement<
            NoName,
            InnerTuple
        >
    {
        return TreeElement<
            NoName,
            InnerTuple
        >(
            size[ T_pos::value ],
            InnerTuple(
                SetUserDomainInTreeImpl<
                    T_DatumDomain,
                    T_UserDomain,
                    std::integral_constant<
                        std::size_t,
                        T_pos::value + 1
                    >
                >()( size )
            )
        );
    }
};

template<
    typename T_DatumDomain,
    typename T_UserDomain
>
struct SetUserDomainInTreeImpl<
    T_DatumDomain,
    T_UserDomain,
    std::integral_constant<
        std::size_t,
        T_UserDomain::count - 1
    >
>
{
    LLAMA_NO_HOST_ACC_WARNING
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_UserDomain const & size ) const
    -> TreeFromDatumDomain< T_DatumDomain >
    {
        return TreeFromDatumDomain< T_DatumDomain >( size[ T_UserDomain::count - 1 ] );
    }
};

} // internal

template<
    typename T_DatumDomain,
    typename T_UserDomain
>
LLAMA_FN_HOST_ACC_INLINE
auto
setUserDomainInTree( T_UserDomain const & size )
-> decltype(
    internal::SetUserDomainInTreeImpl<
        T_DatumDomain,
        T_UserDomain
    >()( size )
)
{
    return internal::SetUserDomainInTreeImpl<
        T_DatumDomain,
        T_UserDomain
    >()( size );
};

namespace internal
{

template<
    typename T_UserDomain,
    std::size_t T_firstDatumDomain,
    typename T_pos = std::integral_constant<
        std::size_t,
        0
    >
>
struct UserDomainToTreeCoord
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_UserDomain const & coord ) const
    -> decltype(
        tupleCat(
            Tuple< TreeCoordElement< 0 > >(),
            UserDomainToTreeCoord<
                T_UserDomain,
                T_firstDatumDomain,
                std::integral_constant<
                    std::size_t,
                    T_pos::value + 1
                >
            >()( coord )
        )
    )
    {
        const Tuple< TreeCoordElement< 0 > > mostLeft{
            TreeCoordElement< 0 >( coord[ T_pos::value ] )
        };
        return tupleCat(
            mostLeft,
            UserDomainToTreeCoord<
                T_UserDomain,
                T_firstDatumDomain,
                std::integral_constant<
                    std::size_t,
                    T_pos::value + 1
                >
            >()( coord )
        );
    }
};

template<
    typename T_UserDomain,
    std::size_t T_firstDatumDomain
>
struct UserDomainToTreeCoord<
    T_UserDomain,
    T_firstDatumDomain,
    std::integral_constant<
        std::size_t,
        T_UserDomain::count - 1
    >
>
{
    using Result = Tuple< TreeCoordElement< T_firstDatumDomain > >;

    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_UserDomain const & coord ) const
    -> Result
    {
        return Result( TreeCoordElement< T_firstDatumDomain >( coord[ T_UserDomain::count - 1 ] ) );
    }
};

template< typename T_DatumCoord >
struct DatumCoordToTreeCoord
{
    using FrontType = Tuple< TreeCoordElementConst< T_DatumCoord::PopFront::front > >;
    using type = decltype(
        tupleCat(
            FrontType(),
            typename DatumCoordToTreeCoord<
                typename T_DatumCoord::PopFront
            >::type()
        )
    );
};

template< std::size_t T_lastCoord >
struct DatumCoordToTreeCoord< DatumCoord< T_lastCoord > >
{
    using type = Tuple< TreeCoordElementConst< > >;
};

} // namespace internal

template<
    typename T_DatumCoord,
    typename T_UserDomain
>
LLAMA_FN_HOST_ACC_INLINE
auto
getBasicTreeCoordFromDomains( T_UserDomain const & coord )
-> decltype(
    tupleCat(
        internal::UserDomainToTreeCoord<
            T_UserDomain,
            T_DatumCoord::front
        >()( coord ),
        typename internal::DatumCoordToTreeCoord< T_DatumCoord >::type()
    )
)
{
    return tupleCat(
        internal::UserDomainToTreeCoord<
            T_UserDomain,
            T_DatumCoord::front
        >()( coord ),
        typename internal::DatumCoordToTreeCoord< T_DatumCoord >::type()
    );
}

} // namespace tree

} // namespace mapping

} // namespace llama


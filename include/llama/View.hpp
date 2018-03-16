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

#include <boost/preprocessor/cat.hpp>
#include <type_traits>

#include "preprocessor/macros.hpp"
#include "GetType.hpp"
#include "Array.hpp"
#include "ForEach.hpp"

namespace llama
{

template<
    typename T_Mapping,
    typename T_BlobType
>
struct View;

#define __LLAMA_DEFINE_FOREACH_FUNCTOR( OP, FUNCTOR )                          \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_RightDatum,                                                 \
        typename T_Source                                                      \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, Functor)                                     \
    {                                                                          \
        template<                                                              \
            typename T_OuterCoord,                                             \
            typename T_InnerCoord                                              \
        >                                                                      \
        LLAMA_FN_HOST_ACC_INLINE                                               \
        auto                                                                   \
        operator()(                                                            \
            T_OuterCoord,                                                      \
            T_InnerCoord                                                       \
        )                                                                      \
        -> void                                                                \
        {                                                                      \
            using Dst = typename T_OuterCoord::template Cat< T_InnerCoord >;   \
            using Src = typename T_Source::template Cat< T_InnerCoord >;       \
            left( Dst() ) OP right( Src() );                                   \
        }                                                                      \
        T_LeftDatum left;                                                      \
        T_RightDatum right;                                                    \
    };                                                                         \
                                                                               \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_RightType                                                   \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, TypeFunctor)                                 \
    {                                                                          \
        template<                                                              \
            typename T_OuterCoord,                                             \
            typename T_InnerCoord                                              \
        >                                                                      \
        LLAMA_FN_HOST_ACC_INLINE                                               \
        auto                                                                   \
        operator()(                                                            \
            T_OuterCoord,                                                      \
            T_InnerCoord                                                       \
        )                                                                      \
        -> void                                                                \
        {                                                                      \
            using Dst = typename T_OuterCoord::template Cat< T_InnerCoord >;   \
            left( Dst() ) OP static_cast< typename std::remove_reference<      \
                decltype( left( Dst() ) ) >::type >( right );                  \
        }                                                                      \
        T_LeftDatum left;                                                      \
        T_RightType right;                                                     \
    };

__LLAMA_DEFINE_FOREACH_FUNCTOR( =  , Assigment )
__LLAMA_DEFINE_FOREACH_FUNCTOR( += , Addition )
__LLAMA_DEFINE_FOREACH_FUNCTOR( -= , Subtraction )
__LLAMA_DEFINE_FOREACH_FUNCTOR( *= , Multiplication )
__LLAMA_DEFINE_FOREACH_FUNCTOR( /= , Division )
__LLAMA_DEFINE_FOREACH_FUNCTOR( %= , Modulo )

#define __LLAMA_VIRTUALDATUM_VIRTUALDATUM_OPERATOR( OP, FUNCTOR, REF )         \
    template< typename T_OtherView >                                           \
    LLAMA_FN_HOST_ACC_INLINE                                                   \
    auto                                                                       \
    operator OP( VirtualDatum< T_OtherView >REF other )                        \
    -> decltype(*this)&                                                        \
    {                                                                          \
        BOOST_PP_CAT( FUNCTOR, Functor)<                                       \
            decltype(*this),                                                   \
            VirtualDatum< T_OtherView >,                                       \
            DatumCoord< >                                                      \
        > functor{                                                             \
            *this,                                                             \
            other                                                              \
        };                                                                     \
        forEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >( functor );                                                          \
        return *this;                                                          \
    }

#define __LLAMA_VIRTUALDATUM_VIEW_OPERATOR( OP, FUNCTOR, REF )                 \
    template<                                                                  \
        typename T_OtherMapping,                                               \
        typename T_OtherBlobType                                               \
    >                                                                          \
    LLAMA_FN_HOST_ACC_INLINE                                                   \
    auto                                                                       \
    operator OP( View<                                                         \
            T_OtherMapping,                                                    \
            T_OtherBlobType                                                    \
        > REF other                                                            \
    )                                                                          \
    -> decltype(*this)&                                                        \
    {                                                                          \
        auto otherVd =                                                         \
            other( userDomainZero< T_OtherMapping::UserDomain::count >() );    \
        BOOST_PP_CAT( FUNCTOR, Functor)<                                       \
            decltype(*this),                                                   \
            decltype(otherVd),                                                 \
            DatumCoord< >                                                      \
        > functor{                                                             \
            *this,                                                             \
            otherVd                                                            \
        };                                                                     \
        forEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >( functor );                                                          \
        return *this;                                                          \
    }

#define __LLAMA_VIRTUALDATUM_TYPE_OPERATOR( OP, FUNCTOR, REF )                 \
    template< typename T_OtherType >                                           \
    LLAMA_FN_HOST_ACC_INLINE                                                   \
    auto                                                                       \
    operator OP( T_OtherType REF other )                                       \
    -> decltype(*this)&                                                        \
    {                                                                          \
        BOOST_PP_CAT( FUNCTOR, TypeFunctor)<                                   \
            decltype(*this),                                                   \
            T_OtherType                                                        \
        > functor{                                                             \
            *this,                                                             \
            other                                                              \
        };                                                                     \
        forEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >( functor );                                                          \
        return *this;                                                          \
    }

#define __LLAMA_VIRTUALDATUM_OPERATOR( OP, FUNCTOR )                           \
    __LLAMA_VIRTUALDATUM_VIRTUALDATUM_OPERATOR( OP, FUNCTOR, & )               \
    __LLAMA_VIRTUALDATUM_VIRTUALDATUM_OPERATOR( OP, FUNCTOR, && )              \
    __LLAMA_VIRTUALDATUM_VIEW_OPERATOR( OP, FUNCTOR, & )                       \
    __LLAMA_VIRTUALDATUM_VIEW_OPERATOR( OP, FUNCTOR, && )                      \
    __LLAMA_VIRTUALDATUM_TYPE_OPERATOR( OP, FUNCTOR, & )                       \
    __LLAMA_VIRTUALDATUM_TYPE_OPERATOR( OP, FUNCTOR, && )

template< typename T_View >
struct VirtualDatum
{
    using ViewType = T_View;
    using Mapping = typename ViewType::Mapping;
    using BlobType = typename ViewType::BlobType;

    template< std::size_t... T_coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    access( DatumCoord< T_coord... > && = DatumCoord< T_coord... >() )
    -> typename GetType<
        typename Mapping::DatumDomain::TypeTree,
        T_coord...
    >::type &
    {
        return view.template accessor< T_coord... >( userDomainPos );
    }

    template< std::size_t... T_coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( DatumCoord< T_coord... > && dc= DatumCoord< T_coord... >() )
    -> typename GetType<
        typename Mapping::DatumDomain::TypeTree,
        T_coord...
    >::type &
    {
        return access< T_coord... >(
            std::forward< DatumCoord< T_coord... > >( dc )
        );
    }

    __LLAMA_VIRTUALDATUM_OPERATOR( = , Assigment )
    __LLAMA_VIRTUALDATUM_OPERATOR( += , Addition )
    __LLAMA_VIRTUALDATUM_OPERATOR( -= , Addition )
    __LLAMA_VIRTUALDATUM_OPERATOR( *= , Multiplication )
    __LLAMA_VIRTUALDATUM_OPERATOR( /= , Division )
    __LLAMA_VIRTUALDATUM_OPERATOR( %= , Modulo )

    typename Mapping::UserDomain const userDomainPos;
    ViewType& view;
};

template<
    typename T_Mapping,
    typename T_BlobType
>
struct View
{
    using BlobType = T_BlobType;
    using Mapping = T_Mapping;
    using VirtualDatumType = VirtualDatum<
        View <
            Mapping,
            BlobType
        >
    >;

    View() = default;
    View( View const & ) = default;
    View( View && ) = default;
    ~View( ) = default;

    LLAMA_NO_HOST_ACC_WARNING
    LLAMA_FN_HOST_ACC_INLINE
    View(
        Mapping mapping,
        Array<
            BlobType,
            Mapping::blobCount
        > blob
    ) :
        mapping( mapping ),
        blob( blob )
    { }

    LLAMA_NO_HOST_ACC_WARNING
    template< std::size_t... T_datumDomain >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    accessor( typename Mapping::UserDomain const userDomain )
    -> typename GetType<
        typename Mapping::DatumDomain::TypeTree,
        T_datumDomain...
    >::type &
    {
        auto const nr =
            mapping.template getBlobNr< T_datumDomain... >( userDomain );
        auto const byte =
            mapping.template getBlobByte< T_datumDomain... >( userDomain );
        return *( reinterpret_cast< typename GetType<
                typename Mapping::DatumDomain::TypeTree,
                T_datumDomain...
            >::type* > (
                &blob[ nr ][ byte ]
            )
        );
    }

    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( typename Mapping::UserDomain const userDomain )
    -> VirtualDatumType
    {
        return VirtualDatumType{
                userDomain,
                *this
            };
    }

    template< typename... T_Coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Coord... coord )
    -> VirtualDatumType
    {
        return VirtualDatumType{
                typename Mapping::UserDomain{ coord... },
                *this
            };
    }

    template< typename... T_Coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Coord... coord ) const
    -> const VirtualDatumType
    {
        return VirtualDatumType{
                typename Mapping::UserDomain{ coord... },
                *this
            };
    }

    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( std::size_t coord )
    -> VirtualDatumType
    {
        return VirtualDatumType{
                typename Mapping::UserDomain{ coord },
                *this
            };
    }

    template< std::size_t... T_coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( DatumCoord< T_coord... > && dc= DatumCoord< T_coord... >() )
    -> typename GetType<
        typename Mapping::DatumDomain::TypeTree,
        T_coord...
    >::type &
    {
        return accessor< T_coord... >(
            userDomainZero< Mapping::UserDomain::count >()
        );
    }


    Array<
        BlobType,
        Mapping::blobCount
    > blob;
    const Mapping mapping;
};

} // namespace llama

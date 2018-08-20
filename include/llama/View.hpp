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
#include "CompareUID.hpp"

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
        typename T_LeftBase,                                                   \
        typename T_LeftLocal,                                                  \
        typename T_RightDatum,                                                 \
        typename T_RightBase,                                                  \
        typename T_RightLocal,                                                 \
        typename SFINAE = void                                                 \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, IfSameUIDFunctor)                            \
    {                                                                          \
        LLAMA_FN_HOST_ACC_INLINE                                               \
        auto                                                                   \
        operator()() const                                                     \
        -> void                                                                \
        {}                                                                     \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
    };                                                                         \
                                                                               \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_LeftBase,                                                   \
        typename T_LeftLocal,                                                  \
        typename T_RightDatum,                                                 \
        typename T_RightBase,                                                  \
        typename T_RightLocal                                                  \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, IfSameUIDFunctor)                            \
    <                                                                          \
        T_LeftDatum,                                                           \
        T_LeftBase,                                                            \
        T_LeftLocal,                                                           \
        T_RightDatum,                                                          \
        T_RightBase,                                                           \
        T_RightLocal,                                                          \
        typename std::enable_if<                                               \
            CompareUID<                                                        \
                typename T_LeftDatum::Mapping::DatumDomain,                    \
                T_LeftBase,                                                    \
                T_LeftLocal,                                                   \
                typename T_RightDatum::Mapping::DatumDomain,                   \
                T_RightBase,                                                   \
                T_RightLocal                                                   \
            >::value                                                           \
        >::type                                                                \
    >                                                                          \
    {                                                                          \
        LLAMA_FN_HOST_ACC_INLINE                                               \
        auto                                                                   \
        operator()()                                                           \
        -> void                                                                \
        {                                                                      \
            using Dst = typename T_LeftBase::template Cat< T_LeftLocal >;      \
            using Src = typename T_RightBase::template Cat< T_RightLocal >;    \
            left( Dst() ) OP right( Src() );                                   \
        }                                                                      \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
    };                                                                         \
                                                                               \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_LeftBase,                                                   \
        typename T_LeftLocal,                                                  \
        typename T_RightDatum                                                  \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, InnerFunctor)                                \
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
            BOOST_PP_CAT( FUNCTOR, IfSameUIDFunctor) <                         \
                typename std::remove_reference<T_LeftDatum>::type,             \
                T_LeftBase,                                                    \
                T_LeftLocal,                                                   \
                typename std::remove_reference<T_RightDatum>::type,            \
                T_OuterCoord,                                                  \
                T_InnerCoord                                                   \
            > functor {                                                        \
                left,                                                          \
                right                                                          \
            };                                                                 \
            functor();                                                         \
        }                                                                      \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
    };                                                                         \
                                                                               \
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
            BOOST_PP_CAT( FUNCTOR, InnerFunctor)<                              \
                T_LeftDatum,                                                   \
                T_OuterCoord,                                                  \
                T_InnerCoord,                                                  \
                T_RightDatum                                                   \
            > functor{                                                         \
                left,                                                          \
                right                                                          \
            };                                                                 \
            ForEach<                                                           \
                typename T_RightDatum::Mapping::DatumDomain,                   \
                T_Source                                                       \
            >::apply( functor );                                               \
        }                                                                      \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
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
        T_LeftDatum & left;                                                    \
        T_RightType & right;                                                   \
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
        ForEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >::apply( functor );                                                   \
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
        ForEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >::apply( functor );                                                   \
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
        ForEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >::apply( functor );                                                   \
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

    typename Mapping::UserDomain const userDomainPos;
    ViewType& view;

    template< typename... T_UIDs >
    struct AccessImpl
    {
        template< typename T_UserDomain >
        static
        LLAMA_FN_HOST_ACC_INLINE
        auto
        apply(
            T_View&& view,
            T_UserDomain const & userDomainPos
        )
        -> decltype( view.template accessor< T_UIDs... >( userDomainPos ) )&
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return view.template accessor< T_UIDs... >( userDomainPos );
        }
    };

    template< std::size_t... T_coord >
    struct AccessImpl< DatumCoord< T_coord... > >
    {
        template< typename T_UserDomain >
        static
        LLAMA_FN_HOST_ACC_INLINE
        auto
        apply(
            T_View&& view,
            T_UserDomain const & userDomainPos
        )
        -> decltype( view.template accessor< T_coord... >( userDomainPos ) )&
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return view.template accessor< T_coord... >( userDomainPos );
        }
    };

    template< typename... T_DatumCoordOrUIDs  >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    access( T_DatumCoordOrUIDs&&... )
    -> decltype( AccessImpl< T_DatumCoordOrUIDs... >::apply(
            std::forward<T_View>(view),
            userDomainPos
        ) ) &
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return AccessImpl< T_DatumCoordOrUIDs... >::apply(
            std::forward<T_View>(view),
            userDomainPos
        );
    }

    template< typename... T_DatumCoordOrUIDs  >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    access( )
    -> decltype( AccessImpl< T_DatumCoordOrUIDs... >::apply(
            std::forward<T_View>(view),
            userDomainPos
        ) ) &
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return AccessImpl< T_DatumCoordOrUIDs... >::apply(
            std::forward<T_View>(view),
            userDomainPos
        );
    }

    template< typename... T_DatumCoordOrUIDs  >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_DatumCoordOrUIDs&&... )
#if !BOOST_COMP_INTEL && !BOOST_COMP_NVCC
    -> decltype( access< T_DatumCoordOrUIDs... >() ) &
#else //Intel compiler bug work around
    -> decltype( AccessImpl< T_DatumCoordOrUIDs... >::apply(
        std::forward<T_View>(view),
        userDomainPos
    ) ) &
#endif
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return access< T_DatumCoordOrUIDs... >();
    }

    __LLAMA_VIRTUALDATUM_OPERATOR( = , Assigment )
    __LLAMA_VIRTUALDATUM_OPERATOR( += , Addition )
    __LLAMA_VIRTUALDATUM_OPERATOR( -= , Addition )
    __LLAMA_VIRTUALDATUM_OPERATOR( *= , Multiplication )
    __LLAMA_VIRTUALDATUM_OPERATOR( /= , Division )
    __LLAMA_VIRTUALDATUM_OPERATOR( %= , Modulo )
};

namespace internal
{
    template< typename T_DatumCoord >
    struct MappingDatumCoordCaller;

    template< std::size_t... T_coords >
    struct MappingDatumCoordCaller< DatumCoord< T_coords... > >
    {
        template<
            typename T_Mapping,
            typename T_UserDomain
        >
        LLAMA_NO_HOST_ACC_WARNING
        static auto
        LLAMA_FN_HOST_ACC_INLINE
        getBlobNr(
            T_Mapping&& mapping,
            T_UserDomain&& userDomain
        )
        -> decltype( mapping.template getBlobNr< T_coords... >( userDomain ) )
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return mapping.template getBlobNr< T_coords... >( userDomain );
        }

        template<
            typename T_Mapping,
            typename T_UserDomain
        >
        LLAMA_NO_HOST_ACC_WARNING
        static auto
        LLAMA_FN_HOST_ACC_INLINE
        getBlobByte(
            T_Mapping&& mapping,
            T_UserDomain&& userDomain
        )
        -> decltype( mapping.template getBlobNr< T_coords... >( userDomain ) )
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return mapping.template getBlobByte< T_coords... >( userDomain );
        }
    };
}; //namespace internal

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
    template< std::size_t... T_coords >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    accessor( typename Mapping::UserDomain const userDomain )
    -> GetType<
        typename Mapping::DatumDomain,
        T_coords...
    > &
    {

        auto const nr =
            mapping.template getBlobNr< T_coords... >( userDomain );
        auto const byte =
            mapping.template getBlobByte< T_coords... >( userDomain );
        return *( reinterpret_cast< GetType<
                typename Mapping::DatumDomain,
                T_coords...
            >* > (
                &blob[ nr ][ byte ]
            )
        );
    }

    LLAMA_NO_HOST_ACC_WARNING
    template< typename... T_UIDs >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    accessor( typename Mapping::UserDomain const userDomain )
    -> GetTypeFromDatumCoord<
        typename Mapping::DatumDomain,
        GetCoordFromUID<
            typename Mapping::DatumDomain,
            T_UIDs...
        >
    > &
    {
        using DatumCoord = GetCoordFromUID<
            typename Mapping::DatumDomain,
            T_UIDs...
        >;
        auto const nr =
            internal::MappingDatumCoordCaller< DatumCoord >::getBlobNr(
                mapping,
                userDomain
            );
        auto const byte =
            internal::MappingDatumCoordCaller< DatumCoord >::getBlobByte(
                mapping,
                userDomain
            );
        return *( reinterpret_cast< GetTypeFromDatumCoord<
                typename Mapping::DatumDomain,
                DatumCoord
            >* > (
                &blob[ nr ][ byte ]
            )
        );
    }

    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( typename Mapping::UserDomain const userDomain )
    -> VirtualDatumType
    {
        LLAMA_FORCE_INLINE_RECURSIVE
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
        LLAMA_FORCE_INLINE_RECURSIVE
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
        LLAMA_FORCE_INLINE_RECURSIVE
        return VirtualDatumType{
                typename Mapping::UserDomain{ coord... },
                *this
            };
    }

    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( std::size_t coord = 0 )
    -> VirtualDatumType
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return VirtualDatumType{
                typename Mapping::UserDomain{ coord },
                *this
            };
    }

    template< std::size_t... T_coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( DatumCoord< T_coord... > && dc= DatumCoord< T_coord... >() )
    -> GetType<
        typename Mapping::DatumDomain,
        T_coord...
    > &
    {
        LLAMA_FORCE_INLINE_RECURSIVE
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

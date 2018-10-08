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

/** Macro that defines two functors for \ref llama::ForEach which apply an operation on
 *  a given virtual datum and either another virtual datum or some other type.
 *  In the first case the operation is applied if the unique id of the two
 *  elements in the datum domain is the same, in the second case the operation
 *  is applied to every combination of elements of the virtual datum and the
 *  second type.
 * \param OP operation, e.g. +=
 * \param FUNCTOR operation naming used for functor name definition, e.g. if
 *        FUNCTOR is "Addition", the functors will be named AdditionFunctor
 *        and AdditionTypeFunctor.
 * */
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

/** Macro that defines an operator overloading inside of \ref llama::VirtualDatum for
 *  itself and a second virtual datum.
 * \param OP operator, e.g. operator +=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "Addition", the AdditionFunctor
 *        will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * */
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

/** Macro that defines an operator overloading inside of \ref llama::VirtualDatum for
 *  itself and a view. Internally the virtual datum at the first postion (all
 *  zeros) will be taken. This is useful for one-element views (e.g. temporary
 *  views).
 * \param OP operator, e.g. operator +=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "Addition", the AdditionFunctor
 *        will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * */
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

/** Macro that defines an operator overloading inside of \ref llama::VirtualDatum for
 *  itself and some other type.
 * \param OP operator, e.g. operator +=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "Addition", the
 *        AdditionTypeFunctor will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * */
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

/** Macro that defines two functors for \ref llama::ForEach which apply a boolean
 *  operation on a given virtual datum and either another virtual datum or some
 *  other type. In the first case the operation is applied if the unique id of
 *  the two elements in the datum domain is the same, in the second case the
 *  operation is applied to every combination of elements of the virtual datum
 *  and the second type. The result is the logical AND combination of all
 *  results. So e.g., if some elements are bigger and some are smaller than in
 *  the other virtual datum or in the type, for both boolean operations ">" and
 *  "<" the functor will return false. For "!=" the operator would return true.
 * \param OP operation, e.g. >=
 * \param FUNCTOR operation naming used for functor name definition, e.g. if
 *        FUNCTOR is "BiggerSameThan", the functors will be named
 *        BiggerSameThanBoolFunctor and BiggerSameThanBoolTypeFunctor.
 * \return a bool inside the member variable "result" of the functor
 * */
#define __LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR( OP, FUNCTOR )                     \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_LeftBase,                                                   \
        typename T_LeftLocal,                                                  \
        typename T_RightDatum,                                                 \
        typename T_RightBase,                                                  \
        typename T_RightLocal,                                                 \
        typename SFINAE = void                                                 \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, BoolIfSameUIDFunctor)                        \
    {                                                                          \
        LLAMA_FN_HOST_ACC_INLINE                                               \
        auto                                                                   \
        operator()()                                                           \
        -> void                                                                \
        {}                                                                     \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
        bool result;                                                           \
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
    struct BOOST_PP_CAT( FUNCTOR, BoolIfSameUIDFunctor)                        \
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
            result = left( Dst() ) OP right( Src() );                          \
        }                                                                      \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
        bool result;                                                           \
    };                                                                         \
                                                                               \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_LeftBase,                                                   \
        typename T_LeftLocal,                                                  \
        typename T_RightDatum                                                  \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, BoolInnerFunctor)                            \
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
            BOOST_PP_CAT( FUNCTOR, BoolIfSameUIDFunctor) <                     \
                typename std::remove_reference<T_LeftDatum>::type,             \
                T_LeftBase,                                                    \
                T_LeftLocal,                                                   \
                typename std::remove_reference<T_RightDatum>::type,            \
                T_OuterCoord,                                                  \
                T_InnerCoord                                                   \
            > functor {                                                        \
                left,                                                          \
                right,                                                         \
                true                                                           \
            };                                                                 \
            functor();                                                         \
            result &= functor.result;                                          \
        }                                                                      \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
        bool result;                                                           \
    };                                                                         \
                                                                               \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_RightDatum,                                                 \
        typename T_Source                                                      \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, BoolFunctor)                                 \
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
            BOOST_PP_CAT( FUNCTOR, BoolInnerFunctor)<                          \
                T_LeftDatum,                                                   \
                T_OuterCoord,                                                  \
                T_InnerCoord,                                                  \
                T_RightDatum                                                   \
            > functor{                                                         \
                left,                                                          \
                right,                                                         \
                true                                                           \
            };                                                                 \
            ForEach<                                                           \
                typename T_RightDatum::Mapping::DatumDomain,                   \
                T_Source                                                       \
            >::apply( functor );                                               \
            result &= functor.result;                                          \
        }                                                                      \
        T_LeftDatum & left;                                                    \
        T_RightDatum & right;                                                  \
        bool result;                                                           \
    };                                                                         \
                                                                               \
    template<                                                                  \
        typename T_LeftDatum,                                                  \
        typename T_RightType                                                   \
    >                                                                          \
    struct BOOST_PP_CAT( FUNCTOR, BoolTypeFunctor)                             \
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
            result &=                                                          \
                left( Dst() ) OP static_cast< typename std::remove_reference<  \
                decltype( left( Dst() ) ) >::type >( right );                  \
        }                                                                      \
        T_LeftDatum & left;                                                    \
        T_RightType & right;                                                   \
        bool result;                                                           \
    };

__LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR( == , SameAs )
__LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR( != , Not )
__LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR( <  , SmallerThan )
__LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR( <= , SmallerSameThan )
__LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR( >  , BiggerThan )
__LLAMA_DEFINE_FOREACH_BOOL_FUNCTOR( >= , BiggerSameThan )

/** Macro that defines a boolean operator overloading inside of
 *  \ref llama::VirtualDatum for itself and a second virtual datum.
 * \param OP operator, e.g. operator >=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "BiggerSameThan", the
 *        BiggerSameThanBoolFunctor will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * \return result of the boolean operation for every combination with the same
 *  UID
 * */
#define __LLAMA_VIRTUALDATUM_VIRTUALDATUM_BOOL_OPERATOR( OP, FUNCTOR, REF )    \
    template< typename T_OtherView >                                           \
    LLAMA_FN_HOST_ACC_INLINE                                                   \
    auto                                                                       \
    operator OP( VirtualDatum< T_OtherView >REF other )                        \
    -> bool                                                                    \
    {                                                                          \
        BOOST_PP_CAT( FUNCTOR, BoolFunctor)<                                   \
            decltype(*this),                                                   \
            VirtualDatum< T_OtherView >,                                       \
            DatumCoord< >                                                      \
        > functor{                                                             \
            *this,                                                             \
            other,                                                             \
            true                                                               \
        };                                                                     \
        ForEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >::apply( functor );                                                   \
        return functor.result;                                                 \
    }

/** Macro that defines a boolean operator overloading inside of
 *  \ref llama::VirtualDatum for itself and a view. Internally the virtual datum at the
 *  first postion (all zeros) will be taken. This is useful for one-element
 *  views (e.g. temporary views).
 * \param OP operator, e.g. operator >=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "BiggerSameThan", the
 *        BiggerSameThanBoolFunctor will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * \return result of the boolean operation for every combination with the same
 *  UID
 * */
#define __LLAMA_VIRTUALDATUM_VIEW_BOOL_OPERATOR( OP, FUNCTOR, REF )            \
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
    -> bool                                                                    \
    {                                                                          \
        auto otherVd =                                                         \
            other( userDomainZero< T_OtherMapping::UserDomain::count >() );    \
        BOOST_PP_CAT( FUNCTOR, BoolFunctor)<                                   \
            decltype(*this),                                                   \
            decltype(otherVd),                                                 \
            DatumCoord< >                                                      \
        > functor{                                                             \
            *this,                                                             \
            otherVd,                                                           \
            true                                                               \
        };                                                                     \
        ForEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >::apply( functor );                                                   \
        return functor.result;                                                 \
    }

/** Macro that defines a boolean operator overloading inside of
 *  \ref llama::VirtualDatum for itself and some other type.
 * \param OP operator, e.g. operator >=
 * \param FUNCTOR used for calling the internal needed functor to operate on
 *        the virtual datums, e.g. if FUNCTOR is "BiggerSameThan", the
 *        BiggerSameThanBoolTypeFunctor will be used internally.
 * \param REF may be & or && to determine whether it is an overloading for
 *        lvalue or rvalue references
 * \return result of the boolean operation for every combination
 * */
#define __LLAMA_VIRTUALDATUM_TYPE_BOOL_OPERATOR( OP, FUNCTOR, REF )            \
    template< typename T_OtherType >                                           \
    LLAMA_FN_HOST_ACC_INLINE                                                   \
    auto                                                                       \
    operator OP( T_OtherType REF other )                                       \
    -> bool                                                                    \
    {                                                                          \
        BOOST_PP_CAT( FUNCTOR, BoolTypeFunctor)<                               \
            decltype(*this),                                                   \
            T_OtherType                                                        \
        > functor{                                                             \
            *this,                                                             \
            other,                                                             \
            true                                                               \
        };                                                                     \
        ForEach<                                                               \
            typename Mapping::DatumDomain,                                     \
            DatumCoord< >                                                      \
        >::apply( functor );                                                   \
        return functor.result;                                                 \
    }

#define __LLAMA_VIRTUALDATUM_BOOL_OPERATOR( OP, FUNCTOR )                      \
    __LLAMA_VIRTUALDATUM_VIRTUALDATUM_BOOL_OPERATOR( OP, FUNCTOR, & )          \
    __LLAMA_VIRTUALDATUM_VIRTUALDATUM_BOOL_OPERATOR( OP, FUNCTOR, && )         \
    __LLAMA_VIRTUALDATUM_VIEW_BOOL_OPERATOR( OP, FUNCTOR, & )                  \
    __LLAMA_VIRTUALDATUM_VIEW_BOOL_OPERATOR( OP, FUNCTOR, && )                 \
    __LLAMA_VIRTUALDATUM_TYPE_BOOL_OPERATOR( OP, FUNCTOR, & )                  \
    __LLAMA_VIRTUALDATUM_TYPE_BOOL_OPERATOR( OP, FUNCTOR, && )

/** Virtual data type returned by \ref View after resolving user domain address,
 *  being "virtual" in that sense that the data of the virtual datum are not
 *  part of the struct itself but a helper object to address them in the compile
 *  time datum domain
 * \tparam T_View parent view of the virtual datum
 */
template< typename T_View >
struct VirtualDatum
{
    /// parent view of the virtual datum
    using ViewType = T_View;
    /// mapping of the underlying view
    using Mapping = typename ViewType::Mapping;
    /// blobtype of the underlying view
    using BlobType = typename ViewType::BlobType;

    /// resolved position in the user domain
    typename Mapping::UserDomain const userDomainPos;
    /// reference to parent view
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

    /** Explicit access function for a coordinate in the datum domain given as
     *  unique identifier or \ref DatumCoord.
     * \tparam T_DatumCoordOrUIDs... variadic number of types as unique
     *  identifier **or** \ref DatumCoord with tree coordinates as template
     *  parameters inside
     * \return reference to element at resolved user domain and given datum
     *  domain coordinate
     */
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

    /** Explicit access function for a coordinate in the datum domain given as
     *  tree position indexes.
     * \tparam T_coord... variadic number std::size_t numbers as tree
     *  coordinates
     * \return reference to element at resolved user domain and given datum
     *  domain coordinate
     */
    template< std::size_t... T_coord  >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    access( )
    -> decltype( AccessImpl< DatumCoord< T_coord... > >::apply(
            std::forward<T_View>(view),
            userDomainPos
        ) ) &
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return AccessImpl< DatumCoord< T_coord... > >::apply(
            std::forward<T_View>(view),
            userDomainPos
        );
    }

    /** operator overload() for a coordinate in the datum domain given as
     *  unique identifier or \ref DatumCoord.
     * \param datumCoordOrUIDs instantiation of variadic number of unique
     *  identifier types **or** \ref DatumCoord with tree coordinates as
     *  template parameters inside
     * \return reference to element at resolved user domain and given datum
     *  domain coordinate
     */
    template< typename... T_DatumCoordOrUIDs  >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_DatumCoordOrUIDs&&... LLAMA_IGNORE_LITERAL( datumCoordOrUIDs ) )
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

    __LLAMA_VIRTUALDATUM_OPERATOR( =  , Assigment )
    __LLAMA_VIRTUALDATUM_OPERATOR( += , Addition )
    __LLAMA_VIRTUALDATUM_OPERATOR( -= , Subtraction )
    __LLAMA_VIRTUALDATUM_OPERATOR( *= , Multiplication )
    __LLAMA_VIRTUALDATUM_OPERATOR( /= , Division )
    __LLAMA_VIRTUALDATUM_OPERATOR( %= , Modulo )

    __LLAMA_VIRTUALDATUM_BOOL_OPERATOR( == , SameAs )
    __LLAMA_VIRTUALDATUM_BOOL_OPERATOR( != , Not )
    __LLAMA_VIRTUALDATUM_BOOL_OPERATOR( <  , SmallerThan )
    __LLAMA_VIRTUALDATUM_BOOL_OPERATOR( <= , SmallerSameThan )
    __LLAMA_VIRTUALDATUM_BOOL_OPERATOR( >  , BiggerThan )
    __LLAMA_VIRTUALDATUM_BOOL_OPERATOR( >= , BiggerSameThan )
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

/** Central LLAMA class holding memory and giving access to it defined by a
 *  mapping. Should not be instantiated "by hand" but with a \ref Factory.
 * \tparam T_Mapping the mapping of the view
 * \tparam T_BlobType the background data type of the raw data, at the moment
 *  always an 8 bit type like "unsigned char"
 */
template<
    typename T_Mapping,
    typename T_BlobType
>
struct View
{
    /// background data type
    using BlobType = T_BlobType;
    /// used mapping
    using Mapping = T_Mapping;
    /** corresponding \ref llama::VirtualDatum type returned after resolving user
     *  domain
     */
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

    /** Explicit access function taking the datum domain as tree index
     *  coordinate template arguments and the user domain as runtime parameter.
     *  The operator() overloadings should be preferred as they show a more
     *  array of struct like interface using \ref llama::VirtualDatum.
     * \tparam T_coords... tree index coordinate
     * \param userDomain user domain as \ref UserDomain
     * \return reference to element
     */
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

    /** Explicit access function taking the datum domain as UID type list
     *  template arguments and the user domain as runtime parameter.
     *  The operator() overloadings should be preferred as they show a more
     *  array of struct like interface using \ref llama::VirtualDatum.
     * \tparam T_UIDs... UID type list
     * \param userDomain user domain as \ref UserDomain
     * \return reference to element
     */
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

    /** Operator overloading to reverse the order of compile time (datum domain)
     *  and run time (user domain) parameter with a helper object
     *  (\ref llama::VirtualDatum). Should be favoured to access data because of the
     *  more array of struct like interface and the handy intermediate
     *  \ref llama::VirtualDatum object.
     * \param userDomain user domain as \ref UserDomain
     * \return \ref llama::VirtualDatum with bound user domain, which can be used to
     *  access the datum domain
     */
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

    /** Operator overloading to reverse the order of compile time (datum domain)
     *  and run time (user domain) parameter with a helper object
     *  (\ref llama::VirtualDatum). Should be favoured to access data because of the
     *  more array of struct like interface and the handy intermediate
     *  \ref llama::VirtualDatum object.
     * \tparam T_Coord... types of user domain coordinates
     * \param coord user domain as list of numbers
     * \return \ref llama::VirtualDatum with bound user domain, which can be used to
     *  access the datum domain
     */
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

    /// mapping of the view
    const Mapping mapping;
    /// memory of the view
    Array<
        BlobType,
        Mapping::blobCount
    > blob;
};

} // namespace llama

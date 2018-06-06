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
#include "preprocessor/macros.hpp"

namespace llama
{

template< typename... T_Elements >
struct Tuple;

template< >
struct Tuple< >
{};

template<
    typename T_FirstElement,
    typename... T_Elements
>
struct Tuple<
    T_FirstElement,
    T_Elements...
>
{
    using FirstElement = T_FirstElement;
    using RestTuple = Tuple< T_Elements... >;

    Tuple() = default;

    LLAMA_FN_HOST_ACC_INLINE
    Tuple(
        T_FirstElement const first,
        T_Elements const ... rest
    ) :
        first( first ),
        rest( rest... )
    {}

    LLAMA_FN_HOST_ACC_INLINE
    Tuple(
        T_FirstElement const first,
        Tuple< T_Elements... > const rest
    ) :
        first( first ),
        rest( rest )
    {}

    FirstElement first;
    RestTuple rest;
};

template< typename T_FirstElement >
struct Tuple< T_FirstElement >
{
    using FirstElement = T_FirstElement;
    using RestTuple = Tuple< >;

    Tuple() = default;

    LLAMA_FN_HOST_ACC_INLINE
    Tuple(
        T_FirstElement const first,
        Tuple< > const rest = Tuple< >()
    ) : first( first ) {}

    FirstElement first;
};


namespace internal
{

template< typename... T_Elements >
struct MakeTupleImpl;

template<
    typename T_FirstElement,
    typename... T_Elements
>
struct MakeTupleImpl<
    T_FirstElement,
    T_Elements...
>
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_FirstElement const firstElement,
        T_Elements const ... elements
    )
    -> Tuple<
        T_FirstElement,
        T_Elements...
    >
    {
        return Tuple<
            T_FirstElement,
            T_Elements...
        >(
            firstElement,
            MakeTupleImpl< T_Elements... >()( elements... )
        );
    }
};

template< typename T_LastElement >
struct MakeTupleImpl< T_LastElement >
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_LastElement const lastElement )
    -> Tuple< T_LastElement >
    {
        return Tuple< T_LastElement >( lastElement );
    }
};

template< >
struct MakeTupleImpl< >
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( )
    -> Tuple< >
    {
        return Tuple< >( );
    }
};

} // namespace internal

template< typename... T_Elements >
LLAMA_FN_HOST_ACC_INLINE
auto
makeTuple( T_Elements... elements )
-> decltype( internal::MakeTupleImpl< T_Elements...>()( elements... ) )
{
    return internal::MakeTupleImpl< T_Elements...>()( elements... );
}

namespace internal
{

template<
    typename T_Tuple,
    std::size_t T_pos
>
struct GetTupleTypeImpl;

template<
    typename T_First,
    std::size_t T_pos,
    typename... T_Rest
>
struct GetTupleTypeImpl<
    Tuple<
        T_First,
        T_Rest...
    >,
    T_pos
>
{
    using type = typename GetTupleTypeImpl<
        Tuple< T_Rest... >,
        T_pos - 1
    >::type;
};

template<
    typename T_First,
    typename... T_Rest
>
struct GetTupleTypeImpl<
    Tuple<
        T_First,
        T_Rest...
    >,
    0
>
{
    using type = T_First;
};

} // internal

template<
    typename T_Tuple,
    std::size_t T_pos
>
using GetTupleType = typename internal::GetTupleTypeImpl<
    T_Tuple,
    T_pos
>::type;

namespace internal
{

template<
    typename T_Tuple,
    std::size_t T_pos
>
struct GetTupleElementImpl;

template<
    typename T_First,
    std::size_t T_pos,
    typename... T_Rest
>
struct GetTupleElementImpl<
    Tuple<
        T_First,
        T_Rest...
    >,
    T_pos
>
{
    using TupleType = Tuple<
        T_First,
        T_Rest...
    >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( TupleType const & tuple )
    -> GetTupleType<
        TupleType,
        T_pos
    > const &
    {
        return GetTupleElementImpl<
            Tuple< T_Rest... >,
            T_pos - 1
        >()( tuple.rest );
    }
};

template<
    typename T_First,
    typename... T_Rest
>
struct GetTupleElementImpl<
    Tuple<
        T_First,
        T_Rest...
    >,
    0
>
{
    using TupleType = Tuple<
        T_First,
        T_Rest...
    >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( TupleType const & tuple )
    -> T_First const &
    {
        return tuple.first;
    }
};

} // internal

template<
    std::size_t T_pos,
    typename T_Tuple
>
LLAMA_FN_HOST_ACC_INLINE
auto
getTupleElement( T_Tuple const & tuple )
-> GetTupleType<
    T_Tuple,
    T_pos
>
{
    return internal::GetTupleElementImpl<
        T_Tuple,
        T_pos
    >()( tuple );
}

template<
    std::size_t T_pos,
    typename T_Tuple
>
LLAMA_FN_HOST_ACC_INLINE
auto
getTupleElementRef( T_Tuple const & tuple )
-> GetTupleType<
    T_Tuple,
    T_pos
> const &
{
    return internal::GetTupleElementImpl<
        T_Tuple,
        T_pos
    >()( tuple );
}

template< typename T_Tuple >
struct SizeOfTuple;

template< typename... T_Elements >
struct SizeOfTuple< Tuple< T_Elements... > >
{
    static constexpr std::size_t value = sizeof...( T_Elements );
};

namespace internal
{

template<
    typename T_Tuple1,
    typename T_Tuple2
>
struct TupleCatTypeImpl;

template<
    typename T_Tuple2,
    typename T_Tuple1First,
    typename... T_Tuple1Rest
>
struct TupleCatTypeImpl<
    Tuple<
        T_Tuple1First,
        T_Tuple1Rest...
    >,
    T_Tuple2
>
{
    using type = typename TupleCatTypeImpl<
        Tuple< T_Tuple1First >,
        typename TupleCatTypeImpl<
            Tuple< T_Tuple1Rest... >,
            T_Tuple2
        >::type
    >::type;
};

template<
    typename T_Tuple1Elem,
    typename... T_Tuple2Elems
>
struct TupleCatTypeImpl<
    Tuple< T_Tuple1Elem >,
    Tuple< T_Tuple2Elems... >
>
{
    using type = Tuple<
        T_Tuple1Elem,
        T_Tuple2Elems...
    >;
};

template< typename T_Tuple2 >
struct TupleCatTypeImpl<
    Tuple< >,
    T_Tuple2
>
{
    using type = T_Tuple2;
};

template<
    typename T_Tuple1,
    typename T_Tuple2
>
struct TupleCatImpl;

template<
    typename T_Tuple2,
    typename T_Tuple1First,
    typename... T_Tuple1Rest
>
struct TupleCatImpl<
    Tuple<
        T_Tuple1First,
        T_Tuple1Rest...
    >,
    T_Tuple2
>
{
    using Tuple1 = Tuple<
        T_Tuple1First,
        T_Tuple1Rest...
    >;
    using Result = typename TupleCatTypeImpl<
        Tuple1,
        T_Tuple2
    >::type;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        const Tuple1 t1,
        T_Tuple2 const t2
    )
    -> Result
    {
        return Result(
            t1.first,
            TupleCatImpl<
                Tuple< T_Tuple1Rest... >,
                T_Tuple2
            >()(
                t1.rest,
                t2
            )
        );
    }
};

template<
    typename T_Tuple1Elem,
    typename... T_Tuple2Elems
>
struct TupleCatImpl<
    Tuple< T_Tuple1Elem >,
    Tuple< T_Tuple2Elems... >
>
{
    using Result = Tuple<
        T_Tuple1Elem,
        T_Tuple2Elems...
    >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        const Tuple< T_Tuple1Elem > t1,
        const Tuple< T_Tuple2Elems... > t2
    )
    -> Result
    {
        return Result(
            t1.first,
            t2
        );
    }
};

template< typename T_Tuple2 >
struct TupleCatImpl<
    Tuple< >,
    T_Tuple2
>
{
    using Result = T_Tuple2;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        const Tuple< > t1,
        T_Tuple2 const t2
    )
    -> Result
    {
        return t2;
    }
};

} // namespace internal

template<
    typename T_Tuple1,
    typename T_Tuple2
>
using TupleCatType = typename internal::TupleCatTypeImpl<
    T_Tuple1,
    T_Tuple2
>::type;

template<
    typename T_Tuple1,
    typename T_Tuple2
>
LLAMA_FN_HOST_ACC_INLINE
auto
tupleCat(
    T_Tuple1 const t1,
    T_Tuple2 const t2
)
-> decltype(
    internal::TupleCatImpl<
        T_Tuple1,
        T_Tuple2
    >()(
        t1,
        t2
    )
)
{
    return internal::TupleCatImpl<
        T_Tuple1,
        T_Tuple2
    >()(
        t1,
        t2
    );
}

namespace internal
{

template<
    std::size_t T_pos,
    typename T_Tuple,
    typename T_Replacement
>
struct TupleReplaceImpl
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tuple const tuple,
        T_Replacement const replacement
    )
    -> decltype(
        tupleCat(
            makeTuple( tuple.first ),
            TupleReplaceImpl<
                T_pos - 1,
                typename T_Tuple::RestTuple,
                T_Replacement
            >()(
                tuple.rest,
                replacement
            )
        )
    )
    {
        return tupleCat(
            makeTuple( tuple.first ),
            TupleReplaceImpl<
                T_pos - 1,
                typename T_Tuple::RestTuple,
                T_Replacement
            >()(
                tuple.rest,
                replacement
            )
        );
    };
};

template<
    typename T_Tuple,
    typename T_Replacement
>
struct TupleReplaceImpl<
    0,
    T_Tuple,
    T_Replacement
>
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tuple const tuple,
        T_Replacement const replacement
    )
    -> decltype(
        tupleCat(
            makeTuple( replacement ),
            tuple.rest
        )
    )
    {
        return tupleCat(
            makeTuple( replacement ),
            tuple.rest
        );
    };
};

template<
    typename T_OneElement,
    typename T_Replacement
>
struct TupleReplaceImpl<
    0,
    Tuple< T_OneElement >,
    T_Replacement
>
{
    using T_Tuple = Tuple< T_OneElement >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tuple const tuple,
        T_Replacement const replacement
    )
    -> decltype( makeTuple( replacement ) )
    {
        return makeTuple( replacement );
    }
};

} // namespace internal

template<
    std::size_t T_pos,
    typename T_Tuple,
    typename T_Replacement
>
LLAMA_FN_HOST_ACC_INLINE
auto
tupleReplace(
    T_Tuple const tuple,
    T_Replacement const replacement
)
-> decltype(
    internal::TupleReplaceImpl<
        T_pos,
        T_Tuple,
        T_Replacement
    >()(
        tuple,
        replacement
    )
)
{
    return internal::TupleReplaceImpl<
        T_pos,
        T_Tuple,
        T_Replacement
    >()(
        tuple,
        replacement
    );
}

namespace internal
{

template<
    typename T_Tuple,
    typename T_Functor
>
struct TupleForEach
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tuple const tuple,
        T_Functor const functor
    )
    -> decltype(
        tupleCat(
            makeTuple( functor( tuple.first ) ),
            TupleForEach<
                typename T_Tuple::RestTuple,
                T_Functor
            >()(
                tuple.rest,
                functor
            )
        )
    )
    {
        return tupleCat(
            makeTuple( functor( tuple.first ) ),
            TupleForEach<
                typename T_Tuple::RestTuple,
                T_Functor
            >()(
                tuple.rest,
                functor
            )
        );
    }
};

template<
    typename T_LastElement,
    typename T_Functor
>
struct TupleForEach
<
    Tuple< T_LastElement >,
    T_Functor
>
{
    using T_Tuple = Tuple< T_LastElement >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tuple const tuple,
        T_Functor const functor
    )
    -> decltype( makeTuple( functor( tuple.first ) ) )
    {
        return makeTuple( functor( tuple.first ) );
    }
};

template< typename T_Functor >
struct TupleForEach
<
    Tuple< >,
    T_Functor
>
{
    using T_Tuple = Tuple< >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()(
        T_Tuple const tuple,
        T_Functor const functor
    )
    -> T_Tuple
    {
        return tuple;
    }
};

} // namespace internal

template<
    typename T_Tuple,
    typename T_Functor
>
LLAMA_FN_HOST_ACC_INLINE
auto
tupleForEach(
    T_Tuple const tuple,
    T_Functor const functor
)
-> decltype(
    internal::TupleForEach<
        T_Tuple,
        T_Functor
    >()(
        tuple,
        functor
    )
)
{
    return internal::TupleForEach<
        T_Tuple,
        T_Functor
    >()(
        tuple,
        functor
    );
}

namespace internal
{

template< typename T_Tuple >
struct TupleRestImpl
{
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Tuple const tuple ) const
    -> typename T_Tuple::RestTuple
    {
        return tuple.rest;
    }
};

template< typename T_OneElement >
struct TupleRestImpl< Tuple< T_OneElement > >
{
    using T_Tuple = Tuple< T_OneElement >;
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Tuple const tuple ) const
    -> Tuple< >
    {
        return Tuple< >();
    }
};

} // namesapce internal

template< typename T_Tuple >
LLAMA_FN_HOST_ACC_INLINE
auto
tupleRest( T_Tuple const tuple )
-> decltype( internal::TupleRestImpl< T_Tuple >()( tuple ) )
{
    return internal::TupleRestImpl< T_Tuple >()( tuple );
}

template< typename T_Tuple >
struct TupleLength;

template< typename... T_Childs >
struct TupleLength< Tuple< T_Childs... > >
{
    static constexpr std::size_t value = sizeof...( T_Childs );
};




} // namespace llama

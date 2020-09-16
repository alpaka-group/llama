// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "macros.hpp"

namespace llama
{
    /// Tuple class like `std::tuple` but suitable for use with offloading
    /// devices like GPUs.
    template<typename... Elements>
    struct Tuple;

    template<>
    struct Tuple<>
    {};

    template<typename T_FirstElement, typename... Elements>
    struct Tuple<T_FirstElement, Elements...>
    {
        using FirstElement = T_FirstElement;
        using RestTuple = Tuple<Elements...>;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple(FirstElement first, Elements... rest) :
                first(first), rest(rest...)
        {}

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple(FirstElement first, Tuple<Elements...> rest) :
                first(first), rest(rest)
        {}

        FirstElement first; ///< the first element (if existing)
        RestTuple rest; ///< the remaining elements
    };

    template<typename T_FirstElement>
    struct Tuple<T_FirstElement>
    {
        using FirstElement = T_FirstElement;
        using RestTuple = Tuple<>;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple(
            FirstElement const first,
            Tuple<> const rest = Tuple<>()) :
                first(first)
        {}

        FirstElement first;
    };

    template<typename... Elements>
    Tuple(Elements...) -> Tuple<Elements...>;

    template<typename Tuple, std::size_t Pos>
    using TupleElement = boost::mp11::mp_at_c<Tuple, Pos>;

    template<std::size_t Pos, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE auto get(const Tuple<Elements...> & tuple)
    {
        if constexpr(Pos == 0)
            return tuple.first;
        else
            return get<Pos - 1>(tuple.rest);
    }

    template<typename Tuple>
    inline constexpr auto tupleSize = boost::mp11::mp_size<Tuple>::value;

    namespace internal
    {
        template<typename Tuple1, typename Tuple2, size_t... Is1, size_t... Is2>
        static LLAMA_FN_HOST_ACC_INLINE auto tupleCatImpl(
            const Tuple1 & t1,
            const Tuple2 & t2,
            std::index_sequence<Is1...>,
            std::index_sequence<Is2...>)
        {
            return Tuple{get<Is1>(t1)..., get<Is2>(t2)...};
        }
    }

    template<typename Tuple1, typename Tuple2>
    LLAMA_FN_HOST_ACC_INLINE auto tupleCat(const Tuple1 & t1, const Tuple2 & t2)
    {
        return internal::tupleCatImpl(
            t1,
            t2,
            std::make_index_sequence<tupleSize<Tuple1>>{},
            std::make_index_sequence<tupleSize<Tuple2>>{});
    }

    namespace internal
    {
        template<std::size_t Pos, typename Tuple, typename Replacement>
        struct TupleReplaceImpl
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(Tuple const tuple, Replacement const replacement)
            {
                return tupleCat(
                    llama::Tuple{tuple.first},
                    TupleReplaceImpl<
                        Pos - 1,
                        typename Tuple::RestTuple,
                        Replacement>()(tuple.rest, replacement));
            };
        };

        template<typename... Elements, typename Replacement>
        struct TupleReplaceImpl<0, Tuple<Elements...>, Replacement>
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto
            operator()(Tuple<Elements...> tuple, Replacement const replacement)
            {
                return tupleCat(Tuple{replacement}, tuple.rest);
            };
        };

        template<typename OneElement, typename Replacement>
        struct TupleReplaceImpl<0, Tuple<OneElement>, Replacement>
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto operator()(Tuple<OneElement>, Replacement const replacement)
            {
                return Tuple{replacement};
            }
        };
    }

    /// Creates a copy of a tuple with the element at position Pos replaced by
    /// replacement.
    template<std::size_t Pos, typename Tuple, typename Replacement>
    LLAMA_FN_HOST_ACC_INLINE auto
    tupleReplace(Tuple tuple, Replacement replacement)
    {
        return internal::TupleReplaceImpl<Pos, Tuple, Replacement>()(
            tuple, replacement);
    }

    namespace internal
    {
        template<typename Seq>
        struct TupleTransformHelper;

        template<size_t... Is>
        struct TupleTransformHelper<std::index_sequence<Is...>>
        {
            template<typename... Elements, typename Functor>
            static LLAMA_FN_HOST_ACC_INLINE auto
            transform(const Tuple<Elements...> & tuple, const Functor & functor)
            {
                // FIXME(bgruber): nvcc fails to compile
                // Tuple{functor(get<Is>(tuple))...}
                return Tuple<decltype(functor(std::declval<Elements>()))...>{
                    functor(get<Is>(tuple))...};
            }
        };
    }

    /// Applies a functor to every element of a tuple, creating a new tuple with
    /// the result of the element transformations. The functor needs to
    /// implement a template `operator()` to which all tuple elements are
    /// passed.
    template<typename... Elements, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE auto
    tupleTransform(const Tuple<Elements...> & tuple, const Functor & functor)
    {
        return internal::TupleTransformHelper<std::make_index_sequence<
            sizeof...(Elements)>>::transform(tuple, functor);
    }

    template<typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE auto
    tupleWithoutFirst(const Tuple<Elements...> & tuple)
    {
        return tuple.rest;
    }

    template<typename Element>
    LLAMA_FN_HOST_ACC_INLINE auto
    tupleWithoutFirst(const Tuple<Element> & tuple)
    {
        return Tuple<>{};
    }
}

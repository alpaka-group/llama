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

/// Documentation of this file is in Tuple.dox!
namespace llama
{
    template<typename... T_Elements>
    struct Tuple;

    template<>
    struct Tuple<>
    {};

    template<typename T_FirstElement, typename... T_Elements>
    struct Tuple<T_FirstElement, T_Elements...>
    {
        using FirstElement = T_FirstElement;
        using RestTuple = Tuple<T_Elements...>;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple(T_FirstElement first, T_Elements... rest) :
                first(first), rest(rest...)
        {}

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Tuple(T_FirstElement first, Tuple<T_Elements...> rest) :
                first(first), rest(rest)
        {}

        FirstElement first;
        RestTuple rest;
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
            T_FirstElement const first,
            Tuple<> const rest = Tuple<>()) :
                first(first)
        {}

        FirstElement first;
    };

    template<typename... Elements>
    Tuple(Elements...) -> Tuple<Elements...>;

    template<typename Tuple, std::size_t Pos>
    using GetTupleType = boost::mp11::mp_at_c<Tuple, Pos>;

    template<std::size_t Pos, typename Tuple>
    LLAMA_FN_HOST_ACC_INLINE auto getTupleElement(const Tuple & tuple)
        -> GetTupleType<Tuple, Pos>
    {
        if constexpr(Pos == 0)
            return tuple.first;
        else
            return getTupleElement<Pos - 1>(tuple.rest);
    }

    template<std::size_t Pos, typename T_Tuple>
    LLAMA_FN_HOST_ACC_INLINE auto getTupleElementRef(T_Tuple const & tuple)
        -> const GetTupleType<T_Tuple, Pos> &
    {
        if constexpr(Pos == 0)
            return tuple.first;
        else
            return getTupleElementRef<Pos - 1>(tuple.rest);
    }

    template<typename Tuple>
    using SizeOfTuple = boost::mp11::mp_size<Tuple>;

    namespace internal
    {
        template<typename Seq1, typename Seq2>
        struct TupleCatHelper;

        template<size_t... Is1, size_t... Is2>
        struct TupleCatHelper<
            std::index_sequence<Is1...>,
            std::index_sequence<Is2...>>
        {
            template<typename Tuple1, typename Tuple2>
            static LLAMA_FN_HOST_ACC_INLINE auto
            cat(const Tuple1 & t1, const Tuple2 & t2)
            {
                return Tuple{
                    getTupleElement<Is1>(t1)..., getTupleElement<Is2>(t2)...};
            }
        };
    }

    template<typename Tuple1, typename Tuple2>
    LLAMA_FN_HOST_ACC_INLINE auto tupleCat(const Tuple1 & t1, const Tuple2 & t2)
    {
        return internal::TupleCatHelper<
            std::make_index_sequence<SizeOfTuple<Tuple1>::value>,
            std::make_index_sequence<SizeOfTuple<Tuple2>::value>>::cat(t1, t2);
    }

    namespace internal
    {
        template<std::size_t Pos, typename T_Tuple, typename T_Replacement>
        struct TupleReplaceImpl
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto
            operator()(T_Tuple const tuple, T_Replacement const replacement)
            {
                return tupleCat(
                    Tuple{tuple.first},
                    TupleReplaceImpl<
                        Pos - 1,
                        typename T_Tuple::RestTuple,
                        T_Replacement>()(tuple.rest, replacement));
            };
        };

        template<typename T_Tuple, typename T_Replacement>
        struct TupleReplaceImpl<0, T_Tuple, T_Replacement>
        {
            LLAMA_FN_HOST_ACC_INLINE
            auto
            operator()(T_Tuple const tuple, T_Replacement const replacement)
            {
                return tupleCat(Tuple{replacement}, tuple.rest);
            };
        };

        template<typename T_OneElement, typename T_Replacement>
        struct TupleReplaceImpl<0, Tuple<T_OneElement>, T_Replacement>
        {
            using T_Tuple = Tuple<T_OneElement>;
            LLAMA_FN_HOST_ACC_INLINE
            auto
            operator()(T_Tuple const tuple, T_Replacement const replacement)
            {
                return Tuple{replacement};
            }
        };
    }

    template<std::size_t Pos, typename T_Tuple, typename T_Replacement>
    LLAMA_FN_HOST_ACC_INLINE auto
    tupleReplace(T_Tuple const tuple, T_Replacement const replacement)
    {
        return internal::TupleReplaceImpl<Pos, T_Tuple, T_Replacement>()(
            tuple, replacement);
    }

    namespace internal
    {
        template<typename Seq>
        struct TupleTransformHelper;

        template<size_t... Is>
        struct TupleTransformHelper<std::index_sequence<Is...>>
        {
            template<typename Tuple, typename Functor>
            static LLAMA_FN_HOST_ACC_INLINE auto
            transform(const Tuple & tuple, const Functor & functor)
            {
                return llama::Tuple{functor(getTupleElement<Is>(tuple))...};
            }
        };
    }

    template<typename... Elements, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE auto
    tupleTransform(const Tuple<Elements...> & tuple, const Functor & functor)
    {
        return internal::TupleTransformHelper<std::make_index_sequence<
            sizeof...(Elements)>>::transform(tuple, functor);
    }

    template<typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE auto tupleRest(const Tuple<Elements...> & tuple)
    {
        return tuple.rest;
    }

    template<typename Element>
    LLAMA_FN_HOST_ACC_INLINE auto tupleRest(const Tuple<Element> & tuple)
    {
        return Tuple<>{};
    }
}

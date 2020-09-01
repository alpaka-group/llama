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

#include "macros.hpp"

namespace llama
{
    /** Tuple class like `std::tuple` but suitable for use with offloading
     * devices like GPUs and extended with some (for LLAMA) useful methods.
     * \tparam Elements... tuple elements, may be empty
     */
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
        RestTuple rest; ///< the rest tuple (may be empty or not existing at all
                        ///< for an empty tuple)
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

    template<std::size_t Pos, typename Tuple>
    LLAMA_FN_HOST_ACC_INLINE auto getTupleElementRef(const Tuple & tuple)
        -> const GetTupleType<Tuple, Pos> &
    {
        if constexpr(Pos == 0)
            return tuple.first;
        else
            return getTupleElementRef<Pos - 1>(tuple.rest);
    }

    template<typename Tuple>
    inline constexpr auto SizeOfTuple = boost::mp11::mp_size<Tuple>::value;

    namespace internal
    {
        template<typename Tuple1, typename Tuple2, size_t... Is1, size_t... Is2>
        static LLAMA_FN_HOST_ACC_INLINE auto tupleCatImpl(
            const Tuple1 & t1,
            const Tuple2 & t2,
            std::index_sequence<Is1...>,
            std::index_sequence<Is2...>)
        {
            return Tuple{
                getTupleElement<Is1>(t1)..., getTupleElement<Is2>(t2)...};
        }
    }

    /** Concatenates two tuples to a new tuple containing the elements of both.
     * \tparam Tuple1 type of the first tuple, probably indirectly given as
     *  template argument deduction
     * \tparam Tuple2 type of the second tuple, probably indirectly given as
     *  template argument deduction
     * \param t1 first tuple
     * \param t2 second tuple
     * \return new tuple with element of both input tuples
     */
    template<typename Tuple1, typename Tuple2>
    LLAMA_FN_HOST_ACC_INLINE auto tupleCat(const Tuple1 & t1, const Tuple2 & t2)
    {
        return internal::tupleCatImpl(
            t1,
            t2,
            std::make_index_sequence<SizeOfTuple<Tuple1>>{},
            std::make_index_sequence<SizeOfTuple<Tuple2>>{});
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

    /** Creates a copy of a tuple with one element replaced by another.
     * \tparam Pos position of element to change
     * \tparam Tuple type of input tuple
     * \tparam Replacement new type of element at replaced position
     * \param tuple tuple in which an element shall be replaced
     * \param replacement new element for the returned tuple
     * \return new tuple with same size but maybe new type and replaced element
     */
    template<std::size_t Pos, typename Tuple, typename Replacement>
    LLAMA_FN_HOST_ACC_INLINE auto
    tupleReplace(Tuple const tuple, Replacement const replacement)
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
                // Tuple{functor(getTupleElement<Is>(tuple))...}
                return Tuple<decltype(functor(std::declval<Elements>()))...>{
                    functor(getTupleElement<Is>(tuple))...};
            }
        };
    }

    /** Applies a functor for every element of a tuple and creating a new tuple
     * with the result of the element transformations. The functor needs to
     * implement an `operator()` with one template parameter and one parameter
     * of this type being the current element, which returns some new element of
     * any type. \param tuple the tuple \param functor the functor \returns a
     * new tuple of the same size but maybe different type
     */
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

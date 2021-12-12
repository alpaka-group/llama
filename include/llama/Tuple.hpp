// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Meta.hpp"
#include "macros.hpp"

namespace llama
{
    // for implementation ideas, see e.g.:
    // http://mitchnull.blogspot.com/2012/06/c11-tuple-implementation-details-part-1.html

    template<typename... Elements>
    struct Tuple;

    namespace internal
    {
        template<std::size_t I, typename T, bool = std::is_empty_v<T> && !std::is_final_v<T>>
        struct TupleImplLeaf
        {
            T val;

            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() -> T&
            {
                return val;
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() const -> const T&
            {
                return val;
            }
        };

        template<std::size_t I, typename T>
        struct TupleImplLeaf<I, T, true> : T
        {
            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() -> T&
            {
                return *this;
            }

            LLAMA_FN_HOST_ACC_INLINE constexpr auto value() const -> const T&
            {
                return *this;
            }
        };

        template<typename... IsAndElements>
        struct TupleImpl;

        template<>
        struct TupleImpl<std::index_sequence<>>
        {
        };

        template<std::size_t... Is, typename... Elements>
        struct TupleImpl<std::index_sequence<Is...>, Elements...> : TupleImplLeaf<Is, Elements>...
        {
            using FirstElement = boost::mp11::mp_first<Tuple<Elements...>>;
            using RestTuple = boost::mp11::mp_pop_front<Tuple<Elements...>>;

            constexpr TupleImpl() = default;

            /// Construct a tuple from values of the same types as the tuple stores.
            LLAMA_FN_HOST_ACC_INLINE constexpr explicit TupleImpl(Elements... args)
                : TupleImplLeaf<Is, Elements>{args}...
            {
            }

            /// Construct a tuple from forwarded values of potentially different types as the tuple stores.
            template<
                typename... Ts,
                std::enable_if_t<
                    sizeof...(Elements) == sizeof...(Ts) && (std::is_constructible_v<Elements, Ts> && ...),
                    int> = 0>
            LLAMA_FN_HOST_ACC_INLINE constexpr explicit TupleImpl(Ts&&... args)
                : TupleImplLeaf<Is, Elements>{Elements(std::move(args))}...
            {
            }
        };
    } // namespace internal

    /// Tuple class like `std::tuple` but suitable for use with offloading devices like GPUs. See also \ref
    /// internal::TupleImpl.
    template<typename... Elements>
    struct Tuple : internal::TupleImpl<std::index_sequence_for<Elements...>, Elements...>
    {
    private:
        template<std::size_t I>
        using Base = internal::TupleImplLeaf<I, boost::mp11::mp_at_c<boost::mp11::mp_list<Elements...>, I>>;

    public:
        using internal::TupleImpl<std::index_sequence_for<Elements...>, Elements...>::TupleImpl;

        template<std::size_t Pos>
        LLAMA_FN_HOST_ACC_INLINE friend constexpr auto get(Tuple& tuple) -> auto&
        {
            return tuple.Base<Pos>::value();
        }

        template<std::size_t Pos>
        LLAMA_FN_HOST_ACC_INLINE friend constexpr auto get(const Tuple& tuple) -> const auto&
        {
            return tuple.Base<Pos>::value();
        }
    };

    template<std::size_t Pos, typename... Elements>
    constexpr auto get(Tuple<Elements...>& tuple);

    template<std::size_t Pos, typename... Elements>
    constexpr auto get(const Tuple<Elements...>& tuple);

    template<typename... Elements>
    Tuple(Elements...) -> Tuple<std::remove_cv_t<std::remove_reference_t<Elements>>...>;
} // namespace llama

template<typename... Elements>
struct std::tuple_size<llama::Tuple<Elements...>>
{
    static constexpr auto value = sizeof...(Elements);
};

template<std::size_t I, typename... Elements>
struct std::tuple_element<I, llama::Tuple<Elements...>>
{
    using type = boost::mp11::mp_at_c<llama::Tuple<Elements...>, I>;
};

namespace llama
{
    namespace internal
    {
        template<typename... Elements, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto areEqual(
            const Tuple<Elements...>& a,
            const Tuple<Elements...>& b,
            std::index_sequence<Is...>) -> bool
        {
            return ((get<Is>(a) == get<Is>(b)) && ...);
        }
    } // namespace internal

    template<typename... ElementsA, typename... ElementsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(const Tuple<ElementsA...>& a, const Tuple<ElementsB...>& b)
        -> bool
    {
        using namespace boost::mp11;
        if constexpr(sizeof...(ElementsA) == sizeof...(ElementsB))
            if constexpr(mp_apply<mp_all, mp_transform<std::is_same, mp_list<ElementsA...>, mp_list<ElementsB...>>>::
                             value)
                return internal::areEqual(a, b, std::make_index_sequence<sizeof...(ElementsA)>{});
        return false;
    }

    template<typename... ElementsA, typename... ElementsB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(const Tuple<ElementsA...>& a, const Tuple<ElementsB...>& b)
        -> bool
    {
        return !(a == b);
    }

    namespace internal
    {
        template<typename Tuple1, typename Tuple2, size_t... Is1, size_t... Is2>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleCatImpl(
            const Tuple1& t1,
            const Tuple2& t2,
            std::index_sequence<Is1...>,
            std::index_sequence<Is2...>)
        {
            return Tuple{get<Is1>(t1)..., get<Is2>(t2)...};
        }
    } // namespace internal

    template<typename Tuple1, typename Tuple2>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleCat(const Tuple1& t1, const Tuple2& t2)
    {
        return internal::tupleCatImpl(
            t1,
            t2,
            std::make_index_sequence<std::tuple_size_v<Tuple1>>{},
            std::make_index_sequence<std::tuple_size_v<Tuple2>>{});
    }

    namespace internal
    {
        template<
            std::size_t Pos,
            typename Tuple,
            typename Replacement,
            std::size_t... IsBefore,
            std::size_t... IsAfter>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleReplaceImpl(
            Tuple&& tuple,
            Replacement&& replacement,
            std::index_sequence<IsBefore...>,
            std::index_sequence<IsAfter...>)
        {
            return llama::Tuple{
                get<IsBefore>(std::forward<Tuple>(tuple))...,
                std::forward<Replacement>(replacement),
                get<Pos + 1 + IsAfter>(std::forward<Tuple>(tuple))...};
        }
    } // namespace internal

    /// Creates a copy of a tuple with the element at position Pos replaced by replacement.
    template<std::size_t Pos, typename Tuple, typename Replacement>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleReplace(Tuple&& tuple, Replacement&& replacement)
    {
        return internal::tupleReplaceImpl<Pos>(
            std::forward<Tuple>(tuple),
            std::forward<Replacement>(replacement),
            std::make_index_sequence<Pos>{},
            std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>> - Pos - 1>{});
    }

    namespace internal
    {
        template<size_t... Is, typename... Elements, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleTransformHelper(
            std::index_sequence<Is...>,
            const Tuple<Elements...>& tuple,
            const Functor& functor)
        {
            // FIXME(bgruber): nvcc fails to compile
            // Tuple{functor(get<Is>(tuple))...}
            return Tuple<decltype(functor(std::declval<Elements>()))...>{functor(get<Is>(tuple))...};
        }
    } // namespace internal

    /// Applies a functor to every element of a tuple, creating a new tuple with the result of the element
    /// transformations. The functor needs to implement a template `operator()` to which all tuple elements are passed.
    // TODO(bgruber): replace by mp11 version in Boost 1.74.
    template<typename... Elements, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto tupleTransform(const Tuple<Elements...>& tuple, const Functor& functor)
    {
        return internal::tupleTransformHelper(std::make_index_sequence<sizeof...(Elements)>{}, tuple, functor);
    }

    namespace internal
    {
        template<std::size_t Count = 1, typename... Elements, size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto pop_front_impl(
            const Tuple<Elements...>& tuple,
            std::index_sequence<Is...>)
        {
            return Tuple{get<Count + Is>(tuple)...};
        }
    } // namespace internal

    /// Returns a copy of the tuple without the first element.
    template<std::size_t Count = 1, typename... Elements>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto pop_front(const Tuple<Elements...>& tuple)
    {
        static_assert(sizeof...(Elements) > 0);
        return internal::pop_front_impl<Count>(tuple, std::make_index_sequence<sizeof...(Elements) - Count>{});
    }
} // namespace llama

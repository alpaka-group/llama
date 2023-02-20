// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <boost/mp11.hpp>

namespace llama
{
    // make mp11 directly available in the llama namespace
    using namespace boost::mp11;

    namespace internal
    {
        template<typename FromList, template<auto...> class ToList>
        struct mp_unwrap_values_into_impl;

        template<template<class...> class FromList, typename... Values, template<auto...> class ToList>
        struct mp_unwrap_values_into_impl<FromList<Values...>, ToList>
        {
            using type = ToList<Values::value...>;
        };

        template<typename FromList, template<auto...> class ToList>
        using mp_unwrap_values_into = typename mp_unwrap_values_into_impl<FromList, ToList>::type;

        template<typename E, typename... Args>
        struct ReplacePlaceholdersImpl
        {
            using type = E;
        };
        template<std::size_t I, typename... Args>
        struct ReplacePlaceholdersImpl<mp_arg<I>, Args...>
        {
            using type = mp_at_c<mp_list<Args...>, I>;
        };

        template<template<typename...> typename E, typename... Ts, typename... Args>
        struct ReplacePlaceholdersImpl<E<Ts...>, Args...>
        {
            using type = E<typename ReplacePlaceholdersImpl<Ts, Args...>::type...>;
        };
    } // namespace internal

    template<typename Expression, typename... Args>
    using ReplacePlaceholders = typename internal::ReplacePlaceholdersImpl<Expression, Args...>::type;
} // namespace llama

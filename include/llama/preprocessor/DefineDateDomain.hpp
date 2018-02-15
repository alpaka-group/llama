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

#include <boost/version.hpp>

#if BOOST_VERSION < 106700 && (__CUDACC__ || __IBMCPP__)
    #define BOOST_PP_VARIADICS 1
#endif

#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/tuple/push_back.hpp>

#define LLAMA_ATOMTYPE 0
#define LLAMA_DATESTRUCT 1
#define LLAMA_DATEARRAY 2

#define LLAMA_AT LLAMA_ATOMTYPE
#define LLAMA_DS LLAMA_DATESTRUCT
#define LLAMA_DA LLAMA_DATEARRAY

#define LLAMA_MAX_DATA_DOMAIN_DEPTH 3

/* Defers the solving of a macro, which is especially needed of the macro
 * creates commas, which may confuse surrounding steering macros like
 * BOOST_PP_IIF or similar.
 */
#define LLAMA_INTERNAL_DEFER(id)                                               \
    id                                                                         \
    BOOST_PP_EMPTY                                                             \
    BOOST_PP_EMPTY                                                             \
    BOOST_PP_EMPTY                                                             \
    BOOST_PP_EMPTY                                                             \
    BOOST_PP_EMPTY                                                             \
    BOOST_PP_EMPTY                                                             \
    BOOST_PP_EMPTY                                                             \
    BOOST_PP_EMPTY () () () () () () () ()

/* Evaluates the previously deferred macros */
#define LLAMA_INTERNAL_EVAL(...)                                               \
    LLAMA_INTERNAL_EVAL1(                                                      \
        LLAMA_INTERNAL_EVAL1(                                                  \
            LLAMA_INTERNAL_EVAL1( __VA_ARGS__ )                                \
        )                                                                      \
    )
#define LLAMA_INTERNAL_EVAL1(...)                                              \
    LLAMA_INTERNAL_EVAL2(                                                      \
        LLAMA_INTERNAL_EVAL2(                                                  \
            LLAMA_INTERNAL_EVAL2( __VA_ARGS__ )                                \
        )                                                                      \
    )
#define LLAMA_INTERNAL_EVAL2(...) __VA_ARGS__

#include "DateStructNameTemplate.hpp"
#include "DateStructTemplate.hpp"

/* Creates a struct with naming and type tree of a date domain */
#define LLAMA_DEFINE_DATEDOMAIN( Name, Content )                               \
struct Name                                                                    \
{                                                                              \
    /* Expands to shortcut structs for llama::DateCoord< x, y, z > */          \
    LLAMA_INTERNAL_EVAL( LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_1( Content ) )   \
    using Type = llama::DateStruct<                                            \
        /* Expands DateStruct tree of date domain types */                     \
        LLAMA_INTERNAL_EVAL( LLAMA_INTERNAL_PARSE_DS_CONTENT_1( Content ) )    \
    >;                                                                         \
};

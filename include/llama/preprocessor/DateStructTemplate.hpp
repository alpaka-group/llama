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

// LLAMA_INTERNAL_PARSE_DS_CONTENT_3

#define LLAMA_INTERNAL_PARSE_TUPLE_3( Tuple ) \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATESTRUCT), /* if (typeID == DateStruct) */\
        llama::DateStruct< \
            LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_DS_CONTENT_4)( BOOST_PP_TUPLE_ELEM(2, Tuple) ) /*not implemented*/ \
        > \
    , /* else (typeID == DateStruct) */ \
        BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATEARRAY), /* if (typeID == DateArray) */ \
            llama::DateArray< \
                LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_DA_CONTENT_4)( BOOST_PP_TUPLE_ELEM(2, Tuple) ) /*not implemented*/ \
            > \
        , /* else (typeID == DateArray) */ \
            BOOST_PP_TUPLE_ELEM(2, Tuple) \
        ) /* fi (typeID == DateArray) */ \
    ) /* fi (typeID == DateStruct) */

#define LLAMA_INTERNAL_PARSE_DS_CONTENT_LOOP_3( Z, N, Content ) \
    LLAMA_INTERNAL_DEFER(BOOST_PP_IF(BOOST_PP_EQUAL(N,0),BOOST_PP_EMPTY,BOOST_PP_COMMA))() \
    LLAMA_INTERNAL_PARSE_TUPLE_3( BOOST_PP_TUPLE_ELEM(N, Content) )

#define LLAMA_INTERNAL_PARSE_DS_CONTENT_3( Content ) \
    BOOST_PP_REPEAT( BOOST_PP_TUPLE_SIZE( Content ), LLAMA_INTERNAL_PARSE_DS_CONTENT_LOOP_3, Content )

#define LLAMA_INTERNAL_PARSE_DA_CONTENT_3( Content ) \
    LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_TUPLE_3)( Content ) , \
    BOOST_PP_TUPLE_ELEM(0, Content)

// LLAMA_INTERNAL_PARSE_DS_CONTENT_2

#define LLAMA_INTERNAL_PARSE_TUPLE_2( Tuple ) \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATESTRUCT), /* if (typeID == DateStruct) */\
        llama::DateStruct< \
            LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_DS_CONTENT_3)( BOOST_PP_TUPLE_ELEM(2, Tuple) ) \
        > \
    , /* else (typeID == DateStruct) */ \
        BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATEARRAY), /* if (typeID == DateArray) */ \
            llama::DateArray< \
                LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_DA_CONTENT_3)( BOOST_PP_TUPLE_ELEM(2, Tuple) ) \
            > \
        , /* else (typeID == DateArray) */ \
            BOOST_PP_TUPLE_ELEM(2, Tuple) \
        ) /* fi (typeID == DateArray) */ \
    ) /* fi (typeID == DateStruct) */

#define LLAMA_INTERNAL_PARSE_DS_CONTENT_LOOP_2( Z, N, Content ) \
    LLAMA_INTERNAL_DEFER(BOOST_PP_IF(BOOST_PP_EQUAL(N,0),BOOST_PP_EMPTY,BOOST_PP_COMMA))() \
    LLAMA_INTERNAL_PARSE_TUPLE_2( BOOST_PP_TUPLE_ELEM(N, Content))

#define LLAMA_INTERNAL_PARSE_DS_CONTENT_2( Content ) \
    BOOST_PP_REPEAT( BOOST_PP_TUPLE_SIZE( Content ), LLAMA_INTERNAL_PARSE_DS_CONTENT_LOOP_2, Content )

#define LLAMA_INTERNAL_PARSE_DA_CONTENT_2( Content ) \
    LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_TUPLE_2)( Content ) , \
    BOOST_PP_TUPLE_ELEM(0, Content)

// LLAMA_INTERNAL_PARSE_DS_CONTENT_1

#define LLAMA_INTERNAL_PARSE_TUPLE_1( Tuple ) \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATESTRUCT), /* if (typeID == DateStruct) */\
        llama::DateStruct< \
            LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_DS_CONTENT_2)( BOOST_PP_TUPLE_ELEM(2, Tuple) ) \
        > \
    , /* else (typeID == DateStruct) */ \
        BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATEARRAY), /* if (typeID == DateArray) */ \
            llama::DateArray< \
                LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_DA_CONTENT_2)( BOOST_PP_TUPLE_ELEM(2, Tuple) ) \
            > \
        , /* else (typeID == DateArray) */ \
            BOOST_PP_TUPLE_ELEM(2, Tuple) \
        ) /* fi (typeID == DateArray) */ \
    ) /* fi (typeID == DateStruct) */

#define LLAMA_INTERNAL_PARSE_DS_CONTENT_LOOP_1( Z, N, Content ) \
    LLAMA_INTERNAL_DEFER(BOOST_PP_IF(BOOST_PP_EQUAL(N,0),BOOST_PP_EMPTY,BOOST_PP_COMMA))() \
    LLAMA_INTERNAL_PARSE_TUPLE_1( BOOST_PP_TUPLE_ELEM(N, Content) )

#define LLAMA_INTERNAL_PARSE_DS_CONTENT_1( Content ) \
    BOOST_PP_REPEAT( BOOST_PP_TUPLE_SIZE( Content ), LLAMA_INTERNAL_PARSE_DS_CONTENT_LOOP_1, Content )

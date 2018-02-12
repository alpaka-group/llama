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

// LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_3

#define LLAMA_INTERNAL_PARSE_NAME_TUPLE_3( Tuple, Coord ) \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATESTRUCT), /* if (typeID == DateStruct) */\
        struct BOOST_PP_TUPLE_ELEM(0, Tuple) { \
            LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_4)( BOOST_PP_TUPLE_ELEM(2, Tuple), Coord ) /*not implemented*/ \
        }; \
    , /* else (typeID == DateStruct) */ \
        BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATEARRAY), /* if (typeID == DateArray) */ \
            BOOST_PP_EMPTY() /*DateArray cannot be named*/ \
        , /* else (typeID == DateArray) */ \
            using BOOST_PP_TUPLE_ELEM(0, Tuple) = llama::DateCoord< LLAMA_INTERNAL_DEFER(BOOST_PP_TUPLE_ENUM)(Coord) >; \
        ) /* fi (typeID == DateArray) */ \
    ) /* fi (typeID == DateStruct) */

#define LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_LOOP_3( Z, N, Content ) \
    LLAMA_INTERNAL_PARSE_NAME_TUPLE_3( \
        BOOST_PP_TUPLE_ELEM(N, BOOST_PP_TUPLE_ELEM(0, Content) ), \
        BOOST_PP_TUPLE_PUSH_BACK( BOOST_PP_TUPLE_ELEM(1, Content), N) \
    )

#define LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_3( Content, Coord ) \
    BOOST_PP_REPEAT( BOOST_PP_TUPLE_SIZE( Content ), LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_LOOP_3, (Content, Coord) )

// LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_2

#define LLAMA_INTERNAL_PARSE_NAME_TUPLE_2( Tuple, Coord ) \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATESTRUCT), /* if (typeID == DateStruct) */\
        struct BOOST_PP_TUPLE_ELEM(0, Tuple) { \
            LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_3)( BOOST_PP_TUPLE_ELEM(2, Tuple), Coord ) \
        }; \
    , /* else (typeID == DateStruct) */ \
        BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATEARRAY), /* if (typeID == DateArray) */ \
            BOOST_PP_EMPTY() /*DateArray cannot be named*/ \
        , /* else (typeID == DateArray) */ \
            using BOOST_PP_TUPLE_ELEM(0, Tuple) = llama::DateCoord< LLAMA_INTERNAL_DEFER(BOOST_PP_TUPLE_ENUM)(Coord) >; \
        ) /* fi (typeID == DateArray) */ \
    ) /* fi (typeID == DateStruct) */

#define LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_LOOP_2( Z, N, Content ) \
    LLAMA_INTERNAL_PARSE_NAME_TUPLE_2( \
        BOOST_PP_TUPLE_ELEM(N, BOOST_PP_TUPLE_ELEM(0, Content) ), \
        BOOST_PP_TUPLE_PUSH_BACK( BOOST_PP_TUPLE_ELEM(1, Content), N) \
    )

#define LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_2( Content, Coord ) \
    BOOST_PP_REPEAT( BOOST_PP_TUPLE_SIZE( Content ), LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_LOOP_2, (Content, Coord) )

// LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_1

#define LLAMA_INTERNAL_PARSE_NAME_TUPLE_1( Tuple, Coord ) \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATESTRUCT), /* if (typeID == DateStruct) */\
        struct BOOST_PP_TUPLE_ELEM(0, Tuple) { \
            LLAMA_INTERNAL_DEFER(LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_2)( BOOST_PP_TUPLE_ELEM(2, Tuple), Coord ) \
        }; \
    , /* else (typeID == DateStruct) */ \
        BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM(1, Tuple), LLAMA_DATEARRAY), /* if (typeID == DateArray) */ \
            BOOST_PP_EMPTY() /*DateArray cannot be named*/ \
        , /* else (typeID == DateArray) */ \
            using BOOST_PP_TUPLE_ELEM(0, Tuple) = llama::DateCoord< LLAMA_INTERNAL_DEFER(BOOST_PP_TUPLE_ENUM)(Coord) >; \
        ) /* fi (typeID == DateArray) */ \
    ) /* fi (typeID == DateStruct) */

#define LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_LOOP_1( Z, N, Content ) \
    LLAMA_INTERNAL_PARSE_NAME_TUPLE_1( BOOST_PP_TUPLE_ELEM(N, Content), (N) )

#define LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_1( Content ) \
    BOOST_PP_REPEAT( BOOST_PP_TUPLE_SIZE( Content ), LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_LOOP_1, Content )

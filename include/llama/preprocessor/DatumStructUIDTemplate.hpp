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

// LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_3

/* 3rd layer of internal uid macro for a DatumStruct tuple */
#define LLAMA_INTERNAL_PARSE_UID_TUPLE_3( Tuple, Coord )                       \
    template< typename Unused >                                                \
    struct UID<                                                                \
            Unused,                                                            \
            LLAMA_INTERNAL_DEFER(                                              \
            BOOST_PP_TUPLE_ENUM                                                \
        )( Coord )                                                             \
    >                                                                          \
    {                                                                          \
        template< std::size_t T_i >                                            \
        struct Name                                                            \
        {                                                                      \
            static constexpr unsigned char value =                             \
                BOOST_PP_STRINGIZE(                                            \
                    BOOST_PP_TUPLE_ELEM( 0, Tuple )                            \
                ) [ T_i ];                                                     \
        };                                                                     \
        static constexpr std::size_t length = sizeof(                          \
                BOOST_PP_STRINGIZE(                                            \
                    BOOST_PP_TUPLE_ELEM( 0, Tuple )                            \
                ) ) - 1;                                                       \
        using type = typename llama::MakeUniqueIdentifier<                     \
                Name,                                                          \
                0,                                                             \
                length                                                         \
            >::type;                                                           \
    };                                                                         \
    /* if (typeID == DatumStruct) */                                           \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM( 1, Tuple ), LLAMA_DS ), \
            LLAMA_INTERNAL_DEFER(                                              \
                LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_4                          \
            )( BOOST_PP_TUPLE_ELEM( 2, Tuple ), Coord ) /*not implemented*/    \
    , /* else (typeID == DatumStruct) */                                       \
        /* if (typeID == DatumArray) */                                        \
        BOOST_PP_IIF( BOOST_PP_EQUAL(                                          \
            BOOST_PP_TUPLE_ELEM( 1, Tuple ),                                   \
            LLAMA_DA                                                           \
        ),                                                                     \
            /*DatumArray TODO */                                               \
            BOOST_PP_EMPTY()                                                   \
        , /* else (typeID == DatumArray) */                                    \
            BOOST_PP_EMPTY()                                                   \
        ) /* fi (typeID == DatumArray) */                                      \
    ) /* fi (typeID == DatumStruct) */

/* 3rd layer of internal loop function for naming a DatumStruct */
#define LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_LOOP_3( Z, N, Content )            \
    LLAMA_INTERNAL_PARSE_UID_TUPLE_3(                                          \
        BOOST_PP_TUPLE_ELEM( N, BOOST_PP_TUPLE_ELEM( 0, Content ) ),           \
        BOOST_PP_TUPLE_PUSH_BACK( BOOST_PP_TUPLE_ELEM( 1, Content ), N )       \
    )

/* 3rd layer of internal loop caller for naming a DatumStruct */
#define LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_3( Content, Coord )                \
    BOOST_PP_REPEAT(                                                           \
        BOOST_PP_TUPLE_SIZE( Content ),                                        \
        LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_LOOP_3,                            \
        (Content, Coord)                                                       \
    )

// LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_2

/* 2nd layer of internal uid macro for a DatumStruct tuple */
#define LLAMA_INTERNAL_PARSE_UID_TUPLE_2( Tuple, Coord )                       \
    template< typename Unused >                                                \
    struct UID<                                                                \
            Unused,                                                            \
            LLAMA_INTERNAL_DEFER(                                              \
            BOOST_PP_TUPLE_ENUM                                                \
        )( Coord )                                                             \
    >                                                                          \
    {                                                                          \
        template< std::size_t T_i >                                            \
        struct Name                                                            \
        {                                                                      \
            static constexpr unsigned char value =                             \
                BOOST_PP_STRINGIZE(                                            \
                    BOOST_PP_TUPLE_ELEM( 0, Tuple )                            \
                ) [ T_i ];                                                     \
        };                                                                     \
        static constexpr std::size_t length = sizeof(                          \
                BOOST_PP_STRINGIZE(                                            \
                    BOOST_PP_TUPLE_ELEM( 0, Tuple )                            \
                ) ) - 1;                                                       \
        using type = typename llama::MakeUniqueIdentifier<                     \
                Name,                                                          \
                0,                                                             \
                length                                                         \
            >::type;                                                           \
    };                                                                         \
    /* if (typeID == DatumStruct) */                                           \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM( 1, Tuple ), LLAMA_DS ), \
            LLAMA_INTERNAL_DEFER(                                              \
                LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_3                          \
            )( BOOST_PP_TUPLE_ELEM( 2, Tuple ), Coord )                        \
    , /* else (typeID == DatumStruct) */                                       \
        /* if (typeID == DatumArray) */                                        \
        BOOST_PP_IIF( BOOST_PP_EQUAL(                                          \
            BOOST_PP_TUPLE_ELEM( 1, Tuple ),                                   \
            LLAMA_DA                                                           \
        ),                                                                     \
            /*DatumArray TODO */                                               \
            BOOST_PP_EMPTY()                                                   \
        , /* else (typeID == DatumArray) */                                    \
            BOOST_PP_EMPTY()                                                   \
        ) /* fi (typeID == DatumArray) */                                      \
    ) /* fi (typeID == DatumStruct) */

/* 2nd layer of internal loop function for naming a DatumStruct */
#define LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_LOOP_2( Z, N, Content )            \
    LLAMA_INTERNAL_PARSE_UID_TUPLE_2(                                          \
        BOOST_PP_TUPLE_ELEM( N, BOOST_PP_TUPLE_ELEM( 0, Content ) ),           \
        BOOST_PP_TUPLE_PUSH_BACK( BOOST_PP_TUPLE_ELEM( 1, Content ), N )       \
    )

/* 2nd layer of internal loop caller for naming a DatumStruct */
#define LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_2( Content, Coord )                \
    BOOST_PP_REPEAT(                                                           \
        BOOST_PP_TUPLE_SIZE( Content ),                                        \
        LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_LOOP_2,                            \
        (Content, Coord)                                                       \
    )


// LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_1

/* 1st layer of internal uid macro for a DatumStruct tuple */
#define LLAMA_INTERNAL_PARSE_UID_TUPLE_1( Tuple, Coord )                       \
    template< typename Unused >                                                \
    struct UID<                                                                \
            Unused,                                                            \
            LLAMA_INTERNAL_DEFER(                                              \
            BOOST_PP_TUPLE_ENUM                                                \
        )( Coord )                                                             \
    >                                                                          \
    {                                                                          \
        template< std::size_t T_i >                                            \
        struct Name                                                            \
        {                                                                      \
            static constexpr unsigned char value =                             \
                BOOST_PP_STRINGIZE(                                            \
                    BOOST_PP_TUPLE_ELEM( 0, Tuple )                            \
                ) [ T_i ];                                                     \
        };                                                                     \
        static constexpr std::size_t length = sizeof(                          \
                BOOST_PP_STRINGIZE(                                            \
                    BOOST_PP_TUPLE_ELEM( 0, Tuple )                            \
                ) ) - 1;                                                       \
        using type = typename llama::MakeUniqueIdentifier<                     \
                Name,                                                          \
                0,                                                             \
                length                                                         \
            >::type;                                                           \
    };                                                                         \
    /* if (typeID == DatumStruct) */                                           \
    BOOST_PP_IIF( BOOST_PP_EQUAL( BOOST_PP_TUPLE_ELEM( 1, Tuple ), LLAMA_DS ), \
            LLAMA_INTERNAL_DEFER(                                              \
                LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_2                          \
            )( BOOST_PP_TUPLE_ELEM( 2, Tuple ), Coord )                        \
    , /* else (typeID == DatumStruct) */                                       \
        /* if (typeID == DatumArray) */                                        \
        BOOST_PP_IIF( BOOST_PP_EQUAL(                                          \
            BOOST_PP_TUPLE_ELEM( 1, Tuple ),                                   \
            LLAMA_DA                                                           \
        ),                                                                     \
            /*DatumArray TODO */                                               \
            BOOST_PP_EMPTY()                                                   \
        , /* else (typeID == DatumArray) */                                    \
            BOOST_PP_EMPTY()                                                   \
        ) /* fi (typeID == DatumArray) */                                      \
    ) /* fi (typeID == DatumStruct) */

/* 1st layer of internal loop function for naming a DatumStruct */
#define LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_LOOP_1( Z, N, Content )            \
    LLAMA_INTERNAL_PARSE_UID_TUPLE_1(                                          \
        BOOST_PP_TUPLE_ELEM( N, Content ),                                     \
        ( N )                                                                  \
    )

/* 1st layer of internal loop caller for naming a DatumStruct */
#define LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_1( Content )                       \
    BOOST_PP_REPEAT(                                                           \
        BOOST_PP_TUPLE_SIZE( Content ),                                        \
        LLAMA_INTERNAL_PARSE_UID_DS_CONTENT_LOOP_1,                            \
        Content                                                                \
    )

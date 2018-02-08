#pragma once

#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/tuple/push_back.hpp>

#define LLAMA_ATOMTYPE 0
#define LLAMA_DATESTRUCT 1
#define LLAMA_DATEARRAY 2

#define LLAMA_MAX_DATA_DOMAIN_DEPTH 3

#define LLAMA_INTERNAL_DEFER(id) id BOOST_PP_EMPTY BOOST_PP_EMPTY BOOST_PP_EMPTY BOOST_PP_EMPTY BOOST_PP_EMPTY BOOST_PP_EMPTY BOOST_PP_EMPTY BOOST_PP_EMPTY () () () () () () () ()
#define LLAMA_INTERNAL_EVAL(...)  LLAMA_INTERNAL_EVAL1(LLAMA_INTERNAL_EVAL1(LLAMA_INTERNAL_EVAL1(__VA_ARGS__)))
#define LLAMA_INTERNAL_EVAL1(...) LLAMA_INTERNAL_EVAL2(LLAMA_INTERNAL_EVAL2(LLAMA_INTERNAL_EVAL2(__VA_ARGS__)))
#define LLAMA_INTERNAL_EVAL2(...) __VA_ARGS__

#include "DateStructNameTemplate.hpp"
#include "DateStructTemplate.hpp"

#define LLAMA_INTERNAL_ITERATE_SEQ(R, DATA, ELEM) ELEM

#define LLAMA_DEFINE_DATEDOMAIN( Name, Content ) \
struct Name \
{ \
	/* TODO: Name accessor types */ \
	 \
		LLAMA_INTERNAL_EVAL(LLAMA_INTERNAL_PARSE_NAME_DS_CONTENT_1( Content ))\
		using Type = llama::DateStruct< \
			LLAMA_INTERNAL_EVAL(LLAMA_INTERNAL_PARSE_DS_CONTENT_1( Content )) \
		>; \
};

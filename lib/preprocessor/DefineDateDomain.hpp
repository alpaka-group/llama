#pragma once

#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/facilities/expand.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#define LLAMA_ATOMTYPE 0
#define LLAMA_DATESTRUCT 1
#define LLAMA_DATEARRAY 2

#define LLAMA_INTERNAL_CREATE_TYPE( Type, Content ) Type<float>

#define LLAMA_INTERNAL_TYPE_BY_KIND( Kind, Rest ) \
	BOOST_PP_IIF( BOOST_PP_EQUAL( Kind , LLAMA_DATEARRAY) , \
		llama::DateArray, \
	BOOST_PP_IIF( BOOST_PP_EQUAL( Kind , LLAMA_DATESTRUCT) , \
		llama::DateStruct, \
	Rest ) )

#include "DateStructTemplate.hpp"

#define LLAMA_INTERNAL_ITERATE_SEQ(R, DATA, ELEM) ELEM

#define LLAMA_DEFINE_DATEDOMAIN( Name, Content ) \
struct Name \
{ \
	/* TODO: Name accessor types */ \
	 \
		using Type = llama::DateStruct< \
			LLAMA_INTERNAL_PARSE_DS_CONTENT_1( Content ) \
		>; \
};

// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

TEST_CASE("ReplacePlaceholders")
{
    using namespace boost::mp11;
    using namespace tag;

    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<int, A, B, C>, int>);
    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<std::string, A, B, C>, std::string>);
    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<Particle, A, B, C>, Particle>);

    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<_1, A, B, C>, A>);
    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<_2, A, B, C>, B>);
    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<_3, A, B, C>, C>);

    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<mp_list<_1>, A, B, C>, mp_list<A>>);
    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<mp_list<_1, _2>, A, B, C>, mp_list<A, B>>);
    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<mp_list<_3, _1, _2>, A, B, C>, mp_list<C, A, B>>);

    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<mp_list<_3, _1, _1>, A, B, C>, mp_list<C, A, A>>);
    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<mp_list<_2, _2, _2>, A, B, C>, mp_list<B, B, B>>);

    STATIC_REQUIRE(std::is_same_v<llama::ReplacePlaceholders<mp_list<mp_list<_2>>, A, B, C>, mp_list<mp_list<B>>>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::ReplacePlaceholders<mp_list<mp_list<_2, _1>, _2, mp_list<_2, mp_list<_3>>>, A, B, C>,
                   mp_list<mp_list<B, A>, B, mp_list<B, mp_list<C>>>>);
}

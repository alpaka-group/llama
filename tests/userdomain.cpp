#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

TEST_CASE("userdomain ctor")
{
    llama::UserDomain<1> ud{};
    CHECK(ud[0] == 0);
}

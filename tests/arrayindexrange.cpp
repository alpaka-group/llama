// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "common.hpp"

TEST_CASE("ArrayIndexIterator.concepts")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 3>;
    STATIC_REQUIRE(std::is_same_v<
                   std::iterator_traits<llama::ArrayIndexIterator<ArrayExtents>>::iterator_category,
                   std::random_access_iterator_tag>);

#ifdef __cpp_lib_concepts
    STATIC_REQUIRE(std::random_access_iterator<llama::ArrayIndexIterator<ArrayExtents>>);
#endif
}

TEST_CASE("ArrayIndexIterator")
{
    llama::ArrayIndexRange r{llama::ArrayExtentsDynamic<int, 2>{3, 3}};

    llama::ArrayIndexIterator it = std::begin(r);
    CHECK(*it == llama::ArrayIndex{0, 0});
    it++;
    CHECK(*it == llama::ArrayIndex{0, 1});
    ++it;
    CHECK(*it == llama::ArrayIndex{0, 2});
    --it;
    CHECK(*it == llama::ArrayIndex{0, 1});
    it--;
    CHECK(*it == llama::ArrayIndex{0, 0});

    it = std::begin(r);
    it += 2;
    CHECK(*it == llama::ArrayIndex{0, 2});
    it += 2;
    CHECK(*it == llama::ArrayIndex{1, 1});
    it -= 2;
    CHECK(*it == llama::ArrayIndex{0, 2});
    it -= 2;
    CHECK(*it == llama::ArrayIndex{0, 0});

    it = std::begin(r);
    CHECK(it[2] == llama::ArrayIndex{0, 2});
    CHECK(it[4] == llama::ArrayIndex{1, 1});
    CHECK(it[8] == llama::ArrayIndex{2, 2});

    it = std::begin(r);
    CHECK(*(it + 8) == llama::ArrayIndex{2, 2});
    CHECK(*(8 + it) == llama::ArrayIndex{2, 2});

    it += 8;
    CHECK(*it == llama::ArrayIndex{2, 2});
    CHECK(*(it - 8) == llama::ArrayIndex{0, 0});
    it -= 8;
    CHECK(*it == llama::ArrayIndex{0, 0});

    CHECK(std::end(r) - std::begin(r) == 9);
    CHECK(std::begin(r) - std::end(r) == -9);

    it = std::begin(r) + 4;
    CHECK(*it == llama::ArrayIndex{1, 1});
    CHECK(std::end(r) - it == 5);
    CHECK(it - std::begin(r) == 4);

    it = std::begin(r);
    CHECK(it == it);
    CHECK(it != it + 1);
    CHECK(it < it + 1);
}


TEST_CASE("ArrayIndexIterator.operator+=")
{
    std::vector<llama::ArrayIndex<int, 2>> indices;
    const llama::ArrayIndexRange r{llama::ArrayExtentsDynamic<int, 2>{3, 4}};
    for(auto it = r.begin(); it != r.end(); it += 2)
        indices.push_back(*it);

    CHECK(indices == std::vector<llama::ArrayIndex<int, 2>>{{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 2}});
}

TEST_CASE("ArrayIndexIterator.constexpr")
{
    constexpr auto r = [&]() constexpr
    {
        bool b = true;
        llama::ArrayIndexIterator it = std::begin(llama::ArrayIndexRange{llama::ArrayExtentsDynamic<int, 2>{3, 3}});
        b &= *it == llama::ArrayIndex{0, 0};
        it++;
        b &= *it == llama::ArrayIndex{0, 1};
        ++it;
        b &= *it == llama::ArrayIndex{0, 2};
        return b;
    }();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayIndexRange.1D")
{
    std::vector<llama::ArrayIndex<int, 1>> indices;
    for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<int, 1>{3}})
        indices.push_back(ai);

    CHECK(indices == std::vector<llama::ArrayIndex<int, 1>>{{0}, {1}, {2}});
}

TEST_CASE("ArrayIndexRange.2D")
{
    std::vector<llama::ArrayIndex<int, 2>> indices;
    for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<int, 2>{3, 3}})
        indices.push_back(ai);

    CHECK(
        indices
        == std::vector<
            llama::ArrayIndex<int, 2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("ArrayIndexRange.3D")
{
    std::vector<llama::ArrayIndex<int, 3>> indices;
    for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<int, 3>{3, 3, 3}})
        indices.push_back(ai);

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<int, 3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

#if CAN_USE_RANGES
#    include <ranges>

TEST_CASE("ArrayIndexRange.concepts")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<int, 3>;
    STATIC_REQUIRE(std::ranges::range<llama::ArrayIndexRange<ArrayExtents>>);
    STATIC_REQUIRE(std::ranges::random_access_range<llama::ArrayIndexRange<ArrayExtents>>);
    // STATIC_REQUIRE(std::ranges::view<llama::ArrayIndexRange<ArrayExtents>>);
}

TEST_CASE("ArrayIndexRange.3D.reverse")
{
    llama::ArrayExtentsDynamic<int, 3> extents{3, 3, 3};

    std::vector<llama::ArrayIndex<int, 3>> indices;
    for(auto ai : llama::ArrayIndexRange{extents} | std::views::reverse)
        indices.push_back(ai);

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<int, 3>>{
            {{2, 2, 2}, {2, 2, 1}, {2, 2, 0}, {2, 1, 2}, {2, 1, 1}, {2, 1, 0}, {2, 0, 2}, {2, 0, 1}, {2, 0, 0},
             {1, 2, 2}, {1, 2, 1}, {1, 2, 0}, {1, 1, 2}, {1, 1, 1}, {1, 1, 0}, {1, 0, 2}, {1, 0, 1}, {1, 0, 0},
             {0, 2, 2}, {0, 2, 1}, {0, 2, 0}, {0, 1, 2}, {0, 1, 1}, {0, 1, 0}, {0, 0, 2}, {0, 0, 1}, {0, 0, 0}}});
}
#endif

TEST_CASE("ArrayIndexRange.1D.constexpr")
{
    constexpr auto r = []() constexpr
    {
        int i = 0;
        for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<int, 1>{3}})
        {
            if(i == 0 && ai != llama::ArrayIndex<int, 1>{0})
                return false;
            if(i == 1 && ai != llama::ArrayIndex<int, 1>{1})
                return false;
            if(i == 2 && ai != llama::ArrayIndex<int, 1>{2})
                return false;
            i++;
        }

        return true;
    }();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayIndexRange.3D.destructering")
{
    for(auto [x, y, z] : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<int, 3>{1, 1, 1}})
    {
        CHECK(x == 0);
        CHECK(y == 0);
        CHECK(z == 0);
    }
}

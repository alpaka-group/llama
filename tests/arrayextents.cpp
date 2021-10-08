#include "common.hpp"

TEST_CASE("ArrayExtents.CTAD")
{
    llama::ArrayExtents ad0{};
    llama::ArrayExtents ad1{1};
    llama::ArrayExtents ad2{1, 1};
    llama::ArrayExtents ad3{1, 1, 1};

    STATIC_REQUIRE(std::is_same_v<decltype(ad0), llama::ArrayExtents<>>);
    STATIC_REQUIRE(std::is_same_v<decltype(ad1), llama::ArrayExtents<llama::dyn>>);
    STATIC_REQUIRE(std::is_same_v<decltype(ad2), llama::ArrayExtents<llama::dyn, llama::dyn>>);
    STATIC_REQUIRE(std::is_same_v<decltype(ad3), llama::ArrayExtents<llama::dyn, llama::dyn, llama::dyn>>);

    STATIC_REQUIRE(decltype(ad0)::rank == 0);
    STATIC_REQUIRE(decltype(ad1)::rank == 1);
    STATIC_REQUIRE(decltype(ad2)::rank == 2);
    STATIC_REQUIRE(decltype(ad3)::rank == 3);
}

TEST_CASE("ArrayExtents.dim0")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x1 = view(llama::ArrayIndex{})(tag::Pos{}, tag::X{});
    double& x2 = view()(tag::Pos{}, tag::X{});
    x1 = 0;
    x2 = 0;
}

TEST_CASE("ArrayExtents.dim1.dynamic")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{16}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.dim1.static")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<16>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.dim2.dynamic")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{16, 16}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1, 1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.dim2.static")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<16, 16>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1, 1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.dim3.dynamic")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{16, 16, 16}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1, 1, 1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.dim3.static")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<16, 16, 16>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1, 1, 1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.dim10.dynamic")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.dim10.static")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<2, 2, 2, 2, 2, 2, 2, 2, 2, 2>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{1, 1, 1, 1, 1, 1, 1, 1, 1, 1})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtents.ctor")
{
    llama::ArrayExtentsDynamic<1> extents{};
    CHECK(extents[0] == 0);
}

TEST_CASE("ArrayExtents.toArray")
{
    CHECK(llama::ArrayExtents{}.toArray() == llama::Array<std::size_t, 0>{});
    CHECK(llama::ArrayExtents<llama::dyn>{42}.toArray() == llama::Array<std::size_t, 1>{42});
    CHECK(llama::ArrayExtents<llama::dyn, llama::dyn>{42, 43}.toArray() == llama::Array<std::size_t, 2>{42, 43});
    CHECK(
        llama::ArrayExtents<llama::dyn, 43, llama::dyn>{42, 44}.toArray() == llama::Array<std::size_t, 3>{42, 43, 44});
}

TEST_CASE("ArrayIndexIterator.concepts")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<3>;
    STATIC_REQUIRE(std::is_same_v<
                   std::iterator_traits<llama::ArrayIndexIterator<ArrayExtents>>::iterator_category,
                   std::random_access_iterator_tag>);

#ifdef __cpp_lib_concepts
    STATIC_REQUIRE(std::random_access_iterator<llama::ArrayIndexIterator<ArrayExtents>>);
#endif
}

TEST_CASE("ArrayIndexIterator")
{
    llama::ArrayIndexRange r{llama::ArrayExtentsDynamic<2>{3, 3}};

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
    std::vector<llama::ArrayIndex<2>> indices;
    llama::ArrayIndexRange r{llama::ArrayExtentsDynamic<2>{3, 4}};
    for(auto it = r.begin(); it != r.end(); it += 2)
        indices.push_back(*it);

    CHECK(indices == std::vector<llama::ArrayIndex<2>>{{0, 0}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 2}});
}

TEST_CASE("ArrayIndexIterator.constexpr")
{
    constexpr auto r = [&]() constexpr
    {
        bool b = true;
        llama::ArrayIndexIterator it = std::begin(llama::ArrayIndexRange{llama::ArrayExtentsDynamic<2>{3, 3}});
        b &= *it == llama::ArrayIndex{0, 0};
        it++;
        b &= *it == llama::ArrayIndex{0, 1};
        ++it;
        b &= *it == llama::ArrayIndex{0, 2};
        return b;
    }
    ();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayDimsIndexRange1D")
{
    std::vector<llama::ArrayIndex<1>> indices;
    for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<1>{3}})
        indices.push_back(ai);

    CHECK(indices == std::vector<llama::ArrayIndex<1>>{{0}, {1}, {2}});
}

TEST_CASE("ArrayDimsIndexRange2D")
{
    std::vector<llama::ArrayIndex<2>> indices;
    for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<2>{3, 3}})
        indices.push_back(ai);

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("ArrayDimsIndexRange3D")
{
    std::vector<llama::ArrayIndex<3>> indices;
    for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<3>{3, 3, 3}})
        indices.push_back(ai);

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

#if CAN_USE_RANGES
#    include <ranges>

TEST_CASE("ArrayIndexRange.concepts")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<3>;
    STATIC_REQUIRE(std::ranges::range<llama::ArrayIndexRange<ArrayExtents>>);
    STATIC_REQUIRE(std::ranges::random_access_range<llama::ArrayIndexRange<ArrayExtents>>);
    // STATIC_REQUIRE(std::ranges::view<llama::ArrayIndexRange<ArrayExtents>>);
}

TEST_CASE("ArrayDimsIndexRange3D.reverse")
{
    llama::ArrayExtentsDynamic<3> extents{3, 3, 3};

    std::vector<llama::ArrayIndex<3>> indices;
    for(auto ai : llama::ArrayIndexRange{extents} | std::views::reverse)
        indices.push_back(ai);

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<3>>{
            {{2, 2, 2}, {2, 2, 1}, {2, 2, 0}, {2, 1, 2}, {2, 1, 1}, {2, 1, 0}, {2, 0, 2}, {2, 0, 1}, {2, 0, 0},
             {1, 2, 2}, {1, 2, 1}, {1, 2, 0}, {1, 1, 2}, {1, 1, 1}, {1, 1, 0}, {1, 0, 2}, {1, 0, 1}, {1, 0, 0},
             {0, 2, 2}, {0, 2, 1}, {0, 2, 0}, {0, 1, 2}, {0, 1, 1}, {0, 1, 0}, {0, 0, 2}, {0, 0, 1}, {0, 0, 0}}});
}
#endif

TEST_CASE("ArrayDimsIndexRange1D.constexpr")
{
    constexpr auto r = []() constexpr
    {
        int i = 0;
        for(auto ai : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<1>{3}})
        {
            if(i == 0 && ai != llama::ArrayIndex<1>{0})
                return false;
            if(i == 1 && ai != llama::ArrayIndex<1>{1})
                return false;
            if(i == 2 && ai != llama::ArrayIndex<1>{2})
                return false;
            i++;
        }

        return true;
    }
    ();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayDimsIndexRange3D.destructering")
{
    for(auto [x, y, z] : llama::ArrayIndexRange{llama::ArrayExtentsDynamic<3>{1, 1, 1}})
    {
        CHECK(x == 0);
        CHECK(y == 0);
        CHECK(z == 0);
    }
}

TEST_CASE("Morton")
{
    using ArrayExtents = llama::ArrayExtentsDynamic<2>;

    llama::mapping::LinearizeArrayDimsMorton lin;
    CHECK(lin.size(ArrayExtents{2, 3}) == 4 * 4);
    CHECK(lin.size(ArrayExtents{2, 4}) == 4 * 4);
    CHECK(lin.size(ArrayExtents{2, 5}) == 8 * 8);
    CHECK(lin.size(ArrayExtents{8, 8}) == 8 * 8);

    CHECK(lin(llama::ArrayIndex{0, 0}, ArrayExtents{}) == 0);
    CHECK(lin(llama::ArrayIndex{0, 1}, ArrayExtents{}) == 1);
    CHECK(lin(llama::ArrayIndex{0, 2}, ArrayExtents{}) == 4);
    CHECK(lin(llama::ArrayIndex{0, 3}, ArrayExtents{}) == 5);
    CHECK(lin(llama::ArrayIndex{1, 0}, ArrayExtents{}) == 2);
    CHECK(lin(llama::ArrayIndex{1, 1}, ArrayExtents{}) == 3);
    CHECK(lin(llama::ArrayIndex{1, 2}, ArrayExtents{}) == 6);
    CHECK(lin(llama::ArrayIndex{1, 3}, ArrayExtents{}) == 7);
    CHECK(lin(llama::ArrayIndex{2, 0}, ArrayExtents{}) == 8);
    CHECK(lin(llama::ArrayIndex{2, 1}, ArrayExtents{}) == 9);
    CHECK(lin(llama::ArrayIndex{2, 2}, ArrayExtents{}) == 12);
    CHECK(lin(llama::ArrayIndex{2, 3}, ArrayExtents{}) == 13);
    CHECK(lin(llama::ArrayIndex{3, 0}, ArrayExtents{}) == 10);
    CHECK(lin(llama::ArrayIndex{3, 1}, ArrayExtents{}) == 11);
    CHECK(lin(llama::ArrayIndex{3, 2}, ArrayExtents{}) == 14);
    CHECK(lin(llama::ArrayIndex{3, 3}, ArrayExtents{}) == 15);
}

TEST_CASE("forEachADCoord_1D")
{
    llama::ArrayExtents extents{3};

    std::vector<llama::ArrayIndex<1>> indices;
    llama::forEachADCoord(extents, [&](llama::ArrayIndex<1> ai) { indices.push_back(ai); });

    CHECK(indices == std::vector<llama::ArrayIndex<1>>{{0}, {1}, {2}});
}

TEST_CASE("forEachADCoord_2D")
{
    llama::ArrayExtents extents{3, 3};

    std::vector<llama::ArrayIndex<2>> indices;
    llama::forEachADCoord(extents, [&](llama::ArrayIndex<2> ai) { indices.push_back(ai); });

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("forEachADCoord_3D")
{
    llama::ArrayExtents extents{3, 3, 3};

    std::vector<llama::ArrayIndex<3>> indices;
    llama::forEachADCoord(extents, [&](llama::ArrayIndex<3> ai) { indices.push_back(ai); });

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

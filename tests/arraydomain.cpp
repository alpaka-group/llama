#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
} // namespace tag

using Name = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>,
        llama::DE<tag::Z, float>
    >>
>;
// clang-format on

TEST_CASE("ArrayDomain.CTAD")
{
    llama::ArrayDomain ad0{};
    llama::ArrayDomain ad1{1};
    llama::ArrayDomain ad2{1, 1};
    llama::ArrayDomain ad3{1, 1, 1};

    STATIC_REQUIRE(decltype(ad0)::rank == 0);
    STATIC_REQUIRE(decltype(ad1)::rank == 1);
    STATIC_REQUIRE(decltype(ad2)::rank == 2);
    STATIC_REQUIRE(decltype(ad3)::rank == 3);
}

TEST_CASE("ArrayDomain.dim0")
{
    using ArrayDomain = llama::ArrayDomain<0>;
    ArrayDomain arrayDomain{};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain{})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDomain.dim1")
{
    using ArrayDomain = llama::ArrayDomain<1>;
    ArrayDomain arrayDomain{16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain{0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDomain.dim2")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain{0, 0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDomain.dim3")
{
    using ArrayDomain = llama::ArrayDomain<3>;
    ArrayDomain arrayDomain{16, 16, 16};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain{0, 0, 0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDomain.dim10")
{
    using ArrayDomain = llama::ArrayDomain<10>;
    ArrayDomain arrayDomain{2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

    using Mapping = llama::mapping::SoA<ArrayDomain, Name>;
    Mapping mapping{arrayDomain};
    auto view = allocView(mapping);

    float& x = view(ArrayDomain{0, 0, 0, 0, 0, 0, 0, 0, 0, 0})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayDomain.ctor")
{
    llama::ArrayDomain<1> ad{};
    CHECK(ad[0] == 0);
}

TEST_CASE("ArrayDomainIndexIterator.concepts")
{
    STATIC_REQUIRE(std::is_same_v<
                   std::iterator_traits<llama::ArrayDomainIndexIterator<3>>::iterator_category,
                   std::random_access_iterator_tag>);

#ifdef __cpp_concepts
    STATIC_REQUIRE(std::random_access_iterator<llama::ArrayDomainIndexIterator<3>>);
#endif
}

TEST_CASE("ArrayDomainIndexIterator")
{
    llama::ArrayDomainIndexRange r{llama::ArrayDomain<2>{3, 3}};

    llama::ArrayDomainIndexIterator it = std::begin(r);
    CHECK(*it == llama::ArrayDomain{0, 0});
    it++;
    CHECK(*it == llama::ArrayDomain{0, 1});
    ++it;
    CHECK(*it == llama::ArrayDomain{0, 2});
    --it;
    CHECK(*it == llama::ArrayDomain{0, 1});
    it--;
    CHECK(*it == llama::ArrayDomain{0, 0});

    it = std::begin(r);
    it += 2;
    CHECK(*it == llama::ArrayDomain{0, 2});
    it += 2;
    CHECK(*it == llama::ArrayDomain{1, 1});
    it -= 2;
    CHECK(*it == llama::ArrayDomain{0, 2});
    it -= 2;
    CHECK(*it == llama::ArrayDomain{0, 0});

    it = std::begin(r);
    CHECK(it[2] == llama::ArrayDomain{0, 2});
    CHECK(it[4] == llama::ArrayDomain{1, 1});
    CHECK(it[8] == llama::ArrayDomain{2, 2});

    it = std::begin(r);
    CHECK(*(it + 8) == llama::ArrayDomain{2, 2});
    CHECK(*(8 + it) == llama::ArrayDomain{2, 2});

    it += 8;
    CHECK(*it == llama::ArrayDomain{2, 2});
    CHECK(*(it - 8) == llama::ArrayDomain{0, 0});
    it -= 8;
    CHECK(*it == llama::ArrayDomain{0, 0});

    CHECK(std::end(r) - std::begin(r) == 9);
    CHECK(std::begin(r) - std::end(r) == -9);

    it = std::begin(r) + 4;
    CHECK(*it == llama::ArrayDomain{1, 1});
    CHECK(std::end(r) - it == 5);
    CHECK(it - std::begin(r) == 4);

    it = std::begin(r);
    CHECK(it == it);
    CHECK(it != it + 1);
    CHECK(it < it + 1);
}

TEST_CASE("ArrayDomainIndexIterator.constexpr")
{
    constexpr auto r = [&]() constexpr
    {
        bool b = true;
        llama::ArrayDomainIndexIterator it = std::begin(llama::ArrayDomainIndexRange{llama::ArrayDomain<2>{3, 3}});
        b &= *it == llama::ArrayDomain{0, 0};
        it++;
        b &= *it == llama::ArrayDomain{0, 1};
        ++it;
        b &= *it == llama::ArrayDomain{0, 2};
        return b;
    }
    ();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayDomainIndexRange1D")
{
    llama::ArrayDomain<1> ad{3};

    std::vector<llama::ArrayDomain<1>> coords;
    for (auto coord : llama::ArrayDomainIndexRange{ad})
        coords.push_back(coord);

    CHECK(coords == std::vector<llama::ArrayDomain<1>>{{0}, {1}, {2}});
}

TEST_CASE("ArrayDomainIndexRange2D")
{
    llama::ArrayDomain<2> ad{3, 3};

    std::vector<llama::ArrayDomain<2>> coords;
    for (auto coord : llama::ArrayDomainIndexRange{ad})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::ArrayDomain<2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("ArrayDomainIndexRange3D")
{
    llama::ArrayDomain<3> ad{3, 3, 3};

    std::vector<llama::ArrayDomain<3>> coords;
    for (auto coord : llama::ArrayDomainIndexRange{ad})
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::ArrayDomain<3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

#if CAN_USE_RANGES
#    include <ranges>

TEST_CASE("ArrayDomainIndexRange.concepts")
{
    STATIC_REQUIRE(std::ranges::range<llama::ArrayDomainIndexRange<3>>);
    STATIC_REQUIRE(std::ranges::random_access_range<llama::ArrayDomainIndexRange<3>>);
    // STATIC_REQUIRE(std::ranges::view<llama::ArrayDomainIndexRange<3>>);
}

TEST_CASE("ArrayDomainIndexRange3D.reverse")
{
    llama::ArrayDomain<3> ad{3, 3, 3};

    std::vector<llama::ArrayDomain<3>> coords;
    for (auto coord : llama::ArrayDomainIndexRange{ad} | std::views::reverse)
        coords.push_back(coord);

    CHECK(
        coords
        == std::vector<llama::ArrayDomain<3>>{
            {{2, 2, 2}, {2, 2, 1}, {2, 2, 0}, {2, 1, 2}, {2, 1, 1}, {2, 1, 0}, {2, 0, 2}, {2, 0, 1}, {2, 0, 0},
             {1, 2, 2}, {1, 2, 1}, {1, 2, 0}, {1, 1, 2}, {1, 1, 1}, {1, 1, 0}, {1, 0, 2}, {1, 0, 1}, {1, 0, 0},
             {0, 2, 2}, {0, 2, 1}, {0, 2, 0}, {0, 1, 2}, {0, 1, 1}, {0, 1, 0}, {0, 0, 2}, {0, 0, 1}, {0, 0, 0}}});
}
#endif

TEST_CASE("ArrayDomainIndexRange1D.constexpr")
{
    constexpr auto r = []() constexpr
    {
        llama::ArrayDomain<1> ad{3};
        int i = 0;
        for (auto coord : llama::ArrayDomainIndexRange{ad})
        {
            if (i == 0 && coord != llama::ArrayDomain<1>{0})
                return false;
            if (i == 1 && coord != llama::ArrayDomain<1>{1})
                return false;
            if (i == 2 && coord != llama::ArrayDomain<1>{2})
                return false;
            i++;
        }

        return true;
    }
    ();
    STATIC_REQUIRE(r);
}

TEST_CASE("ArrayDomainIndexRange3D.destructering")
{
    llama::ArrayDomain<3> ad{1, 1, 1};

    for (auto [x, y, z] : llama::ArrayDomainIndexRange{ad})
    {
        CHECK(x == 0);
        CHECK(y == 0);
        CHECK(z == 0);
    }
}

TEST_CASE("Morton")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    ArrayDomain arrayDomain{2, 3};

    llama::mapping::LinearizeArrayDomainMorton lin;
    CHECK(lin.size(ArrayDomain{2, 3}) == 4 * 4);
    CHECK(lin.size(ArrayDomain{2, 4}) == 4 * 4);
    CHECK(lin.size(ArrayDomain{2, 5}) == 8 * 8);
    CHECK(lin.size(ArrayDomain{8, 8}) == 8 * 8);

    CHECK(lin(ArrayDomain{0, 0}, {}) == 0);
    CHECK(lin(ArrayDomain{0, 1}, {}) == 1);
    CHECK(lin(ArrayDomain{0, 2}, {}) == 4);
    CHECK(lin(ArrayDomain{0, 3}, {}) == 5);
    CHECK(lin(ArrayDomain{1, 0}, {}) == 2);
    CHECK(lin(ArrayDomain{1, 1}, {}) == 3);
    CHECK(lin(ArrayDomain{1, 2}, {}) == 6);
    CHECK(lin(ArrayDomain{1, 3}, {}) == 7);
    CHECK(lin(ArrayDomain{2, 0}, {}) == 8);
    CHECK(lin(ArrayDomain{2, 1}, {}) == 9);
    CHECK(lin(ArrayDomain{2, 2}, {}) == 12);
    CHECK(lin(ArrayDomain{2, 3}, {}) == 13);
    CHECK(lin(ArrayDomain{3, 0}, {}) == 10);
    CHECK(lin(ArrayDomain{3, 1}, {}) == 11);
    CHECK(lin(ArrayDomain{3, 2}, {}) == 14);
    CHECK(lin(ArrayDomain{3, 3}, {}) == 15);
}

TEST_CASE("forEachADCoord_1D")
{
    llama::ArrayDomain<1> adSize{3};

    std::vector<llama::ArrayDomain<1>> coords;
    llama::forEachADCoord(adSize, [&](llama::ArrayDomain<1> coord) { coords.push_back(coord); });

    CHECK(coords == std::vector<llama::ArrayDomain<1>>{{0}, {1}, {2}});
}

TEST_CASE("forEachADCoord_2D")
{
    llama::ArrayDomain<2> adSize{3, 3};

    std::vector<llama::ArrayDomain<2>> coords;
    llama::forEachADCoord(adSize, [&](llama::ArrayDomain<2> coord) { coords.push_back(coord); });

    CHECK(
        coords
        == std::vector<llama::ArrayDomain<2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEST_CASE("forEachADCoord_3D")
{
    llama::ArrayDomain<3> adSize{3, 3, 3};

    std::vector<llama::ArrayDomain<3>> coords;
    llama::forEachADCoord(adSize, [&](llama::ArrayDomain<3> coord) { coords.push_back(coord); });

    CHECK(
        coords
        == std::vector<llama::ArrayDomain<3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

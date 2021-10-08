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

#include "common.hpp"

TEMPLATE_LIST_TEST_CASE("dyn", "", SizeTypes)
{
    STATIC_REQUIRE(static_cast<TestType>(-1) == llama::dyn);
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.CTAD", "", SizeTypes)
{
    TestType one = 1;

    [[maybe_unused]] const llama::ArrayExtents ad0{};
    const llama::ArrayExtents ad1{one};
    const llama::ArrayExtents ad2{one, one};
    const llama::ArrayExtents ad3{one, one, one};

    STATIC_REQUIRE(std::is_same_v<decltype(ad0), const llama::ArrayExtents<std::size_t>>);
    STATIC_REQUIRE(std::is_same_v<decltype(ad1), const llama::ArrayExtents<TestType, llama::dyn>>);
    STATIC_REQUIRE(std::is_same_v<decltype(ad2), const llama::ArrayExtents<TestType, llama::dyn, llama::dyn>>);
    STATIC_REQUIRE(
        std::is_same_v<decltype(ad3), const llama::ArrayExtents<TestType, llama::dyn, llama::dyn, llama::dyn>>);

    STATIC_REQUIRE(decltype(ad0)::rank == 0);
    STATIC_REQUIRE(decltype(ad1)::rank == 1);
    STATIC_REQUIRE(decltype(ad2)::rank == 2);
    STATIC_REQUIRE(decltype(ad3)::rank == 3);
}

TEST_CASE("ArrayExtents.dim0")
{
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x1 = view(llama::ArrayIndex{})(tag::Pos{}, tag::X{});
    double& x2 = view()(tag::Pos{}, tag::X{});
    x1 = 0;
    x2 = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim1.dynamic", "", SizeTypes)
{
    const TestType n = 16;
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{n}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim1.static", "", SizeTypes)
{
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<TestType, 16>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim2.dynamic", "", SizeTypes)
{
    const TestType n = 16;
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{n, n}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i, i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim2.static", "", SizeTypes)
{
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<TestType, 16, 16>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i, i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim3.dynamic", "", SizeTypes)
{
    const TestType n = 16;
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{n, n, n}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i, i, i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim3.static", "", SizeTypes)
{
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<TestType, 16, 16, 16>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i, i, i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim10.dynamic", "", SizeTypes)
{
    const TestType n = 2;
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents{n, n, n, n, n, n, n, n, n, n}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i, i, i, i, i, i, i, i, i, i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.dim10.static", "", SizeTypes)
{
    const TestType i = 1;
    auto mapping = llama::mapping::SoA{llama::ArrayExtents<TestType, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2>{}, Particle{}};
    auto view = llama::allocView(mapping);
    double& x = view(llama::ArrayIndex{i, i, i, i, i, i, i, i, i, i})(tag::Pos{}, tag::X{});
    x = 0;
}

TEST_CASE("ArrayExtentsDynamic")
{
    STATIC_REQUIRE(std::is_same_v<
                   llama::ArrayExtentsDynamic<std::size_t, 3>,
                   llama::ArrayExtents<std::size_t, llama::dyn, llama::dyn, llama::dyn>>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::ArrayExtentsDynamic<int, 3>,
                   llama::ArrayExtents<int, llama::dyn, llama::dyn, llama::dyn>>);
}

TEST_CASE("ArrayExtentsStatic")
{
    STATIC_REQUIRE(
        std::is_same_v<llama::ArrayExtentsNCube<unsigned, 3, 8>, llama::ArrayExtents<unsigned, 8u, 8u, 8u>>);
    STATIC_REQUIRE(std::is_same_v<
                   llama::ArrayExtentsNCube<unsigned long long, 2, 8>,
                   llama::ArrayExtents<unsigned long long, 8ull, 8ull>>);
    STATIC_REQUIRE(
        std::is_same_v<llama::ArrayExtentsNCube<char, 4, 'A'>, llama::ArrayExtents<char, 'A', 'A', 'A', 'A'>>);
}

TEST_CASE("ArrayExtents.ctor_value_init")
{
    [[maybe_unused]] const llama::ArrayExtentsDynamic<std::size_t, 1> extents{};
    CHECK(extents[0] == 0);
}

TEMPLATE_LIST_TEST_CASE("ArrayExtents.toArray", "", SizeTypes)
{
    CHECK(llama::ArrayExtents<TestType>{}.toArray() == llama::Array<TestType, 0>{});

    // dynamic
    CHECK(llama::ArrayExtents<TestType, llama::dyn>{42}.toArray() == llama::Array<TestType, 1>{42});
    CHECK(
        llama::ArrayExtents<TestType, llama::dyn, llama::dyn>{42, 43}.toArray() == llama::Array<TestType, 2>{42, 43});
    CHECK(
        llama::ArrayExtents<TestType, llama::dyn, llama::dyn, llama::dyn>{42, 43, 44}.toArray()
        == llama::Array<TestType, 3>{42, 43, 44});

    // static
    CHECK(llama::ArrayExtents<TestType, 42>{}.toArray() == llama::Array<TestType, 1>{42});
    CHECK(llama::ArrayExtents<TestType, 42, 43>{}.toArray() == llama::Array<TestType, 2>{42, 43});
    CHECK(
        llama::ArrayExtents<TestType, llama::dyn, 43, llama::dyn>{42, 44}.toArray()
        == llama::Array<TestType, 3>{42, 43, 44});

    // mixed
    CHECK(
        llama::ArrayExtents<TestType, 42, llama::dyn, llama::dyn>{43, 44}.toArray()
        == llama::Array<TestType, 3>{42, 43, 44});
    CHECK(
        llama::ArrayExtents<TestType, llama::dyn, 43, llama::dyn>{42, 44}.toArray()
        == llama::Array<TestType, 3>{42, 43, 44});
    CHECK(llama::ArrayExtents<TestType, llama::dyn, 43, 44>{42}.toArray() == llama::Array<TestType, 3>{42, 43, 44});
}

TEMPLATE_LIST_TEST_CASE("forEachADCoord_1D", "", SizeTypes)
{
    const auto n = TestType{3};
    const llama::ArrayExtents extents{n};

    std::vector<llama::ArrayIndex<TestType, 1>> indices;
    llama::forEachADCoord(extents, [&](llama::ArrayIndex<TestType, 1> ai) { indices.push_back(ai); });

    CHECK(indices == std::vector<llama::ArrayIndex<TestType, 1>>{{0}, {1}, {2}});
}

TEMPLATE_LIST_TEST_CASE("forEachADCoord_2D", "", SizeTypes)
{
    const auto n = TestType{3};
    const llama::ArrayExtents extents{n, n};

    std::vector<llama::ArrayIndex<TestType, 2>> indices;
    llama::forEachADCoord(extents, [&](llama::ArrayIndex<TestType, 2> ai) { indices.push_back(ai); });

    CHECK(
        indices
        == std::vector<
            llama::ArrayIndex<TestType, 2>>{{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}});
}

TEMPLATE_LIST_TEST_CASE("forEachADCoord_3D", "", SizeTypes)
{
    const auto n = TestType{3};
    const llama::ArrayExtents extents{n, n, n};

    std::vector<llama::ArrayIndex<TestType, 3>> indices;
    llama::forEachADCoord(extents, [&](llama::ArrayIndex<TestType, 3> ai) { indices.push_back(ai); });

    CHECK(
        indices
        == std::vector<llama::ArrayIndex<TestType, 3>>{
            {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}, {0, 2, 0}, {0, 2, 1}, {0, 2, 2},
            {1, 0, 0}, {1, 0, 1}, {1, 0, 2}, {1, 1, 0}, {1, 1, 1}, {1, 1, 2}, {1, 2, 0}, {1, 2, 1}, {1, 2, 2},
            {2, 0, 0}, {2, 0, 1}, {2, 0, 2}, {2, 1, 0}, {2, 1, 1}, {2, 1, 2}, {2, 2, 0}, {2, 2, 1}, {2, 2, 2},
        });
}

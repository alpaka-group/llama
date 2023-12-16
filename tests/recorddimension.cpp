// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

#include <array>
#include <atomic>
#include <complex>
#include <vector>

// nvc++ 22.7 gets TERMINATED by signal 11 in this file
#ifndef __NVCOMPILER

namespace
{
    struct Tag
    {
    };
} // namespace

TEST_CASE("recorddim.record_with_int")
{
    using RecordDim = llama::Record<llama::Field<Tag, int>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    int& e = view(0u)(Tag{});
    e = 0;
}

TEST_CASE("recorddim.record_with_int[3]")
{
    using namespace llama::literals;

    using RecordDim = llama::Record<llama::Field<Tag, int[3]>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    [[maybe_unused]] const int& e0 = view(0u)(Tag{})(0_RC);
    [[maybe_unused]] const int& e1 = view(0u)(Tag{})(1_RC);
    [[maybe_unused]] const int& e2 = view(0u)(Tag{})(2_RC);
}

TEST_CASE("recorddim.record_with_std::complex<float>")
{
    using RecordDim = llama::Record<llama::Field<Tag, std::complex<float>>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    std::complex<float>& e = view(0u)(Tag{});
    e = {2, 3};
}

TEST_CASE("recorddim.record_with_std::array<float, 4>")
{
    using RecordDim = llama::Record<llama::Field<Tag, std::array<float, 4>>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    std::array<float, 4>& e = view(0u)(Tag{});
    e = {2, 3, 4, 5};
}

// FIXME(bgruber): LLAMA does not handle destructors yet
// TEST_CASE("recorddim.record_with_std::vector<float>")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, std::vector<float>>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});
//
//    std::vector<float>& e = view(0u)(Tag{});
//    e = {2, 3, 4, 5};
//}

TEST_CASE("recorddim.record_with_std::atomic<int>")
{
    using RecordDim = llama::Record<llama::Field<Tag, std::atomic<int>>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    std::atomic<int>& e = view(0u)(Tag{});
    e++;
}

TEST_CASE("recorddim.record_with_noncopyable")
{
    struct Element
    {
        Element() = default;
        Element(const Element&) = delete;
        auto operator=(const Element&) -> Element& = delete;
        Element(Element&&) noexcept = default;
        auto operator=(Element&&) noexcept -> Element& = default;
        ~Element() = default;

        int value;
    };

    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    Element& e = view(0u)(Tag{});
    e.value = 0;
}

TEST_CASE("recorddim.record_with_nonmoveable")
{
    struct Element
    {
        Element() = default;
        Element(const Element&) = delete;
        auto operator=(const Element&) -> Element& = delete;
        Element(Element&&) noexcept = delete;
        auto operator=(Element&&) noexcept -> Element& = delete;
        ~Element() = default;

        int value;
    };

    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    Element& e = view(0u)(Tag{});
    e.value = 0;
}

TEST_CASE("recorddim.record_with_nondefaultconstructible")
{
    struct Element
    {
        Element() = delete;
        int value;
    };

    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
    auto view = llama::allocViewUninitialized(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    Element& e = view(0u)(Tag{});
    e.value = 0;
}

namespace
{
    struct ElementWithCtor
    {
        int value = 42;
    };
} // namespace

TEST_CASE("recorddim.record_with_nontrivial_ctor")
{
    using RecordDim = llama::Record<llama::Field<Tag, ElementWithCtor>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    const ElementWithCtor& e = view(0u)(Tag{});
    CHECK(e.value == 42);
}

#    ifndef __NVCC__
// we get this otherwise: error: identifier "_NV_ANON_NAMESPACE::UniqueInt::counter" is undefined in device code
namespace
{
    struct UniqueInt
    {
        int value = counter++;

    private:
        inline static int counter = 0;
    };
} // namespace

TEST_CASE("recorddim.record_with_nontrivial_ctor2")
{
    using RecordDim = llama::Record<llama::Field<Tag, UniqueInt>>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16}, RecordDim{}});

    CHECK(view(llama::ArrayIndex{0})(Tag{}).value == 0);
    CHECK(view(llama::ArrayIndex{1})(Tag{}).value == 1);
    CHECK(view(llama::ArrayIndex{2})(Tag{}).value == 2);
    CHECK(view(llama::ArrayIndex{15})(Tag{}).value == 15);
}
#    endif

TEST_CASE("recorddim.uninitialized_trivial")
{
    using RecordDim = int;
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{256}, RecordDim{}};
    auto view = llama::allocViewUninitialized(
        mapping,
        [](auto /*alignment*/, std::size_t size) { return std::vector(size, std::byte{0xAA}); });

    for(auto i = 0u; i < 256u; i++)
        CHECK(view(i) == static_cast<int>(0xAAAAAAAA));
}

TEST_CASE("recorddim.uninitialized_ctor.constructFields")
{
    auto mapping = llama::mapping::AoS{llama::ArrayExtents{256}, ElementWithCtor{}};
    auto view = llama::allocViewUninitialized(
        mapping,
        [](auto /*alignment*/, std::size_t size) { return std::vector(size, std::byte{0xAA}); });

    for(auto i = 0u; i < 256u; i++)
        CHECK(view(i).value == static_cast<int>(0xAAAAAAAA)); // ctor has not run

    llama::constructFields(view); // run ctors

    for(auto i = 0u; i < 256u; i++)
        CHECK(view(i).value == 42);
}

TEST_CASE("recorddim.int")
{
    using RecordDim = int;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{16}, RecordDim{}});

    STATIC_REQUIRE(std::is_same_v<decltype(view(0u)), int&>);
    view(0u) = 42;
    CHECK(view(0u) == 42);

    STATIC_REQUIRE(std::is_same_v<decltype(view[0u]), int&>);
    view[0u] = 42;
    CHECK(view[0u] == 42);

    STATIC_REQUIRE(std::is_same_v<decltype(view[llama::ArrayIndex{0}]), int&>);
    view[llama::ArrayIndex{0}] = 42;
    CHECK(view[llama::ArrayIndex{0}] == 42);

    iotaFillView(view);
    iotaCheckView(view);
}

TEST_CASE("recorddim.int[3]")
{
    using namespace llama::literals;

    using RecordDim = int[3];
    auto view
        = llama::allocView(llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, RecordDim>{llama::ArrayExtents{1}});

    view(0u)(0_RC) = 42;
    view(0u)(1_RC) = 43;
    view(0u)(2_RC) = 44;
}

// if we lift the array size higher, we hit the limit on template instantiations
TEST_CASE("recorddim.int[200]")
{
    using namespace llama::literals;

    using RecordDim = int[200];
    auto view
        = llama::allocView(llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, RecordDim>{llama::ArrayExtents{1}});

    view(0u)(0_RC) = 42;
    view(0u)(199_RC) = 43;
}

TEST_CASE("recorddim.int[3][2]")
{
    using namespace llama::literals;

    using RecordDim = int[3][2];
    auto view
        = llama::allocView(llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, RecordDim>{llama::ArrayExtents{1}});

    view(0u)(0_RC)(0_RC) = 42;
    view(0u)(0_RC)(1_RC) = 43;
    view(0u)(1_RC)(0_RC) = 44;
    view(0u)(1_RC)(1_RC) = 45;
    view(0u)(2_RC)(0_RC) = 46;
    view(0u)(2_RC)(1_RC) = 47;
}

TEST_CASE("recorddim.int[1][1][1][1][1][1][1][1][1][1]")
{
    using namespace llama::literals;

    using RecordDim = int[1][1][1][1][1][1][1][1][1][1];
    auto view
        = llama::allocView(llama::mapping::AoS<llama::ArrayExtentsDynamic<int, 1>, RecordDim>{llama::ArrayExtents{1}});

    view(0u)(0_RC)(0_RC)(0_RC)(0_RC)(0_RC)(0_RC)(0_RC)(0_RC)(0_RC)(0_RC) = 42;
}

// clang-format off
struct A1{};
struct A2{};
struct A3{};

using Arrays = llama::Record<
    llama::Field<A1, int[3]>,
    llama::Field<A2, llama::Record<
        llama::Field<Tag, float>
    >[3]>,
    llama::Field<A3, int[2][2]>
>;
// clang-format on

TEST_CASE("recorddim.record_with_arrays")
{
    using namespace llama::literals;

    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, Arrays{}});
    view(0u)(A1{}, 0_RC);
    view(0u)(A1{}, 1_RC);
    view(0u)(A1{}, 2_RC);
    view(0u)(A2{}, 0_RC, Tag{});
    view(0u)(A2{}, 1_RC, Tag{});
    view(0u)(A2{}, 2_RC, Tag{});
    view(0u)(A3{}, 0_RC, 0_RC);
    view(0u)(A3{}, 0_RC, 1_RC);
    view(0u)(A3{}, 1_RC, 0_RC);
    view(0u)(A3{}, 1_RC, 1_RC);
}

TEST_CASE("recorddim.recurring_tags")
{
    using RecordDim = llama::Record<
        llama::Field<tag::X, float>,
        llama::Field<tag::Y, llama::Record<llama::Field<tag::X, float>>>,
        llama::Field<tag::Z, llama::Record<llama::Field<tag::X, llama::Record<llama::Field<tag::X, float>>>>>>;

    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});
    view(0)(tag::X{}) = 42;
    view(0)(tag::Y{}, tag::X{}) = 42;
    view(0)(tag::Z{}, tag::X{}, tag::X{}) = 42;
}
#endif

#ifdef LLAMA_HAS_STRING_FIELDS
TEST_CASE("recorddim.NamedFields")
{
    using RecordDim = llama::Record<
        llama::NamedField<"x", int>,
        llama::NamedField<"y", int>,
        llama::NamedField<"pos", Vec2F>,
        llama::Field<tag::Vel, llama::Record<llama::Field<tag::X, float>, llama::NamedField<"y", float>>>>;

    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    using namespace llama::literals;
    view(0)("x"_Name) = 1;
    view(0)("y"_Name) = 2;
    view(0)("pos"_Name, tag::X{}) = 3;
    view(0)("pos"_Name, tag::Y{}) = 4;
    view(0)(tag::Vel{}, tag::X{}) = 5;
    view(0)(tag::Vel{}, "y"_Name) = 6;

    CHECK(view(0)(llama::RecordCoord<0>{}) == 1);
    CHECK(view(0)(llama::RecordCoord<1>{}) == 2);
    CHECK(view(0)(llama::RecordCoord<2, 0>{}) == 3);
    CHECK(view(0)(llama::RecordCoord<2, 1>{}) == 4);
    CHECK(view(0)(llama::RecordCoord<3, 0>{}) == 5);
    CHECK(view(0)(llama::RecordCoord<3, 1>{}) == 6);
}
#endif

#ifdef LLAMA_CAN_REFLECT_RECORD_DIM
#    include <boost/describe/class.hpp>
namespace
{
    struct MyVel
    {
        int x;
        int y;
    };
    BOOST_DESCRIBE_STRUCT(MyVel, (), (x, y));

    struct MyStruct
    {
        int a;
        int b;
        MyVel pos;
    };
    BOOST_DESCRIBE_STRUCT(MyStruct, (), (a, b, pos));
} // namespace


TEST_CASE("recorddim.Boost.Describe")
{
    using RecordDim = llama::ReflectToRecordDim<MyStruct>;
    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayExtents{1}, RecordDim{}});

    using namespace llama::literals;

    view(0)("a"_Name) = 1;
    view(0)("b"_Name) = 2;
    view(0)("pos"_Name, "x"_Name) = 3;
    view(0)("pos"_Name, "y"_Name) = 4;

    CHECK(view(0)(llama::RecordCoord<0>{}) == 1);
    CHECK(view(0)(llama::RecordCoord<1>{}) == 2);
    CHECK(view(0)(llama::RecordCoord<2, 0>{}) == 3);
    CHECK(view(0)(llama::RecordCoord<2, 1>{}) == 4);

    view(0).at<"a">() = 5;
    view(0).at<"b">() = 6;
    view(0).at<"pos">().at<"x">() = 7;
    view(0).at<"pos">().at<"y">() = 8;

    CHECK(view(0)(llama::RecordCoord<0>{}) == 5);
    CHECK(view(0)(llama::RecordCoord<1>{}) == 6);
    CHECK(view(0)(llama::RecordCoord<2, 0>{}) == 7);
    CHECK(view(0)(llama::RecordCoord<2, 1>{}) == 8);
}
#endif

TEST_CASE("recorddim.forEachLeaf")
{
    {
        std::stringstream ss;
        llama::forEachLeafCoord<Vec3I>([&](auto rc) { ss << rc << ","; });
        CHECK(ss.str() == "RecordCoord<0>,RecordCoord<1>,RecordCoord<2>,");
    }
    {
        std::stringstream ss;
        llama::forEachLeafCoord<Particle>([&](auto rc) { ss << rc << ","; });
        CHECK(
            ss.str()
            == "RecordCoord<0, 0>,RecordCoord<0, 1>,RecordCoord<0, 2>,RecordCoord<1>,RecordCoord<2, 0>,RecordCoord<2, "
               "1>,RecordCoord<2, 2>,RecordCoord<3, 0>,RecordCoord<3, 1>,RecordCoord<3, 2>,RecordCoord<3, 3>,");
    }
    {
        std::stringstream ss;
        llama::forEachLeafCoord<int>([&](auto rc) { ss << rc << ","; });
        CHECK(ss.str() == "RecordCoord<>,");
    }
}

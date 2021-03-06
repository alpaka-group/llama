#include "common.hpp"

#include <array>
#include <atomic>
#include <catch2/catch.hpp>
#include <complex>
#include <llama/llama.hpp>
#include <vector>

namespace
{
    struct Tag
    {
    };
} // namespace

TEST_CASE("recorddim.record_with_int")
{
    using RecordDim = llama::Record<llama::Field<Tag, int>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    int& e = view(0u)(Tag{});
    e = 0;
}

TEST_CASE("recorddim.record_with_int[3]")
{
    using namespace llama::literals;

    using RecordDim = llama::Record<llama::Field<Tag, int[3]>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    int& e0 = view(0u)(Tag{})(0_RC);
    int& e1 = view(0u)(Tag{})(1_RC);
    int& e2 = view(0u)(Tag{})(2_RC);
}

// TEST_CASE("recorddim.record_with_std::complex<float>")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, std::complex<float>>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    std::complex<float>& e = view(0u)(Tag{});
//    e = {2, 3};
//}

TEST_CASE("recorddim.record_with_std::array<float, 4>")
{
    using RecordDim = llama::Record<llama::Field<Tag, std::array<float, 4>>>;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    std::array<float, 4>& e = view(0u)(Tag{});
    e = {2, 3, 4, 5};
}

// TEST_CASE("recorddim.record_with_std::vector<float>")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, std::vector<float>>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    std::vector<float>& e = view(0u)(Tag{});
//    // e = {2, 3, 4, 5}; // FIXME: LLAMA memory is uninitialized
//}

// TEST_CASE("recorddim.record_with_std::atomic<int>")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, std::atomic<int>>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    std::atomic<int>& e = view(0u)(Tag{});
//    // e++; // FIXME: LLAMA memory is uninitialized
//}

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
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

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
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    Element& e = view(0u)(Tag{});
    e.value = 0;
}

// TEST_CASE("recorddim.record_with_nondefaultconstructible")
//{
//    struct Element
//    {
//        Element() = delete;
//        int value;
//    };
//
//    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    Element& e = view(0u)(Tag{});
//    e.value = 0;
//}

// TEST_CASE("recorddim.record_with_nontrivial_ctor")
//{
//    struct Element
//    {
//        int value = 42;
//    };
//
//    using RecordDim = llama::Record<llama::Field<Tag, Element>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    Element& e = view(0u)(Tag{});
//    // CHECK(e.value == 42); // FIXME: LLAMA memory is uninitialized
//}

// namespace
//{
//    struct UniqueInt
//    {
//        int value = counter++;
//
//        explicit operator int() const
//        {
//            return value;
//        }
//
//    private:
//        inline static int counter = 0;
//    };
//} // namespace
//
// TEST_CASE("recorddim.record_with_nontrivial_ctor2")
//{
//    using RecordDim = llama::Record<llama::Field<Tag, UniqueInt>>;
//    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});
//
//    // FIXME: LLAMA memory is uninitialized
//    // CHECK(view(ArrayDims{0})(Tag{}) == 0);
//    // CHECK(view(ArrayDims{1})(Tag{}) == 1);
//    // CHECK(view(ArrayDims{2})(Tag{}) == 2);
//    // CHECK(view(ArrayDims{15})(Tag{}) == 15);
//}

TEST_CASE("recorddim.int")
{
    using RecordDim = int;
    auto view = allocView(llama::mapping::AoS{llama::ArrayDims{1}, RecordDim{}});

    STATIC_REQUIRE(std::is_same_v<decltype(view(0u)), int&>);
    view(0u) = 42;
    CHECK(view(0u) == 42);

    STATIC_REQUIRE(std::is_same_v<decltype(view[0u]), int&>);
    view[0u] = 42;
    CHECK(view[0u] == 42);

    STATIC_REQUIRE(std::is_same_v<decltype(view[llama::ArrayDims{0}]), int&>);
    view[llama::ArrayDims{0}] = 42;
    CHECK(view[llama::ArrayDims{0}] == 42);
}

TEST_CASE("recorddim.int[3]")
{
    using namespace llama::literals;

    using RecordDim = int[3];
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

    view(0u)(0_RC) = 42;
    view(0u)(1_RC) = 43;
    view(0u)(2_RC) = 44;
}

// if we lift the array size higher, we hit the limit on template instantiations
TEST_CASE("recorddim.int[200]")
{
    using namespace llama::literals;

    using RecordDim = int[200];
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

    view(0u)(0_RC) = 42;
    view(0u)(199_RC) = 43;
}

TEST_CASE("recorddim.int[3][2]")
{
    using namespace llama::literals;

    using RecordDim = int[3][2];
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

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
    auto view = allocView(llama::mapping::AoS<llama::ArrayDims<1>, RecordDim>{llama::ArrayDims{1}});

    view(0u)(0_RC)(0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) (0_RC) = 42;
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

    auto view = llama::allocView(llama::mapping::AoS{llama::ArrayDims{1}, Arrays{}});
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

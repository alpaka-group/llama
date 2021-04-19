#include "common.h"

#include <array>
#include <atomic>
#include <catch2/catch.hpp>
#include <complex>
#include <llama/llama.hpp>
#include <vector>

TEST_CASE("type int")
{
    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, int>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    int& e = view(ArrayDims{0})(Tag{});
    e = 0;
}

TEST_CASE("type std::complex<float>")
{
    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, std::complex<float>>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    std::complex<float>& e = view(ArrayDims{0})(Tag{});
    e = {2, 3};
}

TEST_CASE("type std::array<float, 4>")
{
    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, std::array<float, 4>>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    std::array<float, 4>& e = view(ArrayDims{0})(Tag{});
    e = {2, 3, 4, 5};
}

TEST_CASE("type std::vector<float>")
{
    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, std::vector<float>>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    std::vector<float>& e = view(ArrayDims{0})(Tag{});
    // e = {2, 3, 4, 5}; // FIXME: LLAMA memory is uninitialized
}

TEST_CASE("type std::atomic<int>")
{
    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, std::atomic<int>>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    std::atomic<int>& e = view(ArrayDims{0})(Tag{});
    // e++; // FIXME: LLAMA memory is uninitialized
}

TEST_CASE("type noncopyable")
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

    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, Element>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    Element& e = view(ArrayDims{0})(Tag{});
    e.value = 0;
}

TEST_CASE("type nonmoveable")
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

    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, Element>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    Element& e = view(ArrayDims{0})(Tag{});
    e.value = 0;
}

TEST_CASE("type not defaultconstructible")
{
    struct Element
    {
        Element() = delete;
        int value;
    };

    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, Element>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    Element& e = view(ArrayDims{0})(Tag{});
    e.value = 0;
}

TEST_CASE("type nottrivial ctor")
{
    struct Element
    {
        int value = 42;
    };

    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, Element>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    Element& e = view(ArrayDims{0})(Tag{});
    // CHECK(e.value == 42); // FIXME: LLAMA memory is uninitialized
}

namespace
{
    struct UniqueInt
    {
        int value = counter++;

        explicit operator int() const
        {
            return value;
        }

    private:
        inline static int counter = 0;
    };
} // namespace

TEST_CASE("type custom initialization")
{
    struct Tag
    {
    };
    using Name = llama::Record<llama::Field<Tag, UniqueInt>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, Name>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    // FIXME: LLAMA memory is uninitialized
    // CHECK(view(ArrayDims{0})(Tag{}) == 0);
    // CHECK(view(ArrayDims{1})(Tag{}) == 1);
    // CHECK(view(ArrayDims{2})(Tag{}) == 2);
    // CHECK(view(ArrayDims{15})(Tag{}) == 15);
}

TEST_CASE("type just double")
{
    using RecordDim = double;
    llama::ArrayDims arrayDims{16};
    llama::mapping::SoA mapping{arrayDims, RecordDim{}};
    auto view = allocView(mapping);

    STATIC_REQUIRE(std::is_same_v<decltype(view(0u)), double&>);
    view(0u) = 42.0;
    CHECK(view(0u) == 42.0);

    STATIC_REQUIRE(std::is_same_v<decltype(view[0u]), double&>);
    view[0u] = 42.0;
    CHECK(view[0u] == 42.0);

    STATIC_REQUIRE(std::is_same_v<decltype(view[llama::ArrayDims{0}]), double&>);
    view[llama::ArrayDims{0}] = 42.0;
    CHECK(view[llama::ArrayDims{0}] == 42.0);
}

TEST_CASE("static array")
{
    using namespace llama::literals;

    struct Tag
    {
    };
    using RecordDim = llama::Record<llama::Field<Tag, int[3]>>;

    using ArrayDims = llama::ArrayDims<1>;
    ArrayDims arrayDims{16};

    using Mapping = llama::mapping::SoA<ArrayDims, RecordDim>;
    Mapping mapping{arrayDims};
    auto view = allocView(mapping);

    int& e0 = view(ArrayDims{0})(Tag{})(0_RC);
    int& e1 = view(ArrayDims{0})(Tag{})(1_RC);
    int& e2 = view(ArrayDims{0})(Tag{})(2_RC);
}

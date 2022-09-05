#include "common.hpp"

#include <catch2/catch_approx.hpp>
#include <numeric>

TEST_CASE("computedprop")
{
    auto extents = llama::ArrayExtentsDynamic<std::size_t, 1>{10};
    auto mapping = TriangleAoSWithComputedNormal<decltype(extents), Triangle>{extents};

    STATIC_REQUIRE(mapping.blobCount == 1);
    CHECK(mapping.blobSize(0) == sizeof(double) * 10 * 12);

    auto view = llama::allocViewUninitialized(mapping);

    using namespace tag;
    view(5u)(A{}, X{}) = 0.0f;
    view(5u)(A{}, Y{}) = 0.0f;
    view(5u)(A{}, Z{}) = 0.0f;
    view(5u)(B{}, X{}) = 5.0f;
    view(5u)(B{}, Y{}) = 0.0f;
    view(5u)(B{}, Z{}) = 0.0f;
    view(5u)(C{}, X{}) = 0.0f;
    view(5u)(C{}, Y{}) = 3.0f;
    view(5u)(C{}, Z{}) = 0.0f;
    const auto nx = view(5u)(Normal{}, X{});
    const auto ny = view(5u)(Normal{}, Y{});
    const auto nz = view(5u)(Normal{}, Z{});
    CHECK(nx == Catch::Approx(0.0f));
    CHECK(ny == Catch::Approx(0.0f));
    CHECK(nz == Catch::Approx(1.0f));
}

namespace
{
    // Maps accesses to the product of the ArrayIndex.
    template<typename TArrayExtents, typename TRecordDim>
    struct ComputedMapping
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = 0;

        constexpr ComputedMapping() = default;

        constexpr explicit ComputedMapping(ArrayExtents = {}, RecordDim = {})
        {
        }

        constexpr auto extents() const
        {
            return ArrayExtents{};
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blob>
        constexpr auto compute(ArrayIndex ai, llama::RecordCoord<RecordCoords...>, llama::Array<Blob, blobCount>&)
            const -> std::size_t
        {
            return std::reduce(std::begin(ai), std::end(ai), std::size_t{1}, std::multiplies<>{});
        }
    };
} // namespace

TEST_CASE("fully_computed_mapping")
{
    auto mapping = ComputedMapping<llama::ArrayExtents<std::size_t, 10, 10, 10>, int>{{}};
    auto view = llama::allocViewUninitialized(mapping);

    using namespace tag;
    CHECK(view(0u, 1u, 2u) == 0);
    CHECK(view(2u, 1u, 2u) == 4);
    CHECK(view(2u, 5u, 2u) == 20);
}

namespace
{
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename LinearizeArrayDimsFunctor = llama::mapping::LinearizeArrayDimsCpp>
    struct CompressedBoolMapping : TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        constexpr CompressedBoolMapping() = default;

        constexpr explicit CompressedBoolMapping(ArrayExtents extents = {}) : ArrayExtents(extents)
        {
        }

        constexpr auto extents() const -> ArrayExtents
        {
            return ArrayExtents{*this};
        }

        using Word = std::uint64_t;

        struct BoolRef
        {
            Word& word;
            unsigned char bit;

            operator bool() const // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
            {
                return (word & (Word{1} << bit)) != 0;
            }

            auto operator=(bool b) -> BoolRef&
            {
                word ^= (-static_cast<Word>(b) ^ word) & (Word{1} << bit);
                return *this;
            }
        };

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            llama::forEachLeafCoord<RecordDim>([](auto rc) constexpr {
                static_assert(std::is_same_v<llama::GetType<RecordDim, decltype(rc)>, bool>);
            });
            constexpr std::size_t wordBytes = sizeof(Word);
            return (LinearizeArrayDimsFunctor{}.size(extents()) + wordBytes - 1) / wordBytes;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blob>
        constexpr auto compute(
            ArrayIndex ai,
            llama::RecordCoord<RecordCoords...>,
            llama::Array<Blob, blobCount>& blobs) const -> BoolRef
        {
            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents());
            const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;

            constexpr std::size_t wordBits = sizeof(Word) * CHAR_BIT;
            return BoolRef{
                reinterpret_cast<Word&>(blobs[blob][bitOffset / wordBits]),
                static_cast<unsigned char>(bitOffset % wordBits)};
        }
    };

    // clang-format off
    using BoolRecord = llama::Record<
        llama::Field<tag::A, bool>,
        llama::Field<tag::B, llama::Record<
            llama::Field<tag::X, bool>,
            llama::Field<tag::Y, bool>
        >>
    >;
    // clang-format on
} // namespace

TEST_CASE("compressed_bools")
{
    auto mapping = CompressedBoolMapping<llama::ArrayExtents<std::size_t, 8, 8>, BoolRecord>{{}};
    STATIC_REQUIRE(decltype(mapping)::blobCount == 3);
    CHECK(mapping.blobSize(0) == 8);
    CHECK(mapping.blobSize(1) == 8);
    CHECK(mapping.blobSize(2) == 8);

    auto view = llama::allocViewUninitialized(mapping);
    for(auto y = 0u; y < 8; y++)
    {
        for(auto x = 0u; x < 8; x++)
        {
            view(y, x)(tag::A{}) = static_cast<bool>(x * y & 1u);
            view(y, x)(tag::B{}, tag::X{}) = static_cast<bool>(x & 1u);
            view(y, x)(tag::B{}, tag::Y{}) = static_cast<bool>(y & 1u);
        }
    }

    for(auto y = 0u; y < 8; y++)
    {
        for(auto x = 0u; x < 8; x++)
        {
            CHECK(view(y, x)(tag::A{}) == static_cast<bool>(x * y & 1u));
            CHECK(view(y, x)(tag::B{}, tag::X{}) == static_cast<bool>(x & 1u));
            CHECK(view(y, x)(tag::B{}, tag::Y{}) == static_cast<bool>(y & 1u));
        }
    }
}

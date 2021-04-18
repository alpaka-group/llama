#include "common.h"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>
#include <numeric>

// clang-format off
namespace tag {
    struct X {};
    struct Y {};
    struct Z {};
    struct A {};
    struct B {};
    struct C {};
    struct Normal {};
}

using Vec3 = llama::DS<
    llama::DE<tag::X, double>,
    llama::DE<tag::Y, double>,
    llama::DE<tag::Z, double>
>;
using Triangle = llama::DS<
    llama::DE<tag::A, Vec3>,
    llama::DE<tag::B, Vec3>,
    llama::DE<tag::C, Vec3>,
    llama::DE<tag::Normal, Vec3>
>;
// clang-format on

namespace
{
    template <typename ArrayDomain, typename DatumDomain>
    struct AoSWithComputedNormal : llama::mapping::PackedAoS<ArrayDomain, DatumDomain>
    {
        using Base = llama::mapping::PackedAoS<ArrayDomain, DatumDomain>;

        template <std::size_t... DatumDomainCoord>
        static constexpr auto isComputed(llama::DatumCoord<DatumDomainCoord...>)
        {
            return llama::DatumCoordCommonPrefixIsSame<llama::DatumCoord<DatumDomainCoord...>, llama::DatumCoord<3>>;
        }

        template <std::size_t... DatumDomainCoord, typename Blob>
        constexpr auto compute(
            ArrayDomain coord,
            llama::DatumCoord<DatumDomainCoord...>,
            llama::Array<Blob, Base::blobCount>& storageBlobs) const
        {
            auto fetch = [&](llama::NrAndOffset nrAndOffset) -> double {
                return *reinterpret_cast<double*>(&storageBlobs[nrAndOffset.nr][nrAndOffset.offset]);
            };

            const auto ax = fetch(Base::template blobNrAndOffset<0, 0>(coord));
            const auto ay = fetch(Base::template blobNrAndOffset<0, 1>(coord));
            const auto az = fetch(Base::template blobNrAndOffset<0, 2>(coord));
            const auto bx = fetch(Base::template blobNrAndOffset<1, 0>(coord));
            const auto by = fetch(Base::template blobNrAndOffset<1, 1>(coord));
            const auto bz = fetch(Base::template blobNrAndOffset<1, 2>(coord));
            const auto cx = fetch(Base::template blobNrAndOffset<2, 0>(coord));
            const auto cy = fetch(Base::template blobNrAndOffset<2, 1>(coord));
            const auto cz = fetch(Base::template blobNrAndOffset<2, 2>(coord));

            const auto e1x = bx - ax;
            const auto e1y = by - ay;
            const auto e1z = bz - az;
            const auto e2x = cx - ax;
            const auto e2y = cy - ay;
            const auto e2z = cz - az;

            const auto crossx = e1y * e2z - e1z * e2y;
            const auto crossy = -(e1x * e2z - e1z * e2x);
            const auto crossz = e1x * e2y - e1y * e2x;

            const auto length = std::sqrt(crossx * crossx + crossy * crossy + crossz * crossz);

            const auto normalx = crossx / length;
            const auto normaly = crossy / length;
            const auto normalz = crossz / length;

            using DC = llama::DatumCoord<DatumDomainCoord...>;
            if constexpr (std::is_same_v<DC, llama::DatumCoord<3, 0>>)
                return normalx;
            if constexpr (std::is_same_v<DC, llama::DatumCoord<3, 1>>)
                return normaly;
            if constexpr (std::is_same_v<DC, llama::DatumCoord<3, 2>>)
                return normalz;
            // if constexpr (std::is_same_v<DC, llama::DatumCoord<3>>)
            //{
            //    llama::One<llama::GetType<DatumDomain, llama::DatumCoord<3>>> normal;
            //    normal(llama::DatumCoord<0>{}) = normalx;
            //    normal(llama::DatumCoord<1>{}) = normaly;
            //    normal(llama::DatumCoord<2>{}) = normalz;
            //    return normal;
            //}
        }
    };
} // namespace

TEST_CASE("computedprop")
{
    auto arrayDomain = llama::ArrayDomain<1>{10};
    auto mapping = AoSWithComputedNormal<decltype(arrayDomain), Triangle>{arrayDomain};

    STATIC_REQUIRE(mapping.blobCount == 1);
    CHECK(mapping.blobSize(0) == 10 * 12 * sizeof(double));

    auto view = llama::allocView(mapping);

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
    CHECK(nx == Approx(0.0f));
    CHECK(ny == Approx(0.0f));
    CHECK(nz == Approx(1.0f));
}

namespace
{
    // Maps accesses to the product of the ArrayDomain coord.
    template <typename T_ArrayDomain, typename T_DatumDomain>
    struct ComputedMapping
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 0;

        constexpr ComputedMapping() = default;

        constexpr ComputedMapping(ArrayDomain, DatumDomain = {})
        {
        }

        template <std::size_t... DatumDomainCoord>
        static constexpr auto isComputed(llama::DatumCoord<DatumDomainCoord...>)
        {
            return true;
        }

        template <std::size_t... DatumDomainCoord, typename Blob>
        constexpr auto compute(
            ArrayDomain coord,
            llama::DatumCoord<DatumDomainCoord...>,
            llama::Array<Blob, blobCount>&) const -> std::size_t
        {
            return std::reduce(std::begin(coord), std::end(coord), std::size_t{1}, std::multiplies<>{});
        }
    };
} // namespace

TEST_CASE("fully_computed_mapping")
{
    auto arrayDomain = llama::ArrayDomain<3>{10, 10, 10};
    auto mapping = ComputedMapping<decltype(arrayDomain), Triangle>{arrayDomain};

    auto view = llama::allocView(mapping);

    using namespace tag;
    CHECK(view(0u, 1u, 2u)(A{}, X{}) == 0);
    CHECK(view(2u, 1u, 2u)(A{}, Y{}) == 4);
    CHECK(view(2u, 5u, 2u)(A{}, Z{}) == 20);
}

namespace
{
    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        typename LinearizeArrayDomainFunctor = llama::mapping::LinearizeArrayDomainCpp>
    struct CompressedBoolMapping
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlattenDatumDomain<DatumDomain>>::value;

        constexpr CompressedBoolMapping() = default;

        constexpr CompressedBoolMapping(ArrayDomain size) : arrayDomainSize(size)
        {
        }

        using Word = std::uint64_t;

        struct BoolRef
        {
            Word& word;
            unsigned char bit;

            operator bool() const
            {
                return word & (Word{1} << bit);
            }

            auto operator=(bool b)
            {
                word ^= (-Word{b} ^ word) & (Word{1} << bit);
            }
        };

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            llama::forEachLeaf<DatumDomain>([](auto coord) constexpr {
                static_assert(std::is_same_v<llama::GetType<DatumDomain, decltype(coord)>, bool>);
            });
            constexpr std::size_t wordBytes = sizeof(Word);
            return (LinearizeArrayDomainFunctor{}.size(arrayDomainSize) + wordBytes - 1) / wordBytes;
        }

        template <std::size_t... DatumDomainCoord>
        static constexpr auto isComputed(llama::DatumCoord<DatumDomainCoord...>)
        {
            return true;
        }

        template <std::size_t... DatumDomainCoord, typename Blob>
        constexpr auto compute(
            ArrayDomain coord,
            llama::DatumCoord<DatumDomainCoord...>,
            llama::Array<Blob, blobCount>& blobs) const -> BoolRef
        {
            const auto bitOffset = LinearizeArrayDomainFunctor{}(coord, arrayDomainSize);
            const auto blob = llama::flatDatumCoord<DatumDomain, llama::DatumCoord<DatumDomainCoord...>>;

            constexpr std::size_t wordBits = sizeof(Word) * CHAR_BIT;
            return BoolRef{
                reinterpret_cast<Word&>(blobs[blob][bitOffset / wordBits]),
                static_cast<unsigned char>(bitOffset % wordBits)};
        }

        ArrayDomain arrayDomainSize;
    };

    // clang-format off
    using BoolDomain = llama::DatumStruct<
        llama::DatumElement<tag::A, bool>,
        llama::DatumElement<tag::B, llama::DatumStruct<
            llama::DatumElement<tag::X, bool>,
            llama::DatumElement<tag::Y, bool>
        >>
    >;
    // clang-format on
} // namespace

TEST_CASE("compressed_bools")
{
    auto arrayDomain = llama::ArrayDomain{8, 8};
    auto mapping = CompressedBoolMapping<decltype(arrayDomain), BoolDomain>{arrayDomain};
    STATIC_REQUIRE(decltype(mapping)::blobCount == 3);
    CHECK(mapping.blobSize(0) == 8);
    CHECK(mapping.blobSize(1) == 8);
    CHECK(mapping.blobSize(2) == 8);

    auto view = llama::allocView(mapping);
    for (auto y = 0u; y < 8; y++)
    {
        for (auto x = 0u; x < 8; x++)
        {
            view(y, x)(tag::A{}) = static_cast<bool>(x * y & 1);
            view(y, x)(tag::B{}, tag::X{}) = static_cast<bool>(x & 1);
            view(y, x)(tag::B{}, tag::Y{}) = static_cast<bool>(y & 1);
        }
    }

    for (auto y = 0u; y < 8; y++)
    {
        for (auto x = 0u; x < 8; x++)
        {
            CHECK(view(y, x)(tag::A{}) == static_cast<bool>(x * y & 1));
            CHECK(view(y, x)(tag::B{}, tag::X{}) == static_cast<bool>(x & 1));
            CHECK(view(y, x)(tag::B{}, tag::Y{}) == static_cast<bool>(y & 1));
        }
    }
}

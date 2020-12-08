#include <catch2/catch.hpp>
#include <llama/Proofs.hpp>
#include <llama/llama.hpp>
#include <numeric>

// clang-format off
namespace tag {
    struct Pos {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Momentum {};
    struct Weight {};
    struct Flags {};
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Weight, float>,
    llama::DE<tag::Momentum, llama::DS<
        llama::DE<tag::X, double>,
        llama::DE<tag::Y, double>,
        llama::DE<tag::Z, double>
    >>,
    llama::DE<tag::Flags, llama::DA<bool, 4>>
>;
// clang-format on

TEST_CASE("mapsNonOverlappingly.AoS")
{
    using ArrayDomain = llama::ArrayDomain<2>;
    constexpr auto arrayDomain = ArrayDomain{32, 32};
    constexpr auto mapping = llama::mapping::AoS<ArrayDomain, Particle>{arrayDomain};

#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

namespace
{
    template <typename T_ArrayDomain, typename T_DatumDomain>
    struct MapEverythingToZero
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr MapEverythingToZero(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        constexpr auto getBlobSize(std::size_t) const -> std::size_t
        {
            return std::reduce(
                       std::begin(arrayDomainSize),
                       std::end(arrayDomainSize),
                       std::size_t{1},
                       std::multiplies{})
                * llama::sizeOf<DatumDomain>;
        }

        template <std::size_t... DDCs>
        constexpr auto getBlobNrAndOffset(ArrayDomain coord) const -> llama::NrAndOffset
        {
            return {0, 0};
        }

        ArrayDomain arrayDomainSize;
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.MapEverythingToZero")
{
#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDomain<1>{1}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDomain<1>{2}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDomain<2>{32, 32}, Particle{}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

namespace
{
    // maps each element of the datum domain into a separate blobs. Each blob stores Modulus elements. If the array
    // domain is larger than Modulus, elements are overwritten.
    template <typename T_ArrayDomain, typename T_DatumDomain, std::size_t Modulus>
    struct ModulusMapping
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlattenDatumDomain<DatumDomain>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr ModulusMapping(ArrayDomain size, DatumDomain = {}) : arrayDomainSize(size)
        {
        }

        constexpr auto getBlobSize(std::size_t) const -> std::size_t
        {
            return Modulus * llama::sizeOf<DatumDomain>;
        }

        template <std::size_t... DDCs>
        constexpr auto getBlobNrAndOffset(ArrayDomain coord) const -> llama::NrAndOffset
        {
            const auto blob = llama::flatDatumCoord<DatumDomain, llama::DatumCoord<DDCs...>>;
            const auto offset = (llama::mapping::LinearizeArrayDomainCpp{}(coord, arrayDomainSize) % Modulus)
                * sizeof(llama::GetType<DatumDomain, llama::DatumCoord<DDCs...>>);
            return {blob, offset};
        }

        ArrayDomain arrayDomainSize;
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.ModulusMapping")
{
    using Modulus10Mapping = ModulusMapping<llama::ArrayDomain<1>, Particle, 10>;

#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDomain<1>{1}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDomain<1>{9}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDomain<1>{10}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDomain<1>{11}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDomain<1>{25}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

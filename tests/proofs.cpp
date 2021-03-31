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
} // namespace tag

using Particle = llama::Record<
    llama::Field<tag::Pos, llama::Record<
        llama::Field<tag::X, double>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, double>
    >>,
    llama::Field<tag::Weight, float>,
    llama::Field<tag::Momentum, llama::Record<
        llama::Field<tag::X, double>,
        llama::Field<tag::Y, double>,
        llama::Field<tag::Z, double>
    >>,
    llama::Field<tag::Flags, bool[4]>
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
    template <typename T_ArrayDomain, typename T_RecordDim>
    struct MapEverythingToZero
    {
        using ArrayDomain = T_ArrayDomain;
        using RecordDim = T_RecordDim;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit MapEverythingToZero(ArrayDomain size, RecordDim = {}) : arrayDomainSize(size)
        {
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return std::reduce(
                       std::begin(arrayDomainSize),
                       std::end(arrayDomainSize),
                       std::size_t{1},
                       std::multiplies{})
                * llama::sizeOf<RecordDim>;
        }

        template <std::size_t... DDCs>
        constexpr auto blobNrAndOffset(ArrayDomain) const -> llama::NrAndOffset
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
    // maps each element of the record dimension into a separate blobs. Each blob stores Modulus elements. If the array
    // domain is larger than Modulus, elements are overwritten.
    template <typename T_ArrayDomain, typename T_RecordDim, std::size_t Modulus>
    struct ModulusMapping
    {
        using ArrayDomain = T_ArrayDomain;
        using RecordDim = T_RecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlattenRecordDim<RecordDim>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit ModulusMapping(ArrayDomain size, RecordDim = {}) : arrayDomainSize(size)
        {
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return Modulus * llama::sizeOf<RecordDim>;
        }

        template <std::size_t... DDCs>
        constexpr auto blobNrAndOffset(ArrayDomain coord) const -> llama::NrAndOffset
        {
            const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<DDCs...>>;
            const auto offset = (llama::mapping::LinearizeArrayDomainCpp{}(coord, arrayDomainSize) % Modulus)
                * sizeof(llama::GetType<RecordDim, llama::RecordCoord<DDCs...>>);
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

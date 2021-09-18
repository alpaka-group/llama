#include <catch2/catch.hpp>
#include <llama/Proofs.hpp>
#include <llama/llama.hpp>

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

TEST_CASE("mapsNonOverlappingly.PackedAoS")
{
    constexpr auto mapping = llama::mapping::PackedAoS<llama::ArrayDims<2>, Particle>{{32, 32}};
#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

TEST_CASE("mapsNonOverlappingly.AlignedAoS")
{
    constexpr auto mapping = llama::mapping::AlignedAoS<llama::ArrayDims<2>, Particle>{{32, 32}};
#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

namespace
{
    template<typename TArrayDims, typename TRecordDim>
    struct MapEverythingToZero
    {
        using ArrayDims = TArrayDims;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit MapEverythingToZero(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return arrayDimsSize;
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            std::size_t arraySize = 1;
            for(auto i = 0; i < ArrayDims::rank; i++)
                arraySize *= arrayDimsSize[i];
            return arraySize * llama::sizeOf<RecordDim>;
        }

        template<std::size_t... DDCs>
        constexpr auto blobNrAndOffset(ArrayDims) const -> llama::NrAndOffset
        {
            return {0, 0};
        }

    private:
        ArrayDims arrayDimsSize;
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.MapEverythingToZero")
{
#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDims<1>{1}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDims<1>{2}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayDims<2>{32, 32}, Particle{}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

namespace
{
    // maps each element of the record dimension into a separate blobs. Each blob stores Modulus elements. If the array
    // dimensions are larger than Modulus, elements are overwritten.
    template<typename TArrayDims, typename TRecordDim, std::size_t Modulus>
    struct ModulusMapping
    {
        using ArrayDims = TArrayDims;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit ModulusMapping(ArrayDims size, RecordDim = {}) : arrayDimsSize(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayDims() const -> ArrayDims
        {
            return arrayDimsSize;
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return Modulus * llama::sizeOf<RecordDim>;
        }

        template<std::size_t... DDCs>
        constexpr auto blobNrAndOffset(ArrayDims coord) const -> llama::NrAndOffset
        {
            const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<DDCs...>>;
            const auto offset = (llama::mapping::LinearizeArrayDimsCpp{}(coord, arrayDimsSize) % Modulus)
                * sizeof(llama::GetType<RecordDim, llama::RecordCoord<DDCs...>>);
            return {blob, offset};
        }

    private:
        ArrayDims arrayDimsSize;
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.ModulusMapping")
{
    using Modulus10Mapping = ModulusMapping<llama::ArrayDims<1>, Particle, 10>;

#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{1}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{9}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{10}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{11}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayDims<1>{25}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

TEST_CASE("maps.ModulusMapping")
{
    constexpr auto arrayDims = llama::ArrayDims{128};
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::AoS{arrayDims, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<8>(llama::mapping::AoS{arrayDims, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<16>(llama::mapping::AoS{arrayDims, Particle{}}));

    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::SoA{arrayDims, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<8>(llama::mapping::SoA{arrayDims, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<16>(llama::mapping::SoA{arrayDims, Particle{}}));

    STATIC_REQUIRE(
        llama::mapsPiecewiseContiguous<1>(llama::mapping::AoSoA<decltype(arrayDims), Particle, 8>{arrayDims}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<8>(
        llama::mapping::AoSoA<decltype(arrayDims), Particle, 8>{arrayDims, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<16>(
        llama::mapping::AoSoA<decltype(arrayDims), Particle, 8>{arrayDims, Particle{}}));
}

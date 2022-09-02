#include "common.hpp"

#include <llama/Proofs.hpp>

#ifndef __NVCOMPILER
TEST_CASE("mapsNonOverlappingly.PackedAoS")
{
#    ifdef __cpp_constexpr_dynamic_alloc
    constexpr auto mapping = llama::mapping::PackedAoS<llama::ArrayExtentsDynamic<std::size_t, 2>, Particle>{{32, 32}};
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#    else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#    endif
}

TEST_CASE("mapsNonOverlappingly.AlignedAoS")
{
#    ifdef __cpp_constexpr_dynamic_alloc
    constexpr auto mapping
        = llama::mapping::AlignedAoS<llama::ArrayExtentsDynamic<std::size_t, 2>, Particle>{{32, 32}};
    STATIC_REQUIRE(llama::mapsNonOverlappingly(mapping));
#    else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#    endif
}
#endif

namespace
{
    template<typename TArrayExtents, typename TRecordDim>
    struct MapEverythingToZero : TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = 1;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit MapEverythingToZero(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> const ArrayExtents&
        {
            return *this;
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return llama::product(extents()) * llama::sizeOf<RecordDim>;
        }

        template<std::size_t... RecordCoords>
        constexpr auto blobNrAndOffset(ArrayIndex, llama::RecordCoord<RecordCoords...> = {}) const
            -> llama::NrAndOffset<std::size_t>
        {
            return {0, 0};
        }
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.MapEverythingToZero")
{
#ifdef __cpp_constexpr_dynamic_alloc
    STATIC_REQUIRE(
        llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayExtentsDynamic<std::size_t, 1>{1}, double{}}));
    STATIC_REQUIRE(
        !llama::mapsNonOverlappingly(MapEverythingToZero{llama::ArrayExtentsDynamic<std::size_t, 1>{2}, double{}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(
        MapEverythingToZero{llama::ArrayExtentsDynamic<std::size_t, 2>{32, 32}, Particle{}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

namespace
{
    // maps each element of the record dimension into a separate blobs. Each blob stores Modulus elements. If the array
    // dimensions are larger than Modulus, elements are overwritten.
    template<typename TArrayExtents, typename TRecordDim, std::size_t Modulus>
    struct ModulusMapping : TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit ModulusMapping(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> const ArrayExtents&
        {
            return *this;
        }

        constexpr auto blobSize(std::size_t) const -> std::size_t
        {
            return Modulus * llama::sizeOf<RecordDim>;
        }

        template<std::size_t... RecordCoords>
        constexpr auto blobNrAndOffset(ArrayIndex ai, llama::RecordCoord<RecordCoords...> = {}) const
            -> llama::NrAndOffset<std::size_t>
        {
            const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;
            const auto offset = (llama::mapping::LinearizeArrayDimsCpp{}(ai, extents()) % Modulus)
                * sizeof(llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>);
            return {blob, offset};
        }
    };
} // namespace

TEST_CASE("mapsNonOverlappingly.ModulusMapping")
{
#ifdef __cpp_constexpr_dynamic_alloc
    using Modulus10Mapping = ModulusMapping<llama::ArrayExtentsDynamic<std::size_t, 1>, Particle, 10>;
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{1}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{9}}));
    STATIC_REQUIRE(llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{10}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{11}}));
    STATIC_REQUIRE(!llama::mapsNonOverlappingly(Modulus10Mapping{llama::ArrayExtentsDynamic<std::size_t, 1>{25}}));
#else
    INFO("Test disabled because compiler does not support __cpp_constexpr_dynamic_alloc");
#endif
}

TEST_CASE("maps.ModulusMapping")
{
    using Extents = llama::ArrayExtentsDynamic<std::size_t, 1>;
    constexpr auto extents = Extents{128};
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::AoS{extents, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<8>(llama::mapping::AoS{extents, Particle{}}));
    STATIC_REQUIRE(!llama::mapsPiecewiseContiguous<16>(llama::mapping::AoS{extents, Particle{}}));

    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::SoA{extents, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<8>(llama::mapping::SoA{extents, Particle{}}));
    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<16>(llama::mapping::SoA{extents, Particle{}}));

    STATIC_REQUIRE(llama::mapsPiecewiseContiguous<1>(llama::mapping::AoSoA<Extents, Particle, 8>{extents}));
    STATIC_REQUIRE(
        llama::mapsPiecewiseContiguous<8>(llama::mapping::AoSoA<Extents, Particle, 8>{extents, Particle{}}));
    STATIC_REQUIRE(
        !llama::mapsPiecewiseContiguous<16>(llama::mapping::AoSoA<Extents, Particle, 8>{extents, Particle{}}));
}

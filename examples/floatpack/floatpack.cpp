#include "../common/IntegralReference.hpp"

#include <cstdint>
#include <fmt/core.h>
#include <llama/llama.hpp>
#include <random>

namespace mapping
{
    namespace internal
    {
        template<typename T>
        struct FloatBitTraits;

        template<>
        struct FloatBitTraits<float>
        {
            static inline constexpr unsigned mantissa = 23;
            static inline constexpr unsigned exponent = 8;
        };

        template<>
        struct FloatBitTraits<double>
        {
            static inline constexpr unsigned mantissa = 52;
            static inline constexpr unsigned exponent = 11;
        };

        template<typename Float, typename StoredIntegralPointer>
        struct FloatingPointReference
        {
            static_assert(
                std::is_same_v<Float, float> || std::is_same_v<Float, double>,
                "Types other than float or double are not implemented yet");
            static_assert(
                std::numeric_limits<Float>::is_iec559,
                "Only IEEE754/IEC559 floating point formats are implemented");

            using FloatBits = std::conditional_t<std::is_same_v<Float, float>, std::uint32_t, std::uint64_t>;

        private:
            ::internal::IntegralReference<FloatBits, StoredIntegralPointer> intref;
            unsigned exponentBits = 0;
            unsigned mantissaBits = 0;

        public:
            FloatingPointReference(
                StoredIntegralPointer p,
                std::size_t bitOffset,
                unsigned exponentBits,
                unsigned mantissaBits)
                : intref{p, bitOffset, exponentBits + mantissaBits + 1}
                , exponentBits(exponentBits)
                , mantissaBits(mantissaBits)
            {
            }

            template<typename Integral>
            static auto repackFloat(
                Integral inFloat,
                unsigned inMantissaBits,
                unsigned inExponentBits,
                unsigned outMantissaBits,
                unsigned outExponentBits) -> Integral
            {
                const Integral mantissaMask = (Integral{1} << inMantissaBits) - 1u;
                const Integral exponentMask = (Integral{1} << inExponentBits) - 1u;

                const Integral inMantissa = inFloat & mantissaMask;
                const Integral inExponent = (inFloat >> inMantissaBits) & exponentMask;
                const Integral inSign = inFloat >> inExponentBits >> inMantissaBits;

                const int outExponentMax = 1 << (outExponentBits - 1); // NOLINT(hicpp-signed-bitwise)
                const int outExponentMin = -outExponentMax + 1;
                const int outExponentBias = outExponentMax - 1;
                const int inExponentBias = (1 << (inExponentBits - 1)) - 1; // NOLINT(hicpp-signed-bitwise)

                const int exponent = static_cast<int>(inExponent) - inExponentBias;
                const auto clampedExponent = std::clamp(exponent, outExponentMin, outExponentMax);
                const Integral rebiasedExponent = clampedExponent + outExponentBias;
                assert(rebiasedExponent < (1u << outExponentBits));

                const Integral packedMantissa = inMantissaBits > outMantissaBits
                    ? inMantissa >> (inMantissaBits - outMantissaBits)
                    : inMantissa << (outMantissaBits - inMantissaBits);
                const Integral packedExponent = rebiasedExponent << outMantissaBits;
                const Integral packedSign = inSign << outExponentBits << outMantissaBits;

                const auto outFloat = static_cast<Integral>(packedMantissa | packedExponent | packedSign);
                return outFloat;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            operator Float() const
            {
                using Bits = internal::FloatBitTraits<Float>;
                const FloatBits packedFloat = intref;
                const FloatBits unpackedFloat
                    = repackFloat(packedFloat, mantissaBits, exponentBits, Bits::mantissa, Bits::exponent);
                Float f;
                std::memcpy(&f, &unpackedFloat, sizeof(Float));
                return f;
            }

            auto operator=(Float f) -> FloatingPointReference&
            {
                using Bits = internal::FloatBitTraits<Float>;
                FloatBits unpackedFloat = 0;
                std::memcpy(&unpackedFloat, &f, sizeof(Float));
                const FloatBits packedFloat
                    = repackFloat(unpackedFloat, Bits::mantissa, Bits::exponent, mantissaBits, exponentBits);
                intref = packedFloat;
                return *this;
            }
        };
    } // namespace internal

    // TODO(bgruber): we could also split each float in the record dimension into 3 integers, sign bit, exponent and
    // mantissa. might not be efficient though the bit layout is [1 sign bit, exponentBits bits from the exponent,
    // mantissaBits bits from the mantissa]+
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename LinearizeArrayDimsFunctor = llama::mapping::LinearizeArrayDimsCpp>
    struct FloatpackSoA : TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;

        static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

        using StoredIntegral = std::conditional_t<
            boost::mp11::mp_contains<llama::FlatRecordDim<RecordDim>, double>::value,
            std::uint64_t,
            std::uint32_t>;

        constexpr FloatpackSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit FloatpackSoA(
            unsigned exponentBits,
            unsigned mantissaBits,
            ArrayExtents extents,
            RecordDim = {})
            : ArrayExtents(extents)
            , exponentBits{exponentBits}
            , mantissaBits{mantissaBits}
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return *this; // NOLINT(cppcoreguidelines-slicing)
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(std::size_t /*blobIndex*/) const -> std::size_t
        {
            constexpr auto bitsPerStoredIntegral = sizeof(StoredIntegral) * CHAR_BIT;
            return (LinearizeArrayDimsFunctor{}.size(extents()) * (exponentBits + mantissaBits + 1)
                    + bitsPerStoredIntegral - 1)
                / bitsPerStoredIntegral;
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blob>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            ArrayIndex ai,
            llama::RecordCoord<RecordCoords...>,
            llama::Array<Blob, blobCount>& blobs) const
        {
            constexpr auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;
            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents()) * (exponentBits + mantissaBits + 1);

            using DstType = llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>;
            return internal::FloatingPointReference<DstType, StoredIntegral*>{
                reinterpret_cast<StoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                exponentBits,
                mantissaBits};
        }

        template<std::size_t... RecordCoords, typename Blob>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
            ArrayIndex ai,
            llama::RecordCoord<RecordCoords...>,
            const llama::Array<Blob, blobCount>& blobs) const
        {
            constexpr auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;
            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents()) * (exponentBits + mantissaBits + 1);

            using DstType = llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>;
            return internal::FloatingPointReference<DstType, const StoredIntegral*>{
                reinterpret_cast<const StoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                exponentBits,
                mantissaBits};
        }

    private:
        unsigned exponentBits = 0;
        unsigned mantissaBits = 0;
    };

} // namespace mapping

// clang-format off
namespace tag
{
    struct X{};
    struct Y{};
} // namespace tag

using Vector = llama::Record<
    llama::Field<tag::X, double>,
    llama::Field<tag::Y, double>
>;
// clang-format on

auto main() -> int
{
    constexpr auto N = 100;
    constexpr auto exponentBits = 5;
    constexpr auto mantissaBits = 13;
    const auto mapping
        = mapping::FloatpackSoA{exponentBits, mantissaBits, llama::ArrayExtents<llama::dyn>{N}, Vector{}};

    auto view = llama::allocView(mapping);

    boost::mp11::mp_for_each<boost::mp11::mp_iota_c<decltype(mapping)::blobCount>>(
        [&](auto ic)
        {
            fmt::print(
                "Blob {}: {} bytes (uncompressed {} bytes)\n",
                ic,
                mapping.blobSize(ic),
                N * sizeof(llama::GetType<Vector, llama::RecordCoord<decltype(ic)::value>>));
        });

    std::default_random_engine engine;
    std::uniform_real_distribution dist{0.0f, 100.0f};

    // view(0)(tag::X{}) = -123.456789f;
    // float f = view(0)(tag::X{});
    // fmt::print("{}", f);

    for(std::size_t i = 0; i < N; i++)
    {
        const auto v = dist(engine);
        view(i)(tag::X{}) = v;
        view(i)(tag::Y{}) = -v;

        fmt::print("{:11} -> {:11}\n", v, static_cast<float>(view(i)(tag::X{})));
        fmt::print("{:11} -> {:11}\n", -v, static_cast<float>(view(i)(tag::Y{})));
    }
}

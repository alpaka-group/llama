#include <cstdint>
#include <fmt/core.h>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct X{};
    struct Y{};
    struct Z{};
} // namespace tag

using Vector = llama::Record<
    llama::Field<tag::X, std::uint16_t>,
    llama::Field<tag::Y, std::int32_t>,
    llama::Field<tag::Z, std::uint64_t>
>;
// clang-format on

template<
    typename TArrayExtents,
    typename TRecordDim,
    typename LinearizeArrayDimsFunctor = llama::mapping::LinearizeArrayDimsCpp>
struct BitpackSoA : TArrayExtents
{
    using ArrayExtents = TArrayExtents;
    using ArrayIndex = typename ArrayExtents::Index;
    using RecordDim = TRecordDim;

    static constexpr std::size_t blobCount = boost::mp11::mp_size<llama::FlatRecordDim<RecordDim>>::value;

    constexpr BitpackSoA() = default;

    LLAMA_FN_HOST_ACC_INLINE
    constexpr explicit BitpackSoA(unsigned bits, ArrayExtents extents, RecordDim = {})
        : ArrayExtents(extents)
        , bits{bits}
    {
    }

    LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
    {
        return *this; // NOLINT(cppcoreguidelines-slicing)
    }

    LLAMA_FN_HOST_ACC_INLINE
    constexpr auto blobSize(std::size_t /*blobIndex*/) const -> std::size_t
    {
        return (LinearizeArrayDimsFunctor{}.size(extents()) * bits + CHAR_BIT - 1) / CHAR_BIT;
    }

    template<std::size_t... RecordCoords>
    static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
    {
        return true;
    }

    // FIXME: might violate alignment
    using RegisterInt = std::uint64_t;

    template<typename T, typename Pointer>
    struct Reference
    {
        Pointer ptr;
        std::size_t bitOffset;
        unsigned bits;

        static constexpr auto registerBits = sizeof(RegisterInt) * CHAR_BIT;

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        operator T() const
        {
            auto* p = ptr + bitOffset / registerBits;
            const auto innerBitOffset = bitOffset % registerBits;
            auto v = p[0] >> innerBitOffset;

            const auto innerBitEndOffset = innerBitOffset + bits;
            if(innerBitEndOffset <= registerBits)
            {
                const auto mask = (RegisterInt{1} << bits) - 1u;
                v &= mask;
            }
            else
            {
                const auto excessBits = innerBitEndOffset - registerBits;
                const auto bitsLoaded = registerBits - innerBitOffset;
                const auto mask = (RegisterInt{1} << excessBits) - 1u;
                v |= (p[1] & mask) << bitsLoaded;
            }
            if constexpr(std::is_signed_v<T>)
                if((v & (RegisterInt{1} << (bits - 1))) != 0)
                {
                    // sign extend
                    v |= static_cast<RegisterInt>(-1) << bits;
                }
            return static_cast<T>(v);
        }

        auto operator=(T v) -> Reference&
        {
            const auto mask = (RegisterInt{1} << bits) - 1u;
            const auto vBits = (static_cast<RegisterInt>(v) & mask);

            auto* p = ptr + bitOffset / registerBits;
            const auto innerBitOffset = bitOffset % registerBits;
            const auto clearMask = ~(mask << innerBitOffset);
            auto m = p[0] & clearMask; // clear previous bits
            m |= vBits << innerBitOffset; // write new bits
            p[0] = m;

            const auto innerBitEndOffset = innerBitOffset + bits;
            if(innerBitEndOffset > registerBits)
            {
                const auto excessBits = innerBitEndOffset - registerBits;
                const auto bitsWritten = registerBits - innerBitOffset;
                const auto clearMask = ~((RegisterInt{1} << excessBits) - 1u);
                auto m = p[1] & clearMask; // clear previous bits
                m |= vBits >> bitsWritten; // write new bits
                p[1] = m;
            }

            return *this;
        }
    };

    template<std::size_t... RecordCoords, typename Blob>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
        ArrayIndex ai,
        llama::RecordCoord<RecordCoords...>,
        llama::Array<Blob, blobCount>& blobs) const
    {
        constexpr auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;
        const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents()) * bits;

        using DstType = llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>;
        return Reference<DstType, RegisterInt*>{reinterpret_cast<RegisterInt*>(&blobs[blob][0]), bitOffset, bits};
    }

    template<std::size_t... RecordCoords, typename Blob>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
        ArrayIndex ai,
        llama::RecordCoord<RecordCoords...>,
        const llama::Array<Blob, blobCount>& blobs) const
    {
        constexpr auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;
        const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents()) * bits;

        using DstType = llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>;
        return Reference<DstType, const RegisterInt*>{
            reinterpret_cast<const RegisterInt*>(&blobs[blob][0]),
            bitOffset,
            bits};
    }

private:
    unsigned bits = 0;
};

auto main() -> int
{
    constexpr auto N = 128;
    constexpr auto bits = 7;
    const auto mapping = BitpackSoA{bits, llama::ArrayExtents<llama::dyn>{N}, Vector{}};

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

    for(std::size_t i = 0; i < N; i++)
    {
        view(i)(tag::X{}) = i;
        view(i)(tag::Y{}) = -static_cast<std::int32_t>(i); // cut-off of sign bits after -64
        view(i)(tag::Z{}) = i * 2; // exceeds bits
    }

    fmt::print("Bitpacked initial:\n");
    for(std::size_t i = 0; i < N; i++)
        fmt::print("[{}, {}, {}]\n", view(i)(tag::X{}), view(i)(tag::Y{}), view(i)(tag::Z{}));

    // extract into a view of full size integers
    auto viewExtracted
        = llama::allocViewUninitialized(llama::mapping::AoS<llama::ArrayExtents<llama::dyn>, Vector>{{N}});
    llama::copy(view, viewExtracted);
    if(!std::equal(view.begin(), view.end(), viewExtracted.begin(), viewExtracted.end()))
        fmt::print("ERROR: unpacked view is different\n");

    // compute something on the extracted view
    for(std::size_t i = 0; i < N; i++)
        viewExtracted(i) = viewExtracted(i) % 10;

    // compress back
    llama::copy(viewExtracted, view);

    fmt::print("Bitpacked after % 10:\n");
    for(std::size_t i = 0; i < N; i++)
        fmt::print("[{}, {}, {}]\n", view(i)(tag::X{}), view(i)(tag::Y{}), view(i)(tag::Z{}));
}

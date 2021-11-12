#include "../common/IntegralReference.hpp"

#include <cstdint>
#include <fmt/core.h>
#include <llama/llama.hpp>

namespace mapping
{
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

        using StoredIntegral
            = std::uint64_t; // TODO(bgruber): we should choose an integral type which is as large as the
                             // largest type in the record dim. Otherwise, we might violate the alignment of the blobs.

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
            constexpr auto bitsPerStoredIntegral = sizeof(StoredIntegral) * CHAR_BIT;
            return (LinearizeArrayDimsFunctor{}.size(extents()) * bits + bitsPerStoredIntegral - 1)
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
            const auto bitOffset = LinearizeArrayDimsFunctor{}(ai, extents()) * bits;

            using DstType = llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>;
            return internal::IntegralReference<DstType, StoredIntegral*>{
                reinterpret_cast<StoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                bits};
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
            return internal::IntegralReference<DstType, const StoredIntegral*>{
                reinterpret_cast<const StoredIntegral*>(&blobs[blob][0]),
                bitOffset,
                bits};
        }

    private:
        unsigned bits = 0;
    };
} // namespace mapping

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

auto main() -> int
{
    constexpr auto N = 128;
    constexpr auto bits = 7;
    const auto mapping = mapping::BitpackSoA{bits, llama::ArrayExtents<llama::dyn>{N}, Vector{}};

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

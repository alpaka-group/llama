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

auto main() -> int
{
    constexpr auto N = 128;
    constexpr auto bits = 7;
    const auto mapping = llama::mapping::BitPackedIntSoA<llama::ArrayExtentsDynamic<1>, Vector>{bits, {N}};

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

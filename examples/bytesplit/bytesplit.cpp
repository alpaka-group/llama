#include <cstdint>
#include <fmt/core.h>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct A{};
    struct B{};
    struct C{};
    struct D{};
    struct E{};
    struct F{};
} // namespace tag

using Data = llama::Record<
    llama::Field<tag::A, std::uint16_t>,
    llama::Field<tag::B, std::int32_t>,
    llama::Field<tag::C, std::uint64_t>,
    llama::Field<tag::D, float>,
    llama::Field<tag::E, double>,
    llama::Field<tag::F, unsigned char>
>;
// clang-format on

auto main() -> int
{
    constexpr auto n = 128;
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 1>;
    const auto mapping
        = llama::mapping::Bytesplit<ArrayExtents, Data, llama::mapping::BindSoA<llama::mapping::Blobs::Single>::fn>{
            {n}};

    auto view = llama::allocView(mapping);

    int value = 0;
    for(std::size_t i = 0; i < n; i++)
        llama::forEachLeafCoord<Data>([&](auto rc) { view(i)(rc) = ++value; });

    value = 0;
    for(std::size_t i = 0; i < n; i++)
        llama::forEachLeafCoord<Data>(
            [&](auto rc)
            {
                using T = llama::GetType<Data, decltype(rc)>;
                ++value;
                if(view(i)(rc) != static_cast<T>(value))
                    fmt::print("Error: value after store is corrupt. {} != {}\n", view(i)(rc), value);
            });

    // extract into a view of unsplit fields
    auto viewExtracted = llama::allocViewUninitialized(llama::mapping::AoS<ArrayExtents, Data>{{n}});
    llama::copy(view, viewExtracted);
    if(!std::equal(view.begin(), view.end(), viewExtracted.begin(), viewExtracted.end()))
        fmt::print("ERROR: unsplit view is different\n");

    // compute something on the extracted view
    for(std::size_t i = 0; i < n; i++)
        viewExtracted(i) *= 2;

    // rearrange back into split view
    llama::copy(viewExtracted, view);

    value = 0;
    for(std::size_t i = 0; i < n; i++)
        llama::forEachLeafCoord<Data>(
            [&](auto rc)
            {
                using T = llama::GetType<Data, decltype(rc)>;
                ++value;
                if(view(i)(rc) != static_cast<T>(static_cast<T>(value) * 2))
                    fmt::print("Error: value after resplit is corrupt. {} != {}\n", view(i)(rc), value);
            });

    // compute something on the split view
    for(std::size_t i = 0; i < n; i++)
        view(i) = view(i) * 2; // cannot do view(i) *= 2; with proxy references

    value = 0;
    for(std::size_t i = 0; i < n; i++)
        llama::forEachLeafCoord<Data>(
            [&](auto rc)
            {
                using T = llama::GetType<Data, decltype(rc)>;
                ++value;
                if(view(i)(rc) != static_cast<T>(static_cast<T>(value) * 4))
                    fmt::print(
                        "Error: value after computation on split data is corrupt. {} != {}\n",
                        view(i)(rc),
                        value);
            });

    fmt::print("Done\n");
}

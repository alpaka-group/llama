// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <fmt/core.h>
#include <llama/llama.hpp>
#include <random>

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
    constexpr auto n = 100;
    constexpr auto exponentBits = 5;
    constexpr auto mantissaBits = 13;
    const auto mapping
        = llama::mapping::BitPackedFloatSoA{llama::ArrayExtents{n}, exponentBits, mantissaBits, Vector{}};

    auto view = llama::allocView(mapping);

    boost::mp11::mp_for_each<boost::mp11::mp_iota_c<decltype(mapping)::blobCount>>(
        [&](auto ic)
        {
            fmt::print(
                "Blob {}: {} bytes (uncompressed {} bytes)\n",
                ic(),
                mapping.blobSize(ic),
                n * sizeof(llama::GetType<Vector, llama::RecordCoord<decltype(ic)::value>>));
        });

    std::default_random_engine engine;
    std::uniform_real_distribution dist{0.0f, 100.0f};

    // view(0)(tag::X{}) = -123.456789f;
    // float f = view(0)(tag::X{});
    // fmt::print("{}", f);

    for(std::size_t i = 0; i < n; i++)
    {
        const auto v = dist(engine);
        view(i)(tag::X{}) = v;
        view(i)(tag::Y{}) = -v;

        fmt::print("{:11} -> {:11}\n", v, static_cast<float>(view(i)(tag::X{})));
        fmt::print("{:11} -> {:11}\n", -v, static_cast<float>(view(i)(tag::Y{})));
    }
}

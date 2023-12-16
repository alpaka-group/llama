// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#ifndef LLAMA_COMPTIME_RECORD_DIM_SIZE
#    define LLAMA_COMPTIME_RECORD_DIM_SIZE 20
#endif

#include "../common/ttjet_13tev_june2019.hpp"

#include <llama/llama.hpp>

using RecordDim = boost::mp11::mp_take_c<Event, LLAMA_COMPTIME_RECORD_DIM_SIZE>;

auto main() -> int
try
{
    auto extents = llama::ArrayExtents{1024 * 1024};
    //    const auto mapping = llama::mapping::PackedAoS<decltype(extents), RecordDim>{extents};
    const auto mapping = llama::mapping::AlignedAoS<decltype(extents), RecordDim>{extents};
    //    const auto mapping = llama::mapping::MultiBlobSoA<decltype(extents), RecordDim>{extents};
    //    const auto mapping = llama::mapping::AoSoA<decltype(extents), RecordDim, 8>{extents};
    //    const auto mapping = llama::mapping::AoSoA<decltype(extents), RecordDim, 32>{extents};
    //    const auto mapping = llama::mapping::AoSoA<decltype(extents), RecordDim, 64>{extents};

    auto view = llama::allocViewUninitialized(mapping);
    llama::forEachLeafCoord<RecordDim>(
        [&](auto coord)
        {
            using Type = llama::GetType<Event, decltype(coord)>;
            for(std::size_t i = 0; i < extents[0]; i++)
                view(i)(coord) = Type{};
        });
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

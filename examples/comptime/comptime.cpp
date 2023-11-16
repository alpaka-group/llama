// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef LLAMA_COMPTIME_RECORD_DIM_SIZE
#    define LLAMA_COMPTIME_RECORD_DIM_SIZE 20
#endif

#include "../common/ttjet_13tev_june2019.hpp"

#include <llama/llama.hpp>

using RecordDim = boost::mp11::mp_take_c<Event, LLAMA_COMPTIME_RECORD_DIM_SIZE>;

auto main() -> int
try
{
    constexpr auto extents = llama::ArrayExtents{1024 * 1024};
    using ArrayExtents = std::remove_const_t<decltype(extents)>;
    //    const auto packedAoSMapping = llama::mapping::PackedAoS<ArrayExtents, RecordDim>{extents};
    const auto alignedAoSMapping = llama::mapping::AlignedAoS<ArrayExtents, RecordDim>{extents};
    //    const auto multiBlobSoAMapping = llama::mapping::MultiBlobSoA<ArrayExtents, RecordDim>{extents};
    //    const auto aosoa8Mapping = llama::mapping::AoSoA<ArrayExtents, RecordDim, 8>{extents};
    //    const auto aosoa32Mapping = llama::mapping::AoSoA<ArrayExtents, RecordDim, 32>{extents};
    //    const auto aosoa64Mapping = llama::mapping::AoSoA<ArrayExtents, RecordDim, 64>{extents};

    auto view = llama::allocViewUninitialized(alignedAoSMapping);
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

// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "common.hpp"

#include <cstdint>

TEST_CASE("mapping.Null")
{
    auto mapping = llama::mapping::Null<llama::ArrayExtents<std::size_t, 128>, Particle>{{}};
    STATIC_REQUIRE(decltype(mapping)::blobCount == 0);

    auto view = llama::allocView(mapping);
    iotaFillView(view);

    for(auto ai : llama::ArrayIndexRange{view.mapping().extents()})
        llama::forEachLeafCoord<Particle>(
            [&](auto rc)
            {
                CAPTURE(ai, rc);
                using Type = llama::GetType<Particle, decltype(rc)>;
                CHECK(view(ai)(rc) == Type{});
            });
}

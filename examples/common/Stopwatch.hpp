// Copyright 2020 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <chrono>
#include <iostream>
#include <string_view>

struct Stopwatch
{
    using clock = std::chrono::high_resolution_clock;

    auto printAndReset(std::string_view eventName, char nl = '\n') -> double
    {
        const auto end = clock::now();
        const auto seconds = std::chrono::duration<double>{end - last}.count();
        std::cout << eventName << " " << seconds << " s" << nl;
        last = clock::now();
        return seconds;
    }

    auto getAndReset() -> double
    {
        const auto end = clock::now();
        const auto seconds = std::chrono::duration<double>{end - last}.count();
        last = clock::now();
        return seconds;
    }

private:
    clock::time_point last = clock::now();
};

// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <cmath>
#include <numeric>
#include <vector>

namespace common
{
    struct Stats
    {
        bool skipNext = true; // to ignore 1 warmup run
        std::vector<double> values;

        Stats()
        {
            // make sure benchmarks don't incur a reallocation
            values.reserve(100);
        }

        void operator()(double val)
        {
            if(skipNext)
                skipNext = false;
            else
                values.push_back(val);
        }

        auto sum() const -> double
        {
            return std::reduce(values.begin(), values.end());
        }

        auto mean() const -> double
        {
            return sum() / static_cast<double>(values.size());
        }

        // sample standard deviation
        auto sstddev() const -> double
        {
            double sum = 0;
            const auto m = mean();
            for(auto v : values)
                sum += (v - m) * (v - m);
            return std::sqrt(sum / static_cast<double>(values.size() - 1));
        }

        auto sem() const -> double
        {
            return sstddev() / std::sqrt(values.size());
        }

        auto operator+=(const Stats& s) -> Stats&
        {
            values.insert(values.end(), s.values.begin(), s.values.end());
            return *this;
        }

        friend auto operator+(Stats a, const Stats& b) -> Stats
        {
            return a += b;
        }
    };
} // namespace common

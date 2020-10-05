/* Copyright 2020 Benjamin Worpitz, Matthias Werner, Jakob Krude,
 *                Sergei Bastrakov, Bernhard Manfred Gruber
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <llama/llama.hpp>
#include <utility>

using DatumDomain = double;

struct HeatEquationKernel
{
    template <typename View>
    void operator()(uint32_t idx, const View& uCurrBuf, View& uNextBuf, uint32_t extent, double dx, double dt) const
    {
        const auto r = dt / (dx * dx);
        if (idx > 0 && idx < extent - 1u)
            uNextBuf[idx] = uCurrBuf[idx] * (1.0 - 2.0 * r) + uCurrBuf[idx - 1] * r + uCurrBuf[idx + 1] * r;
    }
};

// Exact solution to the test problem
// u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
// u(0, t) = u(1, t) = 0
// u(x, 0) = sin(pi * x)
double exactSolution(double const x, double const t)
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-pi * pi * t) * std::sin(pi * x);
}

auto main() -> int
{
    // Parameters (a user is supposed to change numNodesX, numTimeSteps)
    const auto numNodesX = 10000;
    const auto numTimeSteps = 200000;
    const auto tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    const auto dx = 1.0 / static_cast<double>(numNodesX - 1);
    const auto dt = tMax / static_cast<double>(numTimeSteps - 1);

    const auto r = dt / (dx * dx);
    if (r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r << ", it is required to be <= 0.5\n";
        return 1;
    }

    const auto mapping = llama::mapping::AoS{llama::ArrayDomain{numNodesX}, DatumDomain{}};
    auto uNext = llama::allocView(mapping);
    auto uCurr = llama::allocView(mapping);

    // Apply initial conditions for the test problem
    for (uint32_t i = 0; i < numNodesX; i++)
        uCurr[i] = exactSolution(i * dx, 0.0);

    const auto start = std::chrono::high_resolution_clock::now();
    HeatEquationKernel kernel;
    for (int step = 0; step < numTimeSteps; step++)
    {
        for (auto i = 0; i < numNodesX; i++)
            kernel(i, uCurr, uNext, numNodesX, dx, dt);

        // We assume the boundary conditions are constant and so these values
        // do not need to be updated.
        std::swap(uNext, uCurr);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Runtime: " << std::chrono::duration<double>(end - start).count() << "s\n";

    // Calculate error
    double maxError = 0.0;
    for (uint32_t i = 0; i < numNodesX; i++)
    {
        const auto error = std::abs(uNext[i] - exactSolution(i * dx, tMax));
        maxError = std::max(maxError, error);
    }

    const auto errorThreshold = 1e-5;
    const auto resultCorrect = (maxError < errorThreshold);
    if (resultCorrect)
    {
        std::cout << "Execution results correct!\n";
        return 0;
    }
    else
    {
        std::cout << "Execution results incorrect: error = " << maxError << " (the grid resolution may be too low)\n";
        return 2;
    }
}

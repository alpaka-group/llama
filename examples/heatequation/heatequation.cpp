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

#if __has_include(<xsimd/xsimd.hpp>)
#    include <xsimd/xsimd.hpp>
#    define HAVE_XSIMD
#endif

template<typename View>
inline void kernel(uint32_t idx, const View& uCurr, View& uNext, double r)
{
    uNext[idx] = uCurr[idx] * (1.0 - 2.0 * r) + uCurr[idx - 1] * r + uCurr[idx + 1] * r;
}

template<typename View>
void updateScalar(const View& uCurr, View& uNext, uint32_t extent, double dx, double dt)
{
    const auto r = dt / (dx * dx);
    for(auto i = 0; i < extent; i++)
        if(i > 0 && i < extent - 1u)
            kernel(i, uCurr, uNext, r);
}

#ifdef HAVE_XSIMD

template<typename View>
inline void kernelSimd(uint32_t baseIdx, const View& uCurr, View& uNext, double r)
{
    using Simd = xsimd::batch<double>;
    const auto next = Simd::load_unaligned(&uCurr[baseIdx]) * (1.0 - 2.0 * r)
        + Simd::load_unaligned(&uCurr[baseIdx - 1]) * r + Simd::load_unaligned(&uCurr[baseIdx + 1]) * r;
    next.store_unaligned(&uNext[baseIdx]);
}

template<typename View>
void updateSimd(const View& uCurr, View& uNext, uint32_t extent, double dx, double dt)
{
    constexpr auto l = xsimd::batch<double>::size;
    const auto r = dt / (dx * dx);

    const auto blocks = (extent + l - 1) / l;
    for(auto blockIdx = 0; blockIdx < blocks; blockIdx++)
    {
        const auto baseIdx = static_cast<uint32_t>(blockIdx * l);
        if(baseIdx > 0 && baseIdx + l < extent)
            kernelSimd(baseIdx, uCurr, uNext, r);
        else
            for(auto i = baseIdx; i <= baseIdx + l; i++)
                if(i > 0 && i < extent - 1u)
                    kernel(i, uCurr, uNext, r);
    }
}

template<typename View>
void updateSimdPeel(const View& uCurr, View& uNext, uint32_t extent, double dx, double dt)
{
    constexpr auto l = xsimd::batch<double>::size;
    const auto r = dt / (dx * dx);

    for(auto i = 1; i < l; i++)
        kernel(i, uCurr, uNext, r);

    const auto blocksEnd = ((extent - 1) / l) * l;
    for(auto i = l; i < blocksEnd; i += l)
        kernelSimd(i, uCurr, uNext, r);

    for(auto i = blocksEnd; i < extent - 1; i++)
        kernel(i, uCurr, uNext, r);
}

template<typename View>
void updateSimdPeelUnalignedStore(const View& uCurr, View& uNext, uint32_t extent, double dx, double dt)
{
    constexpr auto l = xsimd::batch<double>::size;
    const auto r = dt / (dx * dx);

    const auto blocksEnd = extent - 1 - l;
    for(auto i = 1; i < blocksEnd; i += l)
        kernelSimd(i, uCurr, uNext, r);

    for(auto i = blocksEnd; i < extent - 1; i++)
        kernel(i, uCurr, uNext, r);
}

#endif

// Exact solution to the test problem
// u_t(x, t) = u_xx(x, t), x in [0, 1], t in [0, T]
// u(0, t) = u(1, t) = 0
// u(x, 0) = sin(pi * x)
auto exactSolution(double const x, double const t) -> double
{
    constexpr double pi = 3.14159265358979323846;
    return std::exp(-pi * pi * t) * std::sin(pi * x);
}

auto main() -> int
try
{
    // Parameters (a user is supposed to change extent, timeSteps)
    const auto extent = 10000;
    const auto timeSteps = 200000;
    const auto tMax = 0.001;
    // x in [0, 1], t in [0, tMax]
    const auto dx = 1.0 / static_cast<double>(extent - 1);
    const auto dt = tMax / static_cast<double>(timeSteps - 1);

    const auto r = dt / (dx * dx);
    if(r > 0.5)
    {
        std::cerr << "Stability condition check failed: dt/dx^2 = " << r << ", it is required to be <= 0.5\n";
        return 1;
    }

    const auto mapping = llama::mapping::SoA{llama::ArrayExtents{extent}, double{}};
    // const auto mapping = llama::mapping::BitPackedFloatSoA{5, 32, llama::ArrayExtents{extent}, double{}};
    auto uNext = llama::allocViewUninitialized(mapping);
    auto uCurr = llama::allocViewUninitialized(mapping);

    auto run = [&](std::string_view updateName, auto update)
    {
        // init
        for(int i = 0; i < extent; i++)
            uCurr[i] = exactSolution(i * dx, 0.0);
        uNext[0] = 0;
        uNext[extent - 1] = 0;

        // run simulation
        const auto start = std::chrono::high_resolution_clock::now();
        for(int step = 0; step < timeSteps; step++)
        {
            update(uCurr, uNext, extent, dx, dt);
            std::swap(uNext, uCurr);
        }
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << updateName << " took " << std::chrono::duration<double>(stop - start).count() << "s\t";

        // calculate error
        double maxError = 0.0;
        for(int i = 0; i < extent; i++)
        {
            const auto error = std::abs(uNext[i] - exactSolution(i * dx, tMax));
            maxError = std::max(maxError, error);
        }

        const auto errorThreshold = 1e-5;
        const auto resultCorrect = (maxError < errorThreshold);
        if(resultCorrect)
            std::cout << "Correct!\n";
        else
            std::cout << "Incorrect! error = " << maxError << " (the grid resolution may be too low)\n";
    };

    run("updateScalar                ", [](auto&... args) { updateScalar(args...); });
#ifdef HAVE_XSIMD
    run("updateSimd                  ", [](auto&... args) { updateSimd(args...); });
    run("updateSimdPeel              ", [](auto&... args) { updateSimdPeel(args...); });
    run("updateSimdPeelUnalignedStore", [](auto&... args) { updateSimdPeelUnalignedStore(args...); });
#endif

    return 0;
}
catch(const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

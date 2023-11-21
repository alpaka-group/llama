// Copyright 2021 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <fmt/core.h>
#if __has_include(<omp.h>)
#    include <omp.h>
#endif
#ifdef ALPAKA_DEBUG // defined when the cmake target links to alpaka
#    include <alpaka/alpaka.hpp>
#endif
#include <algorithm>
#include <string>
#ifdef _WIN32
#    define NOMINMAX
#    define WIN32_LEAN_AND_MEAN
#    include <winsock2.h>
#    pragma comment(lib, "ws2_32")
#else
#    include <unistd.h>
#endif

namespace common
{
    // We used boost::asio::ip::host_name() originally, but it complicated the disassembly and requires asio as
    // additional dependency.
    inline auto hostname() -> std::string
    {
        char name[256];
        ::gethostname(name, 256);
        return name;
    }

    inline auto trim(std::string s) -> std::string
    {
        const auto pred = [](char c) { return std::isspace(c) == 0; };
        s.erase(std::find_if(rbegin(s), rend(s), pred).base(), end(s));
        s.erase(begin(s), std::find_if(begin(s), end(s), pred));
        return s;
    }

    template<typename AlpakaAcc = void>
    inline auto captureEnv() -> std::string
    {
        std::string env;

        // hostname
        env += fmt::format("Host: {}", hostname());

        // OpenMP
#ifdef _OPENMP
        const auto maxThreads = static_cast<std::size_t>(omp_get_max_threads());
        const char* ompProcBind = std::getenv("OMP_PROC_BIND"); // NOLINT(concurrency-mt-unsafe)
        const char* ompPlaces = std::getenv("OMP_PLACES"); // NOLINT(concurrency-mt-unsafe)
        ompProcBind = ompProcBind == nullptr ? "no - PLEASE DEFINE ENV.VAR. OMP_PROC_BIND!" : ompProcBind;
        ompPlaces = ompPlaces == nullptr ? "nothing - PLEASE DEFINE ENV.VAR. OMP_PLACES!" : ompPlaces;
        env += fmt::format("; OpenMP: max {} threads, bound {}, to {}", maxThreads, ompProcBind, ompPlaces);
#endif

        // SIMD
        std::string simdArch =
#if defined(__AVX512F__)
            "AVX512F";
#elif defined(__AVX2__)
            "AVX2";
#elif defined(__AVX__)
            "AVX";
#elif defined(__SSE__SSE4_2__)
            "SSE4.2";
#elif defined(__SSE__SSE4_1__)
            "SSE4.1";
#elif defined(__SSE3__)
            "SSE3";
#elif defined(__SSE2__)
            "SSE2";
#elif defined(__ARM_NEON__)
            "NEON";
#elif defined(__ALTIVEC__)
            "ALTIVEC";
#else
            "unknown";
#endif

#ifdef __FMA__
        simdArch += "+FMA";
#endif
        env += fmt::format("; SIMD: {}", simdArch);

        // alpaka
#ifdef ALPAKA_DEBUG // defined when the cmake target links to alpaka
        if constexpr(!std::is_void_v<AlpakaAcc>)
        {
            using Acc = AlpakaAcc;
            auto accName = alpaka::getAccName<Acc>();
            accName.erase(begin(accName) + accName.find_first_of('<'), end(accName)); // drop template arguments
            const auto dev = getDevByIdx(alpaka::Platform<Acc>{}, 0u);
            const auto devName = trim(getName(dev)); // TODO(bgruber): drop trim after fix lands in alpaka
            const auto devProps = alpaka::getAccDevProps<Acc>(dev);
            env += fmt::format(
                "; alpaka acc: {}, dev[0]: {}, SMem: {}KiB",
                accName,
                devName,
                devProps.m_sharedMemSizeBytes / 1024);
        }
#endif

        // CUDA
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDACC__))
        {
            int device;
            cudaGetDevice(&device);
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);
            env += fmt::format(
                "; CUDA dev: {}, {}MiB GM, {}KiB SM",
                prop.name,
                prop.totalGlobalMem / 1024 / 1024,
                prop.sharedMemPerBlock / 1024);
        }
#endif

        return env;
    }
} // namespace common

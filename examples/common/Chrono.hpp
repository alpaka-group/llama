#pragma once

#include <chrono>
#include <iostream>
#include <string_view>

struct Chrono
{
    void printAndReset(std::string_view eventName)
    {
        const auto end = std::chrono::system_clock::now();
        const auto seconds = std::chrono::duration<double>{end - last}.count();
        std::cout << eventName << ":\t" << seconds << " s\n";
        last = end;
    }

    std::chrono::time_point<std::chrono::system_clock> last = std::chrono::system_clock::now();
};

template <typename NullaryFunc>
inline auto timed(const NullaryFunc& f)
{
    const auto start = std::chrono::high_resolution_clock::now();
    if constexpr (std::is_void_v<decltype(f())>)
    {
        f();
        const auto stop = std::chrono::high_resolution_clock::now();
        return start - stop;
    }
    else
    {
        auto r = f();
        const auto stop = std::chrono::high_resolution_clock::now();
        return std::tuple{start - stop, r};
    }
}

template <typename NullaryFunc>
inline auto printTime(std::string_view caption, const NullaryFunc& f, char nl = '\n')
{
    const auto start = std::chrono::high_resolution_clock::now();
    auto stopAndPrint = [&] {
        const auto stop = std::chrono::high_resolution_clock::now();
        std::cout << caption << " took " << std::chrono::duration<double>(stop - start).count() << 's' << nl;
    };
    if constexpr (std::is_void_v<decltype(f())>)
    {
        f();
        stopAndPrint();
    }
    else
    {
        auto r = f();
        stopAndPrint();
        return r;
    }
}

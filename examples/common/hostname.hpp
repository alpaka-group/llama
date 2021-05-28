// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

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
        std::string name(256, '\0');
        ::gethostname(name.data(), 256);
        return name;
    }
} // namespace common

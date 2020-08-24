/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

#include "../preprocessor/macros.hpp"

#include <memory>

namespace llama::allocator
{
    /** Allocator to allocate memory for a \ref View in the \ref Factory using
     *  `std::shared_ptr` in the background. Meaning every time the view is
     * copied, the shared_ptr reference count is increased and both copies share
     * the same memory! \tparam Alignment aligment of the memory used by
     * `std::shared_ptr`, default value is 64
     */
    template<std::size_t Alignment = 64u>
    struct SharedPtr
    {
        LLAMA_NO_HOST_ACC_WARNING
        static inline auto allocate(std::size_t count)
            -> std::shared_ptr<std::byte[]>
        {
            auto * ptr = static_cast<std::byte *>(::operator new[](
                count * sizeof(std::byte), std::align_val_t{Alignment}));
            auto deleter = [=](std::byte * ptr) {
                ::operator delete[](ptr, std::align_val_t{Alignment});
            };
            return std::shared_ptr<std::byte[]>{ptr, deleter};
        }
    };
}

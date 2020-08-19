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
     * the same memory! \tparam T_alignment aligment of the memory used by
     * `std::shared_ptr`, default value is 64
     */
    template<std::size_t T_alignment = 64u>
    struct SharedPtr
    {
        using PrimType = std::byte;
        using BlobType = std::shared_ptr<PrimType[]>;
        using Parameter = int; ///< the optional allocation parameter is ignored

        LLAMA_NO_HOST_ACC_WARNING
        static inline auto allocate(std::size_t count, Parameter const)
            -> BlobType
        {
            auto * ptr = static_cast<PrimType *>(::operator new[](
                count * sizeof(PrimType), std::align_val_t{T_alignment}));
            auto deleter = [=](PrimType * ptr) {
                ::operator delete[](ptr, std::align_val_t{T_alignment});
            };
            return BlobType{ptr, deleter};
        }
    };
}

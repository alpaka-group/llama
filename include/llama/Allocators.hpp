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

#include "Array.hpp"
#include "macros.hpp"

#include <cstddef>
#include <memory>
#include <vector>

namespace llama::allocator
{
    template<std::size_t BytesToReserve>
    struct Stack
    {
        LLAMA_FN_HOST_ACC_INLINE auto allocate(std::size_t) const
            -> Array<std::byte, BytesToReserve>
        {
            return {};
        }
    };

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
        inline auto allocate(std::size_t count) const
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

    namespace internal
    {
        template<typename T, std::size_t Alignment>
        struct AlignmentAllocator
        {
            using value_type = T;

            inline AlignmentAllocator() noexcept = default;

            template<typename T2>
            inline AlignmentAllocator(
                AlignmentAllocator<T2, Alignment> const &) noexcept
            {}

            inline ~AlignmentAllocator() noexcept = default;

            inline auto allocate(std::size_t n) -> T *
            {
                return static_cast<T *>(::operator new[](
                    n * sizeof(T), std::align_val_t{Alignment}));
            }

            inline void deallocate(T * p, std::size_t)
            {
                ::operator delete[](p, std::align_val_t{Alignment});
            }

            template<typename T2>
            struct rebind
            {
                using other = AlignmentAllocator<T2, Alignment>;
            };

            auto
            operator!=(const AlignmentAllocator<T, Alignment> & other) const
                -> bool
            {
                return !(*this == other);
            }

            auto
            operator==(const AlignmentAllocator<T, Alignment> & other) const
                -> bool
            {
                return true;
            }
        };
    }

    /** Allocator to allocate memory for a \ref View in the \ref Factory using
     *  `std::vector` in the background. Meaning every time the view is copied,
     * the whole memory is copied. When the view is moved, the move operator of
     *  `std::vector` is used.
     * \tparam Alignment aligment of the memory used by `std::vector`, default
     *  value is 64
     */
    template<std::size_t Alignment = 64u>
    struct Vector
    {
        LLAMA_NO_HOST_ACC_WARNING
        inline auto allocate(std::size_t count) const
        {
            return std::vector<
                std::byte,
                internal::AlignmentAllocator<std::byte, Alignment>>(count);
        }
    };
}

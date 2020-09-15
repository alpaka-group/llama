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
    /// Allocates stack memory for a \ref View, which is copied each time a \ref
    /// View is copied.
    /// \tparam BytesToReserve the amount of memory to reserve.
    template<std::size_t BytesToReserve>
    struct Stack
    {
        LLAMA_FN_HOST_ACC_INLINE auto operator()(std::size_t) const
            -> Array<std::byte, BytesToReserve>
        {
            return {};
        }
    };

    /// Allocates heap memory managed by a `std::shared_ptr` for a \ref View.
    /// This memory is shared between all copies of a \ref View.
    /// \tparam Alignment aligment of the allocated block of memory.
    template<std::size_t Alignment = 64>
    struct SharedPtr
    {
        LLAMA_NO_HOST_ACC_WARNING
        inline auto operator()(std::size_t count) const
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

    /// Allocates heap memory managed by a `std::vector` for a \ref View, which
    /// is copied each time a \ref View is copied.
    /// \tparam Alignment aligment of the allocated block of memory.
    template<std::size_t Alignment = 64u>
    struct Vector
    {
        LLAMA_NO_HOST_ACC_WARNING
        inline auto operator()(std::size_t count) const
        {
            return std::vector<
                std::byte,
                internal::AlignmentAllocator<std::byte, Alignment>>(count);
        }
    };
}

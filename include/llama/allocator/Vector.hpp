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

#include <vector>

namespace llama::allocator
{
    namespace internal
    {
        template<typename T, std::size_t Alignment>
        struct AlignmentAllocator
        {
            using value_type = T;
            using size_type = std::size_t;
            using difference_type = std::ptrdiff_t;

            using pointer = T *;
            using const_pointer = T const *;

            using reference = T &;
            using const_reference = T const &;

            inline AlignmentAllocator() noexcept = default;

            template<typename T2>
            inline AlignmentAllocator(
                AlignmentAllocator<T2, Alignment> const &) noexcept
            {}

            inline ~AlignmentAllocator() noexcept = default;

            inline auto adress(reference r) -> pointer
            {
                return &r;
            }
            inline auto adress(const_reference r) const -> const_pointer
            {
                return &r;
            }

            inline auto allocate(size_type n) -> pointer
            {
                return static_cast<pointer>(::operator new[](
                    n * sizeof(value_type), std::align_val_t{Alignment}));
            }

            inline void deallocate(pointer p, size_type)
            {
                ::operator delete[](p, std::align_val_t{Alignment});
            }

            inline void construct(pointer p, value_type const &)
            {
                // commented out for performance reasons
                // new ( p ) value_type ( value ); // FIXME this is a bug
            }

            inline auto destroy(pointer p) -> void
            {
                p->~value_type();
            }

            inline auto max_size() const noexcept -> size_type
            {
                return size_type(-1) / sizeof(value_type);
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

            /* Returns true if and only if storage allocated from *this
             * can be deallocated from other, and vice versa.
             * Always returns true for stateless allocators.
             */
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
     * \tparam T_alignment aligment of the memory used by `std::vector`, default
     *  value is 64
     */
    template<std::size_t T_alignment = 64u>
    struct Vector
    {
        using PrimType = std::byte;
        using BlobType = std::vector<
            PrimType,
            internal::AlignmentAllocator<PrimType, T_alignment>>;
        using Parameter = int; ///< the optional allocation parameter is ignored

        LLAMA_NO_HOST_ACC_WARNING
        static inline auto allocate(std::size_t count, Parameter const)
            -> BlobType
        {
            return BlobType(count);
        }
    };
}

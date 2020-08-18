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
#ifdef __linux__
#include <malloc.h>
#endif

namespace llama::allocator
{
    namespace internal
    {
        struct SharedPtrAccessor
        {
            using PrimType = unsigned char;
            using BlobType = std::shared_ptr<PrimType>;

            SharedPtrAccessor(BlobType blob) : blob(blob) {}

            template<typename T_IndexType>
            auto operator[](T_IndexType && idx) -> PrimType &
            {
                return blob.get()[idx];
            }

            template<typename T_IndexType>
            auto operator[](T_IndexType && idx) const -> const PrimType &
            {
                return blob.get()[idx];
            }
            BlobType blob;
        };
    }

    /** Allocator to allocate memory for a \ref View in the \ref Factory using
     *  `std::shared_ptr` in the background. Meaning every time the view is
     * copied, the shared_ptr reference count is increased and both copies share
     * the same memory! \tparam T_alignment aligment of the memory used by
     * `std::shared_ptr`, default value is 64
     */
    template<std::size_t T_alignment = 64u>
    struct SharedPtr
    {
        using PrimType = typename internal::SharedPtrAccessor::
            PrimType; ///< primary type of this allocator is `unsigned char`
        using BlobType
            = internal::SharedPtrAccessor; ///< blob type of this allocator is
                                           ///< `std::shared_ptr<PrimType>`
        using Parameter = int; ///< the optional allocation parameter is ignored

        LLAMA_NO_HOST_ACC_WARNING
        static inline auto allocate(std::size_t count, Parameter const)
            -> BlobType
        {
#if defined _MSC_VER
            PrimType * raw_pointer = reinterpret_cast<PrimType *>(
                _aligned_malloc(count * sizeof(PrimType), T_alignment));
#elif defined __linux__
            PrimType * raw_pointer = reinterpret_cast<PrimType *>(
                memalign(T_alignment, count * sizeof(PrimType)));
#elif defined __MACH__ // Mac OS X
            PrimType * raw_pointer = reinterpret_cast<PrimType *>(malloc(
                count
                * sizeof(
                    PrimType))); // malloc is always 16 byte aligned on Mac.
#else
            PrimType * raw_pointer = reinterpret_cast<PrimType *>(malloc(
                count
                * sizeof(
                    PrimType))); // other (use valloc for page-aligned memory)
#endif
            BlobType accessor(internal::SharedPtrAccessor::BlobType(
                raw_pointer, [=](PrimType * raw_pointer) {
#if defined _MSC_VER
                    _aligned_free(raw_pointer);
#elif defined __linux__
                        free( raw_pointer );
#elif defined __MACH__
                        free( raw_pointer );
#else
                        free( raw_pointer );
#endif
                }));
            return accessor;
        }
    };
}

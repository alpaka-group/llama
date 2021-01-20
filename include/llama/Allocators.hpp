// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "Concepts.hpp"
#include "macros.hpp"

#include <cstddef>
#include <memory>
#include <vector>
#if BOOST_COMP_INTEL != 0
#    include <aligned_new>
#endif

namespace llama::allocator
{
    /// Allocates stack memory for a \ref View, which is copied each time a \ref
    /// View is copied.
    /// \tparam BytesToReserve the amount of memory to reserve.
    template <std::size_t BytesToReserve>
    struct Stack
    {
        LLAMA_FN_HOST_ACC_INLINE auto operator()(std::size_t) const -> Array<std::byte, BytesToReserve>
        {
            return {};
        }
    };
#ifdef __cpp_concepts
    static_assert(StorageBlob<decltype(Stack<64>{}(0))>);
#endif

    /// Allocates heap memory managed by a `std::shared_ptr` for a \ref View.
    /// This memory is shared between all copies of a \ref View.
    /// \tparam Alignment aligment of the allocated block of memory.
    template <std::size_t Alignment = 64>
    struct SharedPtr
    {
        inline auto operator()(std::size_t count) const -> std::shared_ptr<std::byte[]>
        {
            auto* ptr
                = static_cast<std::byte*>(::operator new[](count * sizeof(std::byte), std::align_val_t{Alignment}));
            auto deleter = [=](std::byte* ptr) { ::operator delete[](ptr, std::align_val_t{Alignment}); };
            return std::shared_ptr<std::byte[]>{ptr, deleter};
        }
    };
#ifdef __cpp_concepts
    static_assert(StorageBlob<decltype(SharedPtr{}(0))>);
#endif

    template <typename T, std::size_t Alignment>
    struct AlignedAllocator
    {
        using value_type = T;

        inline AlignedAllocator() noexcept = default;

        template <typename T2>
        inline AlignedAllocator(AlignedAllocator<T2, Alignment> const&) noexcept
        {
        }

        inline ~AlignedAllocator() noexcept = default;

        inline auto allocate(std::size_t n) -> T*
        {
            return static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t{Alignment}));
        }

        inline void deallocate(T* p, std::size_t)
        {
            ::operator delete[](p, std::align_val_t{Alignment});
        }

        template <typename T2>
        struct rebind
        {
            using other = AlignedAllocator<T2, Alignment>;
        };

        auto operator!=(const AlignedAllocator<T, Alignment>& other) const -> bool
        {
            return !(*this == other);
        }

        auto operator==(const AlignedAllocator<T, Alignment>& other) const -> bool
        {
            return true;
        }
    };

    /// Allocates heap memory managed by a `std::vector` for a \ref View, which
    /// is copied each time a \ref View is copied.
    /// \tparam Alignment aligment of the allocated block of memory.
    template <std::size_t Alignment = 64u>
    struct Vector
    {
        inline auto operator()(std::size_t count) const
        {
            return std::vector<std::byte, AlignedAllocator<std::byte, Alignment>>(count);
        }
    };
#ifdef __cpp_concepts
    static_assert(StorageBlob<decltype(Vector{}(0))>);
#endif
} // namespace llama::allocator

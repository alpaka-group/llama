// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "Concepts.hpp"
#include "macros.hpp"

#include <cstddef>
#include <memory>
#include <vector>
#if __has_include(<cuda_runtime.h>)
#    include <cuda_runtime.h>
#endif
#if __has_include(<sycl/sycl.hpp>)
#    include <sycl/sycl.hpp>
#endif

namespace alpaka
{
    template<typename TElem, typename TIdx, typename TExtent, typename TDev>
    auto allocBuf(const TDev& dev, const TExtent& extent); // NOLINT(readability-redundant-declaration)
} // namespace alpaka

namespace llama::bloballoc
{
    /// Allocates statically sized memory for a \ref View, which is copied each time a \ref View is copied.
    /// \tparam BytesToReserve the amount of memory to reserve.
    LLAMA_EXPORT
    template<std::size_t BytesToReserve>
    struct Array
    {
        template<std::size_t Alignment>
        struct alignas(Alignment) AlignedArray : llama::Array<std::byte, BytesToReserve>
        {
        };

        template<std::size_t Alignment>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(
            std::integral_constant<std::size_t, Alignment>,
            [[maybe_unused]] std::size_t count) const -> AlignedArray<Alignment>
        {
            assert(count == BytesToReserve);
            return {};
        }
    };
#ifdef __cpp_lib_concepts
    static_assert(BlobAllocator<Array<64>>);
#endif

    /// Allocates heap memory managed by a `std::unique_ptr` for a \ref View. This memory can only be uniquely owned by
    /// a single \ref View.
    LLAMA_EXPORT
    struct UniquePtr
    {
        template<std::size_t Alignment>
        auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
        {
            auto* ptr
                = static_cast<std::byte*>(::operator new[](count * sizeof(std::byte), std::align_val_t{Alignment}));
            auto deleter = [](std::byte* ptr) { ::operator delete[](ptr, std::align_val_t{Alignment}); };
            return std::unique_ptr<std::byte[], decltype(deleter)>{ptr, deleter};
        }
    };
#ifdef __cpp_lib_concepts
    static_assert(BlobAllocator<UniquePtr>);
#endif

    /// Allocates heap memory managed by a `std::shared_ptr` for a \ref View. This memory is shared between all copies
    /// of a \ref View.
    LLAMA_EXPORT
    struct SharedPtr
    {
        template<std::size_t Alignment>
        auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
            -> std::shared_ptr<std::byte[]>
        {
            auto* ptr
                = static_cast<std::byte*>(::operator new[](count * sizeof(std::byte), std::align_val_t{Alignment}));
            auto deleter = [](std::byte* ptr) { ::operator delete[](ptr, std::align_val_t{Alignment}); };
            return {ptr, deleter};
        }
    };
#ifdef __cpp_lib_concepts
    static_assert(BlobAllocator<SharedPtr>);
#endif

    /// An STL compatible allocator allowing to specify alignment.
    LLAMA_EXPORT
    template<typename T, std::size_t Alignment>
    struct AlignedAllocator
    {
        using value_type = T;

        inline AlignedAllocator() noexcept = default;

        template<typename T2>
        inline explicit AlignedAllocator(const AlignedAllocator<T2, Alignment>&) noexcept
        {
        }

        inline auto allocate(std::size_t n) -> T*
        {
            return static_cast<T*>(::operator new[](n * sizeof(T), std::align_val_t{Alignment}));
        }

        inline void deallocate(T* p, std::size_t)
        {
            ::operator delete[](p, std::align_val_t{Alignment});
        }

        template<typename T2>
        struct rebind // NOLINT(readability-identifier-naming)
        {
            using other = AlignedAllocator<T2, Alignment>;
        };

        auto operator!=(const AlignedAllocator<T, Alignment>& other) const -> bool
        {
            return !(*this == other);
        }

        auto operator==(const AlignedAllocator<T, Alignment>&) const -> bool
        {
            return true;
        }
    };

    /// Allocates heap memory managed by a `std::vector` for a \ref View, which is copied each time a \ref View is
    /// copied.
    LLAMA_EXPORT
    struct Vector
    {
        template<std::size_t Alignment>
        inline auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
        {
            return std::vector<std::byte, AlignedAllocator<std::byte, Alignment>>(count);
        }
    };
#ifdef __cpp_lib_concepts
    static_assert(BlobAllocator<Vector>);
#endif

#if __has_include(<cuda_runtime.h>)
    /// Allocates GPU device memory using cudaMalloc. The memory is managed by a std::unique_ptr with a deleter that
    /// calles cudaFree. If you want to use a view created with this allocator in a CUDA kernel, call \ref shallowCopy
    /// on the view before passing it to the kernel.
    LLAMA_EXPORT
    struct CudaMalloc
    {
        inline static const auto deleter = [](void* p)
        {
            if(const auto code = cudaFree(p); code != cudaSuccess)
                throw std::runtime_error(std::string{"cudaFree failed with code "} + cudaGetErrorString(code));
        };

        template<std::size_t FieldAlignment>
        inline auto operator()(std::integral_constant<std::size_t, FieldAlignment>, std::size_t count) const
        {
            std::byte* p = nullptr;
            if(const auto code = cudaMalloc(&p, count); code != cudaSuccess)
                throw std::runtime_error(std::string{"cudaMalloc failed with code "} + cudaGetErrorString(code));
            if(reinterpret_cast<std::uintptr_t>(p) & (FieldAlignment - 1 != 0u))
                throw std::runtime_error{"cudaMalloc does not align sufficiently"};
            return std::unique_ptr<std::byte[], decltype(deleter)>(p, deleter);
        }
    };
#endif

    /// Allocates alpaka buffers as blobs.
    LLAMA_EXPORT
    template<typename Size, typename Dev>
    struct AlpakaBuf
    {
        Dev& dev;

        template<std::size_t Alignment>
        inline auto operator()(std::integral_constant<std::size_t, Alignment>, std::size_t count) const
        {
            return alpaka::allocBuf<std::byte, Size>(dev, static_cast<Size>(count));
        }
    };

#if __has_include(<sycl/sycl.hpp>)
    /// Allocates shared USM memory using sycl::aligned_alloc_shared. The memory is managed by a std::unique_ptr with a
    /// deleter that calles sycl::free. If you want to use a view created with this allocator in a kernel, call \ref
    /// shallowCopy on the view before passing it to the kernel.
    LLAMA_EXPORT
    struct SyclMallocShared
    {
        sycl::queue queue;

        static auto makeDeleter(sycl::queue q)
        {
            // create lambda in function independent of FieldAlignment template paramter to avoid different blob types
            return [q](void* p) { sycl::free(p, q); };
        }

        template<std::size_t FieldAlignment>
        inline auto operator()(std::integral_constant<std::size_t, FieldAlignment>, std::size_t count) const
        {
            std::byte* p = sycl::aligned_alloc_shared<std::byte>(FieldAlignment, count, queue);
            if(reinterpret_cast<std::uintptr_t>(p) & (FieldAlignment - 1 != 0u))
                throw std::runtime_error{"sycl::malloc_shared does not align sufficiently"};
            return std::unique_ptr<std::byte[], decltype(makeDeleter(queue))>(p, makeDeleter(queue));
        }
    };
#endif
} // namespace llama::bloballoc

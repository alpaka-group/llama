#pragma once

#include "macros.hpp"

#include <atomic>

namespace llama::accessor
{
    /// Default accessor. Passes through the given reference.
    struct Default
    {
        template<typename Reference>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const -> Reference
        {
            return std::forward<Reference>(r);
        }
    };

    /// Allows only read access and returns values instead of references to memory.
    struct ReadOnlyByValue
    {
        template<typename Reference>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const
        {
            using ValueType = std::decay_t<Reference>;
            if constexpr(isProxyReference<ValueType>)
                return static_cast<typename ValueType::value_type>(r);
            else
                return ValueType{r};
        }
    };

    /// Allows only read access by qualifying the references to memory with const. Only works on l-value references.
    struct Const
    {
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> const T&
        {
            return r;
        }
    };

    /// Qualifies references to memory with __restrict. Only works on l-value references.
    struct Restrict
    {
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> T& __restrict
        {
            return r;
        }
    };

#ifdef __cpp_lib_atomic_ref
    /// Accessor wrapping a reference into a std::atomic_ref. Can only wrap l-value references.
    struct Atomic
    {
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> std::atomic_ref<T>
        {
            return std::atomic_ref<T>{r};
        }
    };
#endif
} // namespace llama::accessor

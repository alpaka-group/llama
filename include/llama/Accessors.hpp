#pragma once

#include "Concepts.hpp"
#include "ProxyRefOpMixin.hpp"
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
    struct ByValue
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
        // for l-value references
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> const T&
        {
            return r;
        }

        template<typename Ref>
        struct Reference : ProxyRefOpMixin<Reference<Ref>, typename Ref::value_type>
        {
            using value_type = typename Ref::value_type;

            Ref ref;

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            operator value_type() const
            {
                return static_cast<value_type>(ref);
            }

            template<typename T>
            auto operator=(T) -> Reference&
            {
                static_assert(sizeof(T) == 0, "You cannot write through a Const accessor");
                return *this;
            }
        };

        // for proxy references
        template<typename ProxyReference, std::enable_if_t<llama::isProxyReference<ProxyReference>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ProxyReference r) const
        {
            return Reference<ProxyReference>{{}, std::move(r)};
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

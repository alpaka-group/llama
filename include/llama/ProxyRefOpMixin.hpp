// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "macros.hpp"

namespace llama
{
    /// CRTP mixin for proxy reference types to support all compound assignment and increment/decrement operators.
    template<typename Derived, typename ValueType>
    struct ProxyRefOpMixin
    {
    private:
        LLAMA_FN_HOST_ACC_INLINE constexpr auto derived() -> Derived&
        {
            return static_cast<Derived&>(*this);
        }

        // in principle, load() could be const, but we use it only from non-const functions
        LLAMA_FN_HOST_ACC_INLINE constexpr auto load() -> ValueType
        {
            return static_cast<ValueType>(derived());
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr void store(ValueType t)
        {
            derived() = std::move(t);
        }

    public:
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator+=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs += rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator-=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs -= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator*=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs *= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator/=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs /= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator%=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs %= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator<<=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs <<= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator>>=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs >>= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator&=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs &= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator|=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs |= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator^=(const ValueType& rhs) -> Derived&
        {
            ValueType lhs = load();
            lhs ^= rhs;
            store(lhs);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator++() -> Derived&
        {
            ValueType v = load();
            ++v;
            store(v);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator++(int) -> ValueType
        {
            ValueType v = load();
            ValueType old = v++;
            store(v);
            return old;
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator--() -> Derived&
        {
            ValueType v = load();
            --v;
            store(v);
            return derived();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator--(int) -> ValueType
        {
            ValueType v = load();
            ValueType old = v--;
            store(v);
            return old;
        }
    };
} // namespace llama
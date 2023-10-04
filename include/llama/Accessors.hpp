// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "Concepts.hpp"
#include "ProxyRefOpMixin.hpp"
#include "macros.hpp"

#include <atomic>
#include <memory>
#include <mutex>

namespace llama::accessor
{
    /// Default accessor. Passes through the given reference.
    LLAMA_EXPORT
    struct Default
    {
        template<typename Reference>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const -> Reference
        {
            return std::forward<Reference>(r);
        }
    };

    /// Allows only read access and returns values instead of references to memory.
    LLAMA_EXPORT
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

    /// Allows only read access by qualifying the references to memory with const.
    LLAMA_EXPORT
    struct Const
    {
        // for l-value references
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(T& r) const -> const T&
        {
            return r;
        }

        template<typename Ref>
        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
        struct Reference : ProxyRefOpMixin<Reference<Ref>, typename Ref::value_type>
        {
        private:
            Ref ref;

        public:
            using value_type = typename Ref::value_type;

            LLAMA_FN_HOST_ACC_INLINE constexpr explicit Reference(Ref ref) : ref{ref}
            {
            }

            Reference(const Reference&) = default;

            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
            LLAMA_FN_HOST_ACC_INLINE constexpr auto operator=(const Reference& other) -> Reference&
            {
                *this = static_cast<value_type>(other);
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE operator value_type() const
            {
                return static_cast<value_type>(ref);
            }

            template<typename T>
            LLAMA_FN_HOST_ACC_INLINE auto operator=(T) -> Reference&
            {
                static_assert(sizeof(T) == 0, "You cannot write through a Const accessor");
                return *this;
            }
        };

        // for proxy references
        template<typename ProxyReference, std::enable_if_t<llama::isProxyReference<ProxyReference>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ProxyReference r) const
        {
            return Reference<ProxyReference>{std::move(r)};
        }
    };

    /// Qualifies references to memory with __restrict. Only works on l-value references.
    LLAMA_EXPORT
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
    LLAMA_EXPORT
    struct Atomic
    {
        template<typename T>
        LLAMA_FORCE_INLINE auto operator()(T& r) const -> std::atomic_ref<T>
        {
            return std::atomic_ref<T>{r};
        }
    };
#endif

    /// Locks a mutex during each access to the data structure.
    LLAMA_EXPORT
    template<typename Mutex = std::mutex>
    struct Locked
    {
        // mutexes are usually not movable, so we put them on the heap, so the accessor is movable
        std::unique_ptr<Mutex> m = std::make_unique<Mutex>();

        template<typename Ref, typename Value>
        // NOLINTNEXTLINE(cppcoreguidelines-special-member-functions,hicpp-special-member-functions)
        struct Reference : ProxyRefOpMixin<Reference<Ref, Value>, Value>
        {
            Ref ref;
            Mutex& m;

            using value_type = Value;

            // NOLINTNEXTLINE(bugprone-unhandled-self-assignment,cert-oop54-cpp)
            LLAMA_FORCE_INLINE constexpr auto operator=(const Reference& other) -> Reference&
            {
                const std::lock_guard<Mutex> lock(m);
                *this = static_cast<value_type>(other);
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FORCE_INLINE operator value_type() const
            {
                const std::lock_guard<Mutex> lock(m);
                return static_cast<value_type>(ref);
            }

            template<typename T>
            LLAMA_FORCE_INLINE auto operator=(T t) -> Reference&
            {
                const std::lock_guard<Mutex> lock(m);
                ref = t;
                return *this;
            }
        };

        template<typename PR>
        LLAMA_FORCE_INLINE auto operator()(PR r) const -> Reference<PR, typename PR::value_type>
        {
            return {{}, r, *m};
        }

        template<typename T>
        LLAMA_FORCE_INLINE auto operator()(T& r) const -> Reference<T&, std::remove_cv_t<T>>
        {
            return {{}, r, *m};
        }
    };

    namespace internal
    {
        template<std::size_t I, typename Accessor>
        struct StackedLeave : Accessor
        {
        };
    } // namespace internal

    /// Accessor combining multiple other accessors. The contained accessors are applied in left to right order to the
    /// memory location when forming the reference returned from a view.
    LLAMA_EXPORT
    template<typename... Accessors>
    struct Stacked : internal::StackedLeave<0, Default>
    {
    };

    LLAMA_EXPORT
    template<typename FirstAccessor, typename... MoreAccessors>
    struct Stacked<FirstAccessor, MoreAccessors...>
        : internal::StackedLeave<1 + sizeof...(MoreAccessors), FirstAccessor>
        , Stacked<MoreAccessors...>
    {
        using First = internal::StackedLeave<1 + sizeof...(MoreAccessors), FirstAccessor>;
        using Rest = Stacked<MoreAccessors...>;

        LLAMA_FN_HOST_ACC_INLINE Stacked() = default;

        LLAMA_FN_HOST_ACC_INLINE explicit Stacked(FirstAccessor first, MoreAccessors... rest)
            : First{std::move(first)}
            , Rest{std::move(rest)...}
        {
        }

        template<typename Reference>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Reference&& r) const -> decltype(auto)
        {
            return static_cast<const Rest&>(*this)(static_cast<const First&>(*this)(std::forward<Reference>(r)));
        }
    };
} // namespace llama::accessor

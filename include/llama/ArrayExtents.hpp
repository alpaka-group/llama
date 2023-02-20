// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "Meta.hpp"

#include <limits>
#include <type_traits>

namespace llama
{
    // TODO(bgruber): make this an alias in C++20, when we have CTAD for aliases
    /// Represents a run-time index into the array dimensions.
    /// \tparam Dim Compile-time number of dimensions.
    template<typename T, std::size_t Dim>
    struct ArrayIndex : Array<T, Dim>
    {
        static constexpr std::size_t rank = Dim;
    };

    // allow comparing ArrayIndex with different size types:
    template<std::size_t Dim, typename TA, typename TB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(ArrayIndex<TA, Dim> a, ArrayIndex<TB, Dim> b) -> bool
    {
        for(std::size_t i = 0; i < Dim; ++i)
            if(a[i] != b[i])
                return false;
        return true;
    }

    template<std::size_t Dim, typename TA, typename TB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(ArrayIndex<TA, Dim> a, ArrayIndex<TB, Dim> b) -> bool
    {
        return !(a == b);
    }

    static_assert(
        std::is_trivially_default_constructible_v<ArrayIndex<int, 1>>); // so ArrayIndex<1>{} will produce a zeroed
                                                                        // index. Should hold for all dimensions,
                                                                        // but just checking for <1> here.
    static_assert(std::is_trivially_copy_constructible_v<ArrayIndex<int, 1>>);
    static_assert(std::is_trivially_move_constructible_v<ArrayIndex<int, 1>>);
    static_assert(std::is_trivially_copy_assignable_v<ArrayIndex<int, 1>>);
    static_assert(std::is_trivially_move_assignable_v<ArrayIndex<int, 1>>);

    namespace internal
    {
        template<typename Default, typename... Ints>
        struct IndexTypeFromArgs
        {
            using type = Default;
        };

        template<typename Default, typename FirstInt, typename... Ints>
        struct IndexTypeFromArgs<Default, FirstInt, Ints...>
        {
            static_assert(std::conjunction_v<std::is_same<FirstInt, Ints>...>, "All index types must be the same");
            using type = FirstInt;
        };
    } // namespace internal

    template<typename... Args>
    ArrayIndex(Args...)
        -> ArrayIndex<typename internal::IndexTypeFromArgs<std::size_t, Args...>::type, sizeof...(Args)>;
} // namespace llama

template<typename V, size_t N>
struct std::tuple_size<llama::ArrayIndex<V, N>> : std::integral_constant<size_t, N> // NOLINT(cert-dcl58-cpp)
{
};

template<size_t I, typename V, size_t N>
struct std::tuple_element<I, llama::ArrayIndex<V, N>> // NOLINT(cert-dcl58-cpp)
{
    using type = V;
};

namespace llama
{
    namespace internal
    {
        struct Dyn
        {
            template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE constexpr operator T() const
            {
                return static_cast<T>(-1);
            }

            template<typename T>
            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator==(T i, Dyn) -> bool
            {
                return i == static_cast<T>(-1);
            }

            template<typename T>
            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator==(Dyn d, T i) -> bool
            {
                return i == d;
            }

            template<typename T>
            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator!=(T i, Dyn d) -> bool
            {
                return !(i == d);
            }

            template<typename T>
            LLAMA_FN_HOST_ACC_INLINE friend constexpr auto operator!=(Dyn d, T i) -> bool
            {
                return !(i == d);
            }
        };
    } // namespace internal

    /// Used as a template argument to \ref ArrayExtents to mark a dynamic extent.
    inline constexpr auto dyn = internal::Dyn{};

    /// ArrayExtents holding compile and runtime indices. This is conceptually equivalent to the std::extent of
    /// std::mdspan (@see: https://wg21.link/P0009) including the changes to make the size_type controllable (@see:
    /// https://wg21.link/P2553).
    template<typename T = std::size_t, T... Sizes>
    struct ArrayExtents : Array<T, ((Sizes == dyn) + ... + 0)>
    {
        static constexpr std::size_t rank = sizeof...(Sizes);
        static constexpr auto rankDynamic = ((Sizes == dyn) + ... + 0);
        static constexpr auto rankStatic = rank - rankDynamic;

        using Index = ArrayIndex<T, rank>;
        using value_type = T;

        template<std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto get() const -> value_type
        {
            using TypeList = mp_list_c<T, Sizes...>;
            constexpr auto extent = mp_at_c<TypeList, I>::value;
            if constexpr(extent != dyn)
                return extent;
            else
                return static_cast<const Array<value_type, rankDynamic>&>(
                    *this)[+mp_count<mp_take_c<TypeList, I>, std::integral_constant<T, dyn>>::value];
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](T i) const -> value_type
        {
            return mp_with_index<rank>(i, [&](auto ic) { return get<decltype(ic)::value>(); });
        }

    private:
        template<std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto toArray(std::index_sequence<Is...>) const -> Index
        {
            return {get<Is>()...};
        }

    public:
        LLAMA_FN_HOST_ACC_INLINE constexpr auto toArray() const -> Index
        {
            return toArray(std::make_index_sequence<rank>{});
        }
    };

    template<typename T>
    struct ArrayExtents<T>
    {
        static constexpr std::size_t rank = 0;
        static constexpr auto rankDynamic = 0;
        static constexpr auto rankStatic = 0;

        using Index = ArrayIndex<T, 0>;
        using value_type = T;

        LLAMA_FN_HOST_ACC_INLINE constexpr auto toArray() const -> Index
        {
            return {};
        }
    };

    template<typename... Args>
    ArrayExtents(Args...) -> ArrayExtents<
        typename internal::IndexTypeFromArgs<std::size_t, Args...>::type,
    // we just use Args to repeat dyn as often as Args is big
#ifdef __NVCC__
        (static_cast<std::void_t<Args>>(0),
         dyn.operator typename internal::IndexTypeFromArgs<std::size_t, Args...>::type())...
#else
        (Args{}, dyn)...
#endif
        >;

    static_assert(std::is_trivially_default_constructible_v<ArrayExtents<std::size_t, 1>>);
    static_assert(std::is_trivially_copy_constructible_v<ArrayExtents<std::size_t, 1>>);
    static_assert(std::is_trivially_move_constructible_v<ArrayExtents<std::size_t, 1>>);
    static_assert(std::is_trivially_copy_assignable_v<ArrayExtents<std::size_t, 1>>);
    static_assert(std::is_trivially_move_assignable_v<ArrayExtents<std::size_t, 1>>);
    static_assert(std::is_empty_v<ArrayExtents<std::size_t, 1>>);

    template<typename SizeTypeA, SizeTypeA... SizesA, typename SizeTypeB, SizeTypeB... SizesB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(
        ArrayExtents<SizeTypeA, SizesA...> a,
        ArrayExtents<SizeTypeB, SizesB...> b) -> bool
    {
        return a.toArray() == b.toArray();
    }

    template<typename SizeTypeA, SizeTypeA... SizesA, typename SizeTypeB, SizeTypeB... SizesB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(
        ArrayExtents<SizeTypeA, SizesA...> a,
        ArrayExtents<SizeTypeB, SizesB...> b) -> bool
    {
        return !(a == b);
    }

    template<typename SizeType, SizeType... Sizes>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto product(ArrayExtents<SizeType, Sizes...> e) -> SizeType
    {
        return product(e.toArray());
    }

    namespace internal
    {
        template<typename SizeType, SizeType Extent, std::size_t... Is>
        constexpr auto makeArrayExtents(std::index_sequence<Is...>)
        {
            return ArrayExtents<SizeType, (static_cast<void>(Is), Extent)...>{};
        }
    } // namespace internal

    /// N-dimensional ArrayExtents where all N extents are Extent.
    template<typename SizeType, std::size_t N, SizeType Extent>
    using ArrayExtentsNCube = decltype(internal::makeArrayExtents<SizeType, Extent>(std::make_index_sequence<N>{}));

    /// N-dimensional ArrayExtents where all values are dynamic.
    template<typename SizeType, std::size_t N>
    using ArrayExtentsDynamic = ArrayExtentsNCube<SizeType, N, dyn>;

    template<typename SizeType, std::size_t Dim, typename Func, typename... OuterIndices>
    LLAMA_FN_HOST_ACC_INLINE void forEachADCoord(
        [[maybe_unused]] ArrayIndex<SizeType, Dim> adSize,
        Func&& func,
        OuterIndices... outerIndices)
    {
        LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
        if constexpr(Dim > 0)
        {
            for(SizeType i = 0; i < adSize[0]; i++)
                forEachADCoord(
                    ArrayIndex<SizeType, Dim - 1>{popFront(adSize)},
                    std::forward<Func>(func),
                    outerIndices...,
                    i);
        }
        else
        {
            std::forward<Func>(func)(ArrayIndex<SizeType, sizeof...(outerIndices)>{outerIndices...});
        }
        LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
    }

    template<typename SizeType, SizeType... Sizes, typename Func>
    LLAMA_FN_HOST_ACC_INLINE void forEachADCoord(ArrayExtents<SizeType, Sizes...> extents, Func&& func)
    {
        forEachADCoord(extents.toArray(), std::forward<Func>(func));
    }
} // namespace llama

template<typename SizeType, SizeType... Sizes>
struct std::tuple_size<llama::ArrayExtents<SizeType, Sizes...>> // NOLINT(cert-dcl58-cpp)
    : std::integral_constant<std::size_t, sizeof...(Sizes)>
{
};

template<typename SizeType, std::size_t I, SizeType... Sizes>
struct std::tuple_element<I, llama::ArrayExtents<SizeType, Sizes...>> // NOLINT(cert-dcl58-cpp)
{
    using type = SizeType;
};

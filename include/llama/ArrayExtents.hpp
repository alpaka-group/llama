// SPDX-License-Identifier: GPL-3.0-or-later

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
    template<std::size_t Dim>
    struct ArrayIndex : Array<std::size_t, Dim>
    {
    };

    static_assert(
        std::is_trivially_default_constructible_v<ArrayIndex<1>>); // so ArrayIndex<1>{} will produce a zeroed
                                                                   // index. Should hold for all dimensions,
                                                                   // but just checking for <1> here.
    static_assert(std::is_trivially_copy_constructible_v<ArrayIndex<1>>);
    static_assert(std::is_trivially_move_constructible_v<ArrayIndex<1>>);
    static_assert(std::is_trivially_copy_assignable_v<ArrayIndex<1>>);
    static_assert(std::is_trivially_move_assignable_v<ArrayIndex<1>>);

    template<typename... Args>
    ArrayIndex(Args...) -> ArrayIndex<sizeof...(Args)>;
} // namespace llama

template<size_t N>
struct std::tuple_size<llama::ArrayIndex<N>> : std::integral_constant<size_t, N>
{
};

template<size_t I, size_t N>
struct std::tuple_element<I, llama::ArrayIndex<N>>
{
    using type = size_t;
};

namespace llama
{
    /// Used as a template argument to \ref ArrayExtents to mark a dynamic extent.
    inline constexpr std::size_t dyn = std::numeric_limits<std::size_t>::max();

    /// ArrayExtents holding compile and runtime indices. This is conceptually equivalent to the std::extent of
    /// std::mdspan. See: https://wg21.link/P0009
    template<std::size_t... Sizes>
    struct ArrayExtents : Array<typename ArrayIndex<sizeof...(Sizes)>::value_type, ((Sizes == dyn) + ... + 0)>
    {
        static constexpr std::size_t rank = sizeof...(Sizes);
        static constexpr auto rank_dynamic = ((Sizes == dyn) + ... + 0);
        static constexpr auto rank_static = rank - rank_dynamic;

        using Index = ArrayIndex<rank>;
        using value_type = typename Index::value_type;

        template<std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto get() const
        {
            using namespace boost::mp11;
            using TypeList = mp_list_c<std::size_t, Sizes...>;
            constexpr auto extent = mp_at_c<TypeList, I>::value;
            if constexpr(extent != dyn)
                return extent;
            else
                return static_cast<const Array<value_type, rank_dynamic>&>(
                    *this)[+mp_count<mp_take_c<TypeList, I>, mp_size_t<dyn>>::value];
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator[](std::size_t i) const
        {
            return boost::mp11::mp_with_index<rank>(i, [&](auto ic) { return get<decltype(ic)::value>(); });
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

        LLAMA_FN_HOST_ACC_INLINE constexpr operator Index() const
        {
            return toArray();
        }
    };

    template<>
    struct ArrayExtents<>
    {
        static constexpr std::size_t rank = 0;
        static constexpr auto rank_dynamic = 0;
        static constexpr auto rank_static = 0;

        using Index = ArrayIndex<rank>;
        using value_type = typename Index::value_type;

        LLAMA_FN_HOST_ACC_INLINE constexpr auto toArray() const -> Index
        {
            return {};
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr operator Index() const
        {
            return toArray();
        }
    };

    template<typename... Args>
    ArrayExtents(Args... args) -> ArrayExtents<(Args{}, dyn)...>;

    template<std::size_t... SizesA, std::size_t... SizesB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator==(ArrayExtents<SizesA...> a, ArrayExtents<SizesB...> b) -> bool
    {
        return a.toArray() == b.toArray();
    }

    template<std::size_t... SizesA, std::size_t... SizesB>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto operator!=(ArrayExtents<SizesA...> a, ArrayExtents<SizesB...> b) -> bool
    {
        return !(a == b);
    }

    template<std::size_t... Sizes>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto product(ArrayExtents<Sizes...> e) ->
        typename ArrayExtents<Sizes...>::value_type
    {
        return product(e.toArray());
    }

    /// N-dimensional ArrayExtents where all values are dynamic.
    template<std::size_t N>
    using ArrayExtentsDynamic = internal::
        mp_unwrap_values_into<boost::mp11::mp_repeat_c<boost::mp11::mp_list_c<std::size_t, dyn>, N>, ArrayExtents>;

    /// N-dimensional ArrayExtents where all values are Extent.
    template<std::size_t N, std::size_t Extent>
    using ArrayExtentsStatic = internal::
        mp_unwrap_values_into<boost::mp11::mp_repeat_c<boost::mp11::mp_list_c<std::size_t, Extent>, N>, ArrayExtents>;
} // namespace llama

template<std::size_t... Sizes>
struct std::tuple_size<llama::ArrayExtents<Sizes...>> : std::integral_constant<std::size_t, sizeof...(Sizes)>
{
};

template<std::size_t I, std::size_t... Sizes>
struct std::tuple_element<I, llama::ArrayExtents<Sizes...>>
{
    using type = typename llama::ArrayExtents<Sizes...>::value_type;
};

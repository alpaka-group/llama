#pragma once

#include "Core.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#if CAN_USE_RANGES
#    include <ranges>
#endif

namespace llama
{
    /// Iterator supporting \ref ArrayDimsIndexRange.
    template <std::size_t Dim>
    struct ArrayDimsIndexIterator
    {
        using value_type = ArrayDims<Dim>;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;
        using pointer = internal::IndirectValue<value_type>;
        using iterator_category = std::random_access_iterator_tag;

        constexpr ArrayDimsIndexIterator() noexcept = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr ArrayDimsIndexIterator(ArrayDims<Dim> size, ArrayDims<Dim> current) noexcept
            : size(size)
            , current(current)
        {
        }

        constexpr ArrayDimsIndexIterator(const ArrayDimsIndexIterator&) noexcept = default;
        constexpr ArrayDimsIndexIterator(ArrayDimsIndexIterator&&) noexcept = default;
        constexpr auto operator=(const ArrayDimsIndexIterator&) noexcept -> ArrayDimsIndexIterator& = default;
        constexpr auto operator=(ArrayDimsIndexIterator&&) noexcept -> ArrayDimsIndexIterator& = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator*() const noexcept -> value_type
        {
            return current;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator->() const noexcept -> pointer
        {
            return {**this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++() noexcept -> ArrayDimsIndexIterator&
        {
            current[Dim - 1]++;
            for (auto i = (int) Dim - 2; i >= 0; i--)
            {
                if (current[i + 1] != size[i + 1])
                    return *this;
                current[i + 1] = 0;
                current[i]++;
            }
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++(int) noexcept -> ArrayDimsIndexIterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--() noexcept -> ArrayDimsIndexIterator&
        {
            current[Dim - 1]--;
            for (auto i = (int) Dim - 2; i >= 0; i--)
            {
                if (current[i + 1] != std::numeric_limits<std::size_t>::max())
                    return *this;
                current[i + 1] = size[i] - 1;
                current[i]--;
            }
            // decrementing beyond [0, 0, ..., 0] is UB
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--(int) noexcept -> ArrayDimsIndexIterator
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator[](difference_type i) const noexcept -> reference
        {
            return *(*this + i);
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator+=(difference_type n) noexcept -> ArrayDimsIndexIterator&
        {
            // add n to all lower dimensions with carry
            for (auto i = (int) Dim - 1; i > 0 && n != 0; i--)
            {
                n += static_cast<difference_type>(current[i]);
                const auto s = static_cast<difference_type>(size[i]);
                auto mod = n % s;
                n /= s;
                if (mod < 0)
                {
                    mod += s;
                    n--;
                }
                current[i] = mod;
                assert(current[i] < size[i]);
            }

            current[0] = static_cast<difference_type>(current[0]) + n;
            // current is either within bounds or at the end ([last + 1, 0, 0, ..., 0])
            assert(
                (current[0] < size[0]
                 || (current[0] == size[0]
                     && std::all_of(std::begin(current) + 1, std::end(current), [](auto c) { return c == 0; })))
                && "Iterator was moved past the end");

            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(ArrayDimsIndexIterator it, difference_type n) noexcept -> ArrayDimsIndexIterator
        {
            it += n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(difference_type n, ArrayDimsIndexIterator it) noexcept -> ArrayDimsIndexIterator
        {
            return it + n;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator-=(difference_type n) noexcept -> ArrayDimsIndexIterator&
        {
            return operator+=(-n);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(ArrayDimsIndexIterator it, difference_type n) noexcept -> ArrayDimsIndexIterator
        {
            it -= n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> difference_type
        {
            assert(a.size == b.size);

            difference_type n = a.current[Dim - 1] - b.current[Dim - 1];
            difference_type size = a.size[Dim - 1];
            for (auto i = (int) Dim - 2; i >= 0; i--)
            {
                n += (a.current[i] - b.current[i]) * size;
                size *= a.size[i];
            }

            return n;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator==(
            const ArrayDimsIndexIterator<Dim>& a,
            const ArrayDimsIndexIterator<Dim>& b) noexcept -> bool
        {
            assert(a.size == b.size);
            return a.current == b.current;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator!=(
            const ArrayDimsIndexIterator<Dim>& a,
            const ArrayDimsIndexIterator<Dim>& b) noexcept -> bool
        {
            return !(a == b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            assert(a.size == b.size);
            return std::lexicographical_compare(
                std::begin(a.current),
                std::end(a.current),
                std::begin(b.current),
                std::end(b.current));
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            return b < a;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<=(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            return !(a > b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>=(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            return !(a < b);
        }

    private:
        ArrayDims<Dim> size; // TODO: we only need to store Dim - 1 sizes
        ArrayDims<Dim> current;
    };

    /// Range allowing to iterate over all indices in a \ref ArrayDims.
    template <std::size_t Dim>
    struct ArrayDimsIndexRange
#if CAN_USE_RANGES
        : std::ranges::view_base
#endif
    {
        constexpr ArrayDimsIndexRange() noexcept = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr ArrayDimsIndexRange(ArrayDims<Dim> size) noexcept : size(size)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto begin() const noexcept -> ArrayDimsIndexIterator<Dim>
        {
            return {size, ArrayDims<Dim>{}};
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto end() const noexcept -> ArrayDimsIndexIterator<Dim>
        {
            auto endPos = ArrayDims<Dim>{};
            endPos[0] = size[0];
            return {size, endPos};
        }

    private:
        ArrayDims<Dim> size;
    };
} // namespace llama

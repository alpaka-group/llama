#pragma once

#include "Core.hpp"

#include <algorithm>
#include <iterator>
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

        constexpr ArrayDimsIndexIterator(ArrayDims<Dim> size, ArrayDims<Dim> current) noexcept
            : lastIndex(
                [size]() mutable
                {
                    for (auto i = 0; i < Dim; i++)
                        size[i]--;
                    return size;
                }())
            , current(current)
        {
        }

        constexpr ArrayDimsIndexIterator(const ArrayDimsIndexIterator&) noexcept = default;
        constexpr ArrayDimsIndexIterator(ArrayDimsIndexIterator&&) noexcept = default;
        constexpr auto operator=(const ArrayDimsIndexIterator&) noexcept -> ArrayDimsIndexIterator& = default;
        constexpr auto operator=(ArrayDimsIndexIterator&&) noexcept -> ArrayDimsIndexIterator& = default;

        constexpr auto operator*() const noexcept -> value_type
        {
            return current;
        }

        constexpr auto operator->() const noexcept -> pointer
        {
            return {**this};
        }

        constexpr auto operator++() noexcept -> ArrayDimsIndexIterator&
        {
            for (auto i = (int) Dim - 1; i >= 0; i--)
            {
                if (current[i] < lastIndex[i])
                {
                    current[i]++;
                    return *this;
                }
                current[i] = 0;
            }
            current[0] = lastIndex[0] + 1;
            return *this;
        }

        constexpr auto operator++(int) noexcept -> ArrayDimsIndexIterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr auto operator--() noexcept -> ArrayDimsIndexIterator&
        {
            for (auto i = (int) Dim - 1; i >= 0; i--)
            {
                if (current[i] > 0)
                {
                    current[i]--;
                    return *this;
                }
                current[i] = lastIndex[i];
            }
            // decrementing beyond [0, 0, ..., 0] is UB
            return *this;
        }

        constexpr auto operator--(int) noexcept -> ArrayDimsIndexIterator
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr auto operator[](difference_type i) const noexcept -> reference
        {
            return *(*this + i);
        }

        constexpr auto operator+=(difference_type n) noexcept -> ArrayDimsIndexIterator&
        {
            for (auto i = (int) Dim - 1; i >= 0 && n != 0; i--)
            {
                n += static_cast<difference_type>(current[i]);
                const auto size = static_cast<difference_type>(lastIndex[i]) + 1;
                auto mod = n % size;
                n /= size;
                if (mod < 0)
                {
                    mod += size;
                    n--;
                }
                current[i] = mod;
            }
            assert(n == 0);
            return *this;
        }

        friend constexpr auto operator+(ArrayDimsIndexIterator it, difference_type n) noexcept -> ArrayDimsIndexIterator
        {
            it += n;
            return it;
        }

        friend constexpr auto operator+(difference_type n, ArrayDimsIndexIterator it) noexcept -> ArrayDimsIndexIterator
        {
            return it + n;
        }

        constexpr auto operator-=(difference_type n) noexcept -> ArrayDimsIndexIterator&
        {
            return operator+=(-n);
        }

        friend constexpr auto operator-(ArrayDimsIndexIterator it, difference_type n) noexcept -> ArrayDimsIndexIterator
        {
            it -= n;
            return it;
        }

        friend constexpr auto operator-(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> difference_type
        {
            assert(a.lastIndex == b.lastIndex);

            difference_type n = a.current[Dim - 1] - b.current[Dim - 1];
            difference_type size = a.lastIndex[Dim - 1] + 1;
            for (auto i = (int) Dim - 2; i >= 0; i--)
            {
                n += (a.current[i] - b.current[i]) * size;
                size *= a.lastIndex[i] + 1;
            }

            return n;
        }

        friend constexpr auto operator==(
            const ArrayDimsIndexIterator<Dim>& a,
            const ArrayDimsIndexIterator<Dim>& b) noexcept -> bool
        {
            assert(a.lastIndex == b.lastIndex);
            return a.current == b.current;
        }

        friend constexpr auto operator!=(
            const ArrayDimsIndexIterator<Dim>& a,
            const ArrayDimsIndexIterator<Dim>& b) noexcept -> bool
        {
            return !(a == b);
        }

        friend constexpr auto operator<(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            assert(a.lastIndex == b.lastIndex);
            return std::lexicographical_compare(
                std::begin(a.current),
                std::end(a.current),
                std::begin(b.current),
                std::end(b.current));
        }

        friend constexpr auto operator>(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            return b < a;
        }

        friend constexpr auto operator<=(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            return !(a > b);
        }

        friend constexpr auto operator>=(const ArrayDimsIndexIterator& a, const ArrayDimsIndexIterator& b) noexcept
            -> bool
        {
            return !(a < b);
        }

    private:
        ArrayDims<Dim> lastIndex;
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

        constexpr ArrayDimsIndexRange(ArrayDims<Dim> size) noexcept : size(size)
        {
        }

        constexpr auto begin() const noexcept -> ArrayDimsIndexIterator<Dim>
        {
            return {size, ArrayDims<Dim>{}};
        }

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

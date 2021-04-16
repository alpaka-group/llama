#pragma once

#include "Core.hpp"

#include <algorithm>
#include <iterator>
#if CAN_USE_RANGES
#    include <ranges>
#endif

namespace llama
{
    /// Iterator supporting \ref ArrayDomainIndexRange.
    template <std::size_t Dim>
    struct ArrayDomainIndexIterator
    {
        using value_type = ArrayDomain<Dim>;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;
        using pointer = internal::IndirectValue<value_type>;
        using iterator_category = std::random_access_iterator_tag;

        constexpr ArrayDomainIndexIterator() noexcept = default;

        constexpr ArrayDomainIndexIterator(ArrayDomain<Dim> size, ArrayDomain<Dim> current) noexcept
            : lastIndex([size]() mutable {
                for (auto i = 0; i < Dim; i++)
                    size[i]--;
                return size;
            }())
            , current(current)
        {
        }

        constexpr ArrayDomainIndexIterator(const ArrayDomainIndexIterator&) noexcept = default;
        constexpr ArrayDomainIndexIterator(ArrayDomainIndexIterator&&) noexcept = default;
        constexpr auto operator=(const ArrayDomainIndexIterator&) noexcept -> ArrayDomainIndexIterator& = default;
        constexpr auto operator=(ArrayDomainIndexIterator&&) noexcept -> ArrayDomainIndexIterator& = default;

        constexpr auto operator*() const noexcept -> value_type
        {
            return current;
        }

        constexpr auto operator->() const noexcept -> pointer
        {
            return internal::IndirectValue{**this};
        }

        constexpr auto operator++() noexcept -> ArrayDomainIndexIterator&
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

        constexpr auto operator++(int) noexcept -> ArrayDomainIndexIterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr auto operator--() noexcept -> ArrayDomainIndexIterator&
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

        constexpr auto operator--(int) noexcept -> ArrayDomainIndexIterator
        {
            auto tmp = *this;
            --*this;
            return tmp;
        }

        constexpr auto operator[](difference_type i) const noexcept -> reference
        {
            return *(*this + i);
        }

        constexpr auto operator+=(difference_type n) noexcept -> ArrayDomainIndexIterator&
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

        friend constexpr auto operator+(ArrayDomainIndexIterator it, difference_type n) noexcept
            -> ArrayDomainIndexIterator
        {
            it += n;
            return it;
        }

        friend constexpr auto operator+(difference_type n, ArrayDomainIndexIterator it) noexcept
            -> ArrayDomainIndexIterator
        {
            return it + n;
        }

        constexpr auto operator-=(difference_type n) noexcept -> ArrayDomainIndexIterator&
        {
            return operator+=(-n);
        }

        friend constexpr auto operator-(ArrayDomainIndexIterator it, difference_type n) noexcept
            -> ArrayDomainIndexIterator
        {
            it -= n;
            return it;
        }

        friend constexpr auto operator-(const ArrayDomainIndexIterator& a, const ArrayDomainIndexIterator& b) noexcept
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
            const ArrayDomainIndexIterator<Dim>& a,
            const ArrayDomainIndexIterator<Dim>& b) noexcept -> bool
        {
            assert(a.lastIndex == b.lastIndex);
            return a.current == b.current;
        }

        friend constexpr auto operator!=(
            const ArrayDomainIndexIterator<Dim>& a,
            const ArrayDomainIndexIterator<Dim>& b) noexcept -> bool
        {
            return !(a == b);
        }

        friend constexpr auto operator<(const ArrayDomainIndexIterator& a, const ArrayDomainIndexIterator& b) noexcept
            -> bool
        {
            assert(a.lastIndex == b.lastIndex);
            return std::lexicographical_compare(
                std::begin(a.current),
                std::end(a.current),
                std::begin(b.current),
                std::end(b.current));
        }

        friend constexpr auto operator>(const ArrayDomainIndexIterator& a, const ArrayDomainIndexIterator& b) noexcept
            -> bool
        {
            return b < a;
        }

        friend constexpr auto operator<=(const ArrayDomainIndexIterator& a, const ArrayDomainIndexIterator& b) noexcept
            -> bool
        {
            return !(a > b);
        }

        friend constexpr auto operator>=(const ArrayDomainIndexIterator& a, const ArrayDomainIndexIterator& b) noexcept
            -> bool
        {
            return !(a < b);
        }

    private:
        ArrayDomain<Dim> lastIndex;
        ArrayDomain<Dim> current;
    };

    /// Range allowing to iterate over all indices in a \ref ArrayDomain.
    template <std::size_t Dim>
    struct ArrayDomainIndexRange
#if CAN_USE_RANGES
        : std::ranges::view_base
#endif
    {
        constexpr ArrayDomainIndexRange() noexcept = default;

        constexpr ArrayDomainIndexRange(ArrayDomain<Dim> size) noexcept : size(size)
        {
        }

        constexpr auto begin() const noexcept -> ArrayDomainIndexIterator<Dim>
        {
            return {size, ArrayDomain<Dim>{}};
        }

        constexpr auto end() const noexcept -> ArrayDomainIndexIterator<Dim>
        {
            auto endPos = ArrayDomain<Dim>{};
            endPos[0] = size[0];
            return {size, endPos};
        }

    private:
        ArrayDomain<Dim> size;
    };
} // namespace llama

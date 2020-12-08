#pragma once

#include "Core.hpp"

#include <iterator>

namespace llama
{
    /// Iterator supporting \ref ArrayDomainIndexRange.
    template <std::size_t Dim>
    struct ArrayDomainIndexIterator
    {
        using value_type = ArrayDomain<Dim>;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;
        using pointer = value_type*;
        using iterator_category = std::forward_iterator_tag;

        constexpr ArrayDomainIndexIterator() noexcept = default;

        constexpr ArrayDomainIndexIterator(ArrayDomain<Dim> size, ArrayDomain<Dim> current) noexcept
            : size(size)
            , current(current)
        {
        }

        constexpr ArrayDomainIndexIterator(const ArrayDomainIndexIterator&) noexcept = default;
        constexpr auto operator=(const ArrayDomainIndexIterator&) noexcept -> ArrayDomainIndexIterator& = default;

        constexpr auto operator*() const noexcept -> ArrayDomain<Dim>
        {
            return current;
        }

        constexpr auto operator++() noexcept -> ArrayDomainIndexIterator&
        {
            for (auto i = (int) Dim - 1; i >= 0; i--)
            {
                current[i]++;
                if (current[i] != size[i])
                    return *this;
                current[i] = 0;
            }
            // we reached the end, the iterator now needs to compare equal to a value initialized one
            current = {};
            size = {};
            return *this;
        }

        constexpr auto operator++(int) noexcept -> ArrayDomainIndexIterator&
        {
            return ++*this;
        }

        friend constexpr auto operator==(
            const ArrayDomainIndexIterator<Dim>& a,
            const ArrayDomainIndexIterator<Dim>& b) noexcept -> bool
        {
            return a.size == b.size && a.current == b.current;
        }

        friend constexpr auto operator!=(
            const ArrayDomainIndexIterator<Dim>& a,
            const ArrayDomainIndexIterator<Dim>& b) noexcept -> bool
        {
            return !(a == b);
        }

    private:
        ArrayDomain<Dim> size;
        ArrayDomain<Dim> current;
    };

    /// Range allowing to iterate over all indices in a \ref ArrayDomain.
    template <std::size_t Dim>
    struct ArrayDomainIndexRange
    {
        constexpr ArrayDomainIndexRange(ArrayDomain<Dim> size) : size(size)
        {
        }

        constexpr auto begin() const -> ArrayDomainIndexIterator<Dim>
        {
            return {size, ArrayDomain<Dim>{}};
        }

        constexpr auto end() const -> ArrayDomainIndexIterator<Dim>
        {
            return {};
        }

    private:
        ArrayDomain<Dim> size;
    };
} // namespace llama

#pragma once

#include "Core.hpp"

#include <boost/stl_interfaces/iterator_interface.hpp>

namespace llama
{
    /// Iterator supporting \ref ArrayDomainIndexRange.
    template <std::size_t Dim>
    struct ArrayDomainIndexIterator
        : boost::stl_interfaces::iterator_interface<
              ArrayDomainIndexIterator<Dim>,
              std::forward_iterator_tag,
              ArrayDomain<Dim>,
              ArrayDomain<Dim>>
    {
        constexpr ArrayDomainIndexIterator() noexcept = default;

        constexpr ArrayDomainIndexIterator(ArrayDomain<Dim> size, ArrayDomain<Dim> current) noexcept
            : size(size)
            , current(current)
        {
        }

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
            // we reached the end
            current[0] = size[0];
            return *this;
        }

        constexpr auto operator++(int) noexcept -> ArrayDomainIndexIterator&
        {
            return ++*this;
        }

        //template <std::size_t Dim>
        friend constexpr auto operator==(
            const ArrayDomainIndexIterator<Dim>& a,
            const ArrayDomainIndexIterator<Dim>& b) noexcept -> bool
        {
            return a.size == b.size && a.current == b.current;
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
            ArrayDomain<Dim> e{};
            e[0] = size[0];
            return {size, e};
        }

    private:
        ArrayDomain<Dim> size;
    };
} // namespace llama

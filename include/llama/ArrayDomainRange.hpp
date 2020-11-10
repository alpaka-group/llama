#pragma once

#include "Core.hpp"

namespace llama
{
    /// Iterator supporting \ref ArrayDomainIndexRange.
    template <std::size_t Dim>
    struct ArrayDomainIndexIterator
        : boost::iterator_facade<
              ArrayDomainIndexIterator<Dim>,
              ArrayDomain<Dim>,
              std::forward_iterator_tag,
              ArrayDomain<Dim>>
    {
        ArrayDomainIndexIterator() = default;

        ArrayDomainIndexIterator(ArrayDomain<Dim> size, ArrayDomain<Dim> current) : size(size), current(current)
        {
        }

    private:
        friend class boost::iterator_core_access;

        auto dereference() const -> ArrayDomain<Dim>
        {
            return current;
        }

        void increment()
        {
            for (auto i = (int) Dim - 1; i >= 0; i--)
            {
                current[i]++;
                if (current[i] != size[i])
                    return;
                current[i] = 0;
            }
            // we reached the end
            current[0] = size[0];
        }

        auto equal(const ArrayDomainIndexIterator& other) const -> bool
        {
            return size == other.size && current == other.current;
        }

        ArrayDomain<Dim> size;
        ArrayDomain<Dim> current;
    };

    /// Range allowing to iterate over all indices in a \ref ArrayDomain.
    template <std::size_t Dim>
    struct ArrayDomainIndexRange
    {
        ArrayDomainIndexRange(ArrayDomain<Dim> size) : size(size)
        {
        }

        auto begin() const -> ArrayDomainIndexIterator<Dim>
        {
            return {size, ArrayDomain<Dim>{}};
        }

        auto end() const -> ArrayDomainIndexIterator<Dim>
        {
            ArrayDomain<Dim> e{};
            e[0] = size[0];
            return {size, e};
        }

    private:
        ArrayDomain<Dim> size;
    };
} // namespace llama

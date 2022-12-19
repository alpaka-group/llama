#pragma once

#include "ArrayExtents.hpp"
#include "Core.hpp"
#include "HasRanges.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#if CAN_USE_RANGES
#    include <ranges>
#endif

namespace llama
{
    /// Iterator supporting \ref ArrayIndexRange.
    template<typename ArrayExtents>
    struct ArrayIndexIterator
    {
        static_assert(!std::is_const_v<ArrayExtents>);

        using value_type = typename ArrayExtents::Index;
        using difference_type = std::ptrdiff_t;
        using reference = value_type;
        using pointer = internal::IndirectValue<value_type>;
        using iterator_category = std::random_access_iterator_tag;

        static constexpr std::size_t rank = ArrayExtents::rank;

        constexpr ArrayIndexIterator() noexcept = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr ArrayIndexIterator(ArrayExtents extents, value_type current) noexcept
            : extents(extents)
            , current(current)
        {
        }

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
        constexpr auto operator++() noexcept -> ArrayIndexIterator&
        {
            current[rank - 1]++;
            for(auto i = static_cast<int>(rank) - 2; i >= 0; i--)
            {
                if(current[i + 1] != extents[i + 1])
                    return *this;
                current[i + 1] = 0;
                current[i]++;
            }
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++(int) noexcept -> ArrayIndexIterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--() noexcept -> ArrayIndexIterator&
        {
            current[rank - 1]--;
            for(auto i = static_cast<int>(rank) - 2; i >= 0; i--)
            {
                // return if no underflow
                if(current[i + 1] != static_cast<typename ArrayExtents::value_type>(-1))
                    return *this;
                current[i + 1] = extents[i] - 1;
                current[i]--;
            }
            // decrementing beyond [0, 0, ..., 0] is UB
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--(int) noexcept -> ArrayIndexIterator
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
        constexpr auto operator+=(difference_type n) noexcept -> ArrayIndexIterator&
        {
            // add n to all lower dimensions with carry
            for(auto i = static_cast<int>(rank) - 1; i > 0 && n != 0; i--)
            {
                n += static_cast<difference_type>(current[i]);
                const auto s = static_cast<difference_type>(extents[i]);
                auto mod = n % s;
                n /= s;
                if(mod < 0)
                {
                    mod += s;
                    n--;
                }
                current[i] = mod;
                assert(current[i] < extents[i]);
            }

            current[0] = static_cast<difference_type>(current[0]) + n;
            // current is either within bounds or at the end ([last + 1, 0, 0, ..., 0])
            assert(
                (current[0] < extents[0]
                 || (current[0] == extents[0]
                     && std::all_of(std::begin(current) + 1, std::end(current), [](auto c) { return c == 0; })))
                && "Iterator was moved past the end");

            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(ArrayIndexIterator it, difference_type n) noexcept -> ArrayIndexIterator
        {
            it += n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(difference_type n, ArrayIndexIterator it) noexcept -> ArrayIndexIterator
        {
            return it + n;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator-=(difference_type n) noexcept -> ArrayIndexIterator&
        {
            return operator+=(-n);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(ArrayIndexIterator it, difference_type n) noexcept -> ArrayIndexIterator
        {
            it -= n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept
            -> difference_type
        {
            assert(a.extents == b.extents);

            difference_type n = a.current[rank - 1] - b.current[rank - 1];
            difference_type size = a.extents[rank - 1];
            for(auto i = static_cast<int>(rank) - 2; i >= 0; i--)
            {
                n += (a.current[i] - b.current[i]) * size;
                size *= a.extents[i];
            }

            return n;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator==(
            const ArrayIndexIterator<ArrayExtents>& a,
            const ArrayIndexIterator<ArrayExtents>& b) noexcept -> bool
        {
            assert(a.extents == b.extents);
            return a.current == b.current;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator!=(
            const ArrayIndexIterator<ArrayExtents>& a,
            const ArrayIndexIterator<ArrayExtents>& b) noexcept -> bool
        {
            return !(a == b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
        {
            assert(a.extents == b.extents);
#ifdef __NVCC__
            // from: https://en.cppreference.com/w/cpp/algorithm/lexicographical_compare
            auto first1 = std::begin(a.current);
            auto last1 = std::end(a.current);
            auto first2 = std::begin(b.current);
            auto last2 = std::end(b.current);
            for(; (first1 != last1) && (first2 != last2); ++first1, (void) ++first2)
            {
                if(*first1 < *first2)
                    return true;
                if(*first2 < *first1)
                    return false;
            }

            return (first1 == last1) && (first2 != last2);
#else
            return std::lexicographical_compare(
                std::begin(a.current),
                std::end(a.current),
                std::begin(b.current),
                std::end(b.current));
#endif
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
        {
            return b < a;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<=(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
        {
            return !(a > b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>=(const ArrayIndexIterator& a, const ArrayIndexIterator& b) noexcept -> bool
        {
            return !(a < b);
        }

    private:
        ArrayExtents extents; // TODO(bgruber): we only need to store rank - 1 sizes
        value_type current;
    };

    /// Range allowing to iterate over all indices in an \ref ArrayExtents.
    template<typename ArrayExtents>
    struct ArrayIndexRange
        : private ArrayExtents
#if CAN_USE_RANGES
        , std::ranges::view_base
#endif
    {
        static_assert(!std::is_const_v<ArrayExtents>);

        constexpr ArrayIndexRange() noexcept = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit ArrayIndexRange(ArrayExtents extents) noexcept : ArrayExtents(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto begin() const noexcept -> ArrayIndexIterator<ArrayExtents>
        {
            return {*this, typename ArrayExtents::Index{}};
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto end() const noexcept -> ArrayIndexIterator<ArrayExtents>
        {
            auto endPos = typename ArrayExtents::Index{};
            endPos[0] = this->toArray()[0];
            return {*this, endPos};
        }
    };
} // namespace llama

#include "View.hpp"

//#include <boost/stl_interfaces/iterator_interface.hpp>

namespace llama
{
    // requires Boost 1.74 which is quite new
#if 0
    template <typename View>
    struct Iterator
        : boost::stl_interfaces::proxy_iterator_interface<
              Iterator<View>,
              std::random_access_iterator_tag,
              decltype(std::declval<View>()(ArrayDomain<1>{}))>
    {
        constexpr decltype(auto) operator*() const
        {
            return (*view)(coord);
        }

        constexpr auto operator+=(std::ptrdiff_t n) -> Iterator&
        {
            coord[0] += n;
            return *this;
        }

        friend constexpr auto operator-(const Iterator& a, const Iterator& b)
        {
            return a.coord[0] - b.coord[0];
        }

        friend constexpr bool operator==(const Iterator& a, const Iterator& b)
        {
            return a.coord == b.coord;
        }

        ArrayDomain<1> coord;
        View* view;
    };
#endif

    namespace internal
    {
        template <typename T>
        struct IndirectValue
        {
            T value;

            auto operator->() -> T*
            {
                return &value;
            }

            auto operator->() const -> const T*
            {
                return &value;
            }
        };
    } // namespace internal

    template <typename View>
    struct Iterator
    {
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename View::VirtualDatumType;
        using difference_type = std::ptrdiff_t;
        using pointer = internal::IndirectValue<value_type>;
        using reference = value_type;

        constexpr auto operator++() -> Iterator&
        {
            ++coord[0];
            return *this;
        }
        constexpr auto operator++(int) -> Iterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr auto operator--() -> Iterator&
        {
            --coord[0];
            return *this;
        }

        constexpr auto operator--(int) -> Iterator
        {
            auto tmp{*this};
            --*this;
            return tmp;
        }

        constexpr auto operator*() const -> reference
        {
            return (*view)(coord);
        }

        constexpr auto operator->() const -> pointer
        {
            return internal::IndirectValue{**this};
        }

        constexpr auto operator[](difference_type i) const -> reference
        {
            return *(*this + i);
        }

        constexpr auto operator+=(difference_type n) -> Iterator&
        {
            coord[0] = static_cast<difference_type>(coord[0]) + n;
            return *this;
        }

        friend constexpr auto operator+(Iterator it, difference_type n) -> Iterator
        {
            it += n;
            return it;
        }

        friend constexpr auto operator+(difference_type n, Iterator it) -> Iterator
        {
            return it + n;
        }

        constexpr auto operator-=(difference_type n) -> Iterator&
        {
            coord[0] = static_cast<difference_type>(coord[0]) - n;
            return *this;
        }

        friend constexpr auto operator-(Iterator it, difference_type n) -> Iterator
        {
            it -= n;
            return it;
        }

        friend constexpr auto operator-(const Iterator& a, const Iterator& b) -> difference_type
        {
            return static_cast<std::ptrdiff_t>(a.coord[0]) - static_cast<std::ptrdiff_t>(b.coord[0]);
        }

        friend constexpr auto operator==(const Iterator& a, const Iterator& b) -> bool
        {
            return a.coord[0] == b.coord[0];
        }

        friend constexpr auto operator!=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a == b);
        }

        friend constexpr auto operator<(const Iterator& a, const Iterator& b) -> bool
        {
            return a.coord[0] < b.coord[0];
        }

        friend constexpr auto operator>(const Iterator& a, const Iterator& b) -> bool
        {
            return b < a;
        }

        friend constexpr auto operator<=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a > b);
        }

        friend constexpr auto operator>=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a < b);
        }

        ArrayDomain<1> coord;
        View* view;
    };

    // Currently, only 1D iterators are supported, becaues higher dimensional iterators are difficult if we also want to
    // preserve good codegen. Multiple nested loops seem to be superior to a single iterator over multiple dimensions.
    // At least compilers are able to produce better code.
    // std::mdspan also discovered similar difficulties and there was a discussion in WG21 in Oulu 2016 to
    // remove/postpone iterators from the design. In std::mdspan's design, the iterator iterated over the co-domain.

    template <typename View>
    auto begin(View& view) -> Iterator<View>
    {
        static_assert(View::ArrayDomain::rank == 1, "Iterators for non-1D views are not implemented");
        return {ArrayDomain<1>{}, &view};
    }

    template <typename View>
    auto end(View& view) -> Iterator<View>
    {
        static_assert(View::ArrayDomain::rank == 1, "Iterators for non-1D views are not implemented");
        return {view.mapping.arrayDomainSize, &view};
    }
} // namespace llama

#include "View.hpp"

#include <boost/iterator/iterator_facade.hpp>
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

    template <typename View>
    struct Iterator
        : boost::iterators::iterator_facade<
              Iterator<View>,
              typename View::VirtualDatumType,
              std::random_access_iterator_tag,
              typename View::VirtualDatumType,
              std::ptrdiff_t>
    {
        constexpr decltype(auto) dereference() const
        {
            return (*view)(coord);
        }

        constexpr bool equal(const Iterator& other) const
        {
            return coord == other.coord;
        }

        constexpr auto increment() -> Iterator&
        {
            coord[0]++;
            return *this;
        }

        constexpr auto decrement() -> Iterator&
        {
            coord[0]--;
            return *this;
        }

        constexpr auto advance(std::ptrdiff_t n) -> Iterator&
        {
            coord[0] += n;
            return *this;
        }

        constexpr auto distance_to(const Iterator& other) const -> std::ptrdiff_t
        {
            return static_cast<std::ptrdiff_t>(other.coord[0]) - static_cast<std::ptrdiff_t>(coord[0]);
        }

        ArrayDomain<1> coord;
        View* view;
    };

    template <typename View>
    auto begin(View& view) -> Iterator<View>
    {
        static_assert(View::ArrayDomain::rank == 1, "Iterators for non-1D views are not implemented");
        return {{}, ArrayDomain<1>{}, &view};
    }

    template <typename View>
    auto end(View& view) -> Iterator<View>
    {
        static_assert(View::ArrayDomain::rank == 1, "Iterators for non-1D views are not implemented");
        return {{}, view.mapping.arrayDomainSize, &view};
    }
} // namespace llama

// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "View.hpp"
#include "VirtualRecord.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace llama
{
    // TODO: expose blob allocator
    /// An equivalent of std::vector<T> backed by a \ref View. Elements are never value initialized though. No strong
    /// exception guarantee.
    /// WARNING: This class is experimental.
    /// @tparam Mapping The mapping to be used for the underlying view. Needs to have 1 array dimension.
    template<typename Mapping>
    struct Vector
    {
        static_assert(Mapping::ArrayDims::rank == 1, "llama::Vector only supports 1D mappings");

        using ViewType = decltype(allocView<Mapping>());
        using RecordDim = typename Mapping::RecordDim;

        using iterator = decltype(std::declval<ViewType>().begin());
        using value_type = typename iterator::value_type;

        Vector() = default;

        template<typename VirtualRecord = One<RecordDim>>
        LLAMA_FN_HOST_ACC_INLINE explicit Vector(std::size_t count, const VirtualRecord& value = {})
        {
            reserve(count);
            for(std::size_t i = 0; i < count; i++)
                push_back(value);
        }

        template<typename Iterator>
        LLAMA_FN_HOST_ACC_INLINE Vector(Iterator first, Iterator last)
        {
            if constexpr(std::is_same_v<
                             typename std::iterator_traits<Iterator>::iterator_category,
                             std::random_access_iterator_tag>)
                reserve(std::distance(first, last));
            for(; first != last; ++first)
                push_back(*first);
        }

        LLAMA_FN_HOST_ACC_INLINE Vector(const Vector& other)
        {
            m_view = other.m_view; // we depend on the copy semantic of View here
            m_size = other.m_size;
        }

        LLAMA_FN_HOST_ACC_INLINE Vector(Vector&& other) noexcept
        {
            swap(other);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator=(const Vector& other) -> Vector&
        {
            m_view = other.m_view; // we depend on the copy semantic of View here
            m_size = other.m_size;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator=(Vector&& other) noexcept -> Vector&
        {
            swap(other);
            return *this;
        }

        // TODO: assign

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) at(std::size_t i)
        {
            if(i >= m_size)
                throw std::out_of_range{
                    "Index " + std::to_string(i) + "out of range [0:" + std::to_string(m_size) + "["};
            return m_view(i);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) at(std::size_t i) const
        {
            if(i >= m_size)
                throw std::out_of_range{
                    "Index " + std::to_string(i) + "out of range [0:" + std::to_string(m_size) + "["};
            return m_view(i);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator[](std::size_t i)
        {
            return m_view(i);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator[](std::size_t i) const
        {
            return m_view(i);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) front()
        {
            return m_view(0);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) front() const
        {
            return m_view(0);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) back()
        {
            return m_view(m_size - 1);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) back() const
        {
            return m_view(m_size - 1);
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) begin()
        {
            return m_view.begin();
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) begin() const
        {
            return m_view.begin();
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) cbegin()
        {
            return std::as_const(m_view).begin();
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) cbegin() const
        {
            return m_view.begin();
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) end()
        {
            return m_view.begin() + m_size;
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) end() const
        {
            return m_view.begin() + m_size;
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) cend()
        {
            return std::as_const(m_view).begin() + m_size;
        }

        LLAMA_FN_HOST_ACC_INLINE decltype(auto) cend() const
        {
            return m_view.begin() + m_size;
        }

        LLAMA_FN_HOST_ACC_INLINE auto empty() const -> bool
        {
            return m_size == 0;
        }

        LLAMA_FN_HOST_ACC_INLINE auto size() const -> std::size_t
        {
            return m_size;
        }

        LLAMA_FN_HOST_ACC_INLINE void reserve(std::size_t cap)
        {
            if(cap > capacity())
                changeCapacity(cap);
        }

        LLAMA_FN_HOST_ACC_INLINE auto capacity() const -> std::size_t
        {
            return m_view.mapping().arrayDims()[0];
        }

        LLAMA_FN_HOST_ACC_INLINE void shrink_to_fit()
        {
            changeCapacity(m_size);
        }

        LLAMA_FN_HOST_ACC_INLINE void clear()
        {
            m_size = 0;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto insert(iterator pos, T&& t) -> iterator
        {
            const auto i = pos - begin();
            reserve(m_size + 1); // might invalidate pos
            pos = begin() + i;
            std::copy_backward(pos, end(), end() + 1);
            m_view[i] = std::forward<T>(t);
            m_size++;
            return pos;
        }

        // TODO: more insert overloads

        // TODO emplace

        LLAMA_FN_HOST_ACC_INLINE auto erase(iterator pos) -> iterator
        {
            std::copy(pos + 1, end(), pos);
            m_size--;
            return pos;
        }

        // TODO: more erase overloads

        // TODO: T here is probably a virtual record. We could also allow any struct that is storable to the view via
        // VirtualRecord::store().
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE void push_back(T&& t)
        {
            if(const auto cap = capacity(); m_size == cap)
                reserve(std::max(cap + cap / 2, m_size + 1));

            m_view[m_size++] = std::forward<T>(t);
        }

        // TODO: emplace_back

        LLAMA_FN_HOST_ACC_INLINE void pop_back()
        {
            m_size--;
        }

        template<typename VirtualRecord = One<RecordDim>>
        LLAMA_FN_HOST_ACC_INLINE void resize(std::size_t count, const VirtualRecord& value = {})
        {
            reserve(count);
            for(std::size_t i = m_size; i < count; i++)
                m_view[i] = value;
            m_size = count;
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const Vector& a, const Vector& b) -> bool
        {
            if(a.m_size != b.m_size)
                return false;
            return std::equal(a.begin(), a.end(), b.begin());
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const Vector& a, const Vector& b) -> bool
        {
            return !(a == b);
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const Vector& a, const Vector& b) -> bool
        {
            return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const Vector& a, const Vector& b) -> bool
        {
            return !(b < a);
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const Vector& a, const Vector& b) -> bool
        {
            return b < a;
        }

        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const Vector& a, const Vector& b) -> bool
        {
            return !(a < b);
        }

        LLAMA_FN_HOST_ACC_INLINE friend void swap(Vector& a, Vector& b) noexcept
        {
            a.swap(b);
        }

    private:
        LLAMA_FN_HOST_ACC_INLINE void changeCapacity(std::size_t cap)
        {
            auto newView = llama::allocView<Mapping>(Mapping{typename Mapping::ArrayDims{cap}});
            auto b = begin();
            std::copy(begin(), b + std::min(m_size, cap), newView.begin());
            using std::swap;
            swap(m_view, newView); // depends on move semantic of View
        }

        LLAMA_FN_HOST_ACC_INLINE void swap(Vector& other)
        {
            using std::swap;
            swap(m_view, other.m_view); // depends on move semantic of View
            swap(m_size, other.m_size);
        }

        ViewType m_view = {};
        std::size_t m_size = 0;
    };


} // namespace llama

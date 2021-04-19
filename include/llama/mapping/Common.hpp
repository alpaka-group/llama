// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"

#include <climits>

namespace llama::mapping
{
    namespace internal
    {
        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto product(const ArrayDomain<Dim>& size) -> std::size_t
        {
            std::size_t prod = 1;
            for (auto s : size)
                prod *= s;
            return prod;
        }
    } // namespace internal

    /// Functor that maps a \ref ArrayDomain coordinate into linear numbers the
    /// way C++ arrays work.
    struct LinearizeArrayDomainCpp
    {
        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDomain<Dim>& size) -> std::size_t
        {
            return internal::product(size);
        }

        /// \param coord coordinate in the array domain
        /// \param size total size of the array domain
        /// \return linearized index
        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDomain<Dim>& coord, const ArrayDomain<Dim>& size)
            const -> std::size_t
        {
            std::size_t address = coord[0];
            for (auto i = 1; i < Dim; i++)
            {
                address *= size[i];
                address += coord[i];
            }
            return address;
        }
    };

    /// Functor that maps a \ref ArrayDomain coordinate into linear numbers the
    /// way Fortran arrays work.
    struct LinearizeArrayDomainFortran
    {
        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDomain<Dim>& size) -> std::size_t
        {
            return internal::product(size);
        }

        /// \param coord coordinate in the array domain
        /// \param size total size of the array domain
        /// \return linearized index
        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDomain<Dim>& coord, const ArrayDomain<Dim>& size)
            const -> std::size_t
        {
            std::size_t address = coord[Dim - 1];
            for (int i = (int) Dim - 2; i >= 0; i--)
            {
                address *= size[i];
                address += coord[i];
            }
            return address;
        }
    };

    /// Functor that maps a \ref ArrayDomain coordinate into linear numbers using
    /// the Z-order space filling curve (Morton codes).
    struct LinearizeArrayDomainMorton
    {
        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDomain<Dim>& size) const -> std::size_t
        {
            std::size_t longest = size[0];
            for (auto i = 1; i < Dim; i++)
                longest = std::max(longest, size[i]);
            const auto longestPO2 = bit_ceil(longest);
            return intPow(longestPO2, Dim);
        }

        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDomain<Dim>& coord, const ArrayDomain<Dim>&) const
            -> std::size_t
        {
            std::size_t r = 0;
            for (std::size_t bit = 0; bit < (sizeof(std::size_t) * CHAR_BIT) / Dim; bit++)
                for (std::size_t i = 0; i < Dim; i++)
                    r |= (coord[i] & (std::size_t{1} << bit)) << ((bit + 1) * (Dim - 1) - i);
            return r;
        }

    private:
        LLAMA_FN_HOST_ACC_INLINE static constexpr auto bit_ceil(std::size_t n) -> std::size_t
        {
            std::size_t r = 1;
            while (r < n)
                r <<= 1;
            return r;
        }

        LLAMA_FN_HOST_ACC_INLINE static constexpr auto intPow(std::size_t b, std::size_t e) -> std::size_t
        {
            e--;
            auto r = b;
            while (e)
            {
                r *= b;
                e--;
            }
            return r;
        }
    };
} // namespace llama::mapping

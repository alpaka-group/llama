/* Copyright 2018 Alexander Matthes
 *
 * This file is part of LLAMA.
 *
 * LLAMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * LLAMA is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with LLAMA.  If not, see <www.gnu.org/licenses/>.
 */

#pragma once

#include "../Types.hpp"

#include <climits>

namespace llama
{
    /// Functor that calculates the extent of a user domain
    struct ExtentUserDomainAdress
    {
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto
        operator()(const UserDomain<Dim> & size) const -> std::size_t
        {
            std::size_t prod = 1;
            for(auto s : size) prod *= s;
            return prod;
        }
    };

    /** Functor to get the linear position of a coordinate in the user domain
     * space if the n-dimensional domain is flattened to one dimension with the
     * last user domain index being the fastet resp. already linearized index (C
     * like). \see LinearizeUserDomainAdressLikeFortran
     * */
    struct LinearizeUserDomainAdress
    {
        /**
         * \param coord coordinate in the user domain
         * \param size total size of the user domain
         * \return linearized index
         * */
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
            const UserDomain<Dim> & coord,
            const UserDomain<Dim> & size) const -> std::size_t
        {
            std::size_t address = coord[0];
            for(auto i = 1; i < Dim; i++)
            {
                address *= size[i];
                address += coord[i];
            }
            return address;
        }
    };

    /** Functor to get the linear position of a coordinate in the user domain
     * space if the n-dimensional domain is flattened to one dimension with the
     * first user domain index being the fastet resp. already linearized index
     * (Fortran like). \tparam Dim dimension of the user domain \see
     * LinearizeUserDomainAdress
     * */
    struct LinearizeUserDomainAdressLikeFortran
    {
        /**
         * \param coord coordinate in the user domain
         * \param size total size of the user domain
         * \return linearized index
         * */
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
            const UserDomain<Dim> & coord,
            const UserDomain<Dim> & size) const -> std::size_t
        {
            std::size_t address = coord[Dim - 1];
            for(int i = (int)Dim - 2; i >= 0; i--)
            {
                address *= size[i];
                address += coord[i];
            }
            return address;
        }
    };

    struct ExtentUserDomainMorton
    {
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto
        operator()(const UserDomain<Dim> & size) const -> std::size_t
        {
            std::size_t longest = size[0];
            for(auto i = 1; i < Dim; i++) longest = std::max(longest, size[i]);
            const auto longestPO2 = bit_ceil(longest);
            return intPow(longestPO2, Dim);
        }

    private:
        // TODO: replace by std::bit_ceil in C++20
        static constexpr auto bit_ceil(std::size_t n) -> std::size_t
        {
            std::size_t r = 1;
            while(r < n) r <<= 1;
            return r;
        }

        static constexpr auto intPow(std::size_t b, std::size_t e)
            -> std::size_t
        {
            e--;
            auto r = b;
            while(e)
            {
                r *= b;
                e--;
            }
            return r;
        }
    };

    struct LinearizeUserDomainMorton
    {
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto
        operator()(const UserDomain<Dim> & coord, const UserDomain<Dim> &) const
            -> std::size_t
        {
            std::size_t r = 0;
            for(std::size_t bit = 0;
                bit < (sizeof(std::size_t) * CHAR_BIT) / Dim;
                bit++)
                for(std::size_t i = 0; i < Dim; i++)
                    r |= (coord[i] & (std::size_t{1} << bit))
                        << ((bit + 1) * (Dim - 1) - i);
            return r;
        }
    };
}

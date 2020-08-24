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

#include "Types.hpp"

#include <boost/iterator/iterator_facade.hpp>

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

    template<std::size_t Dim>
    struct UserDomainCoordIterator :
            boost::iterator_facade<
                UserDomainCoordIterator<Dim>,
                UserDomain<Dim>,
                boost::forward_traversal_tag,
                UserDomain<Dim>>
    {
        UserDomainCoordIterator(UserDomain<Dim> size, UserDomain<Dim> current) :
                size(size), current(current)
        {}

        auto dereference() const -> UserDomain<Dim>
        {
            return current;
        }

        void increment()
        {
            for(auto i = (int)Dim - 1; i >= 0; i--)
            {
                current[i]++;
                if(current[i] != size[i])
                    return;
                current[i] = 0;
            }
            // we reached the end
            current[0] = size[0];
        }

        auto equal(const UserDomainCoordIterator & other) const -> bool
        {
            return size == other.size && current == other.current;
        }

        UserDomain<Dim> size;
        UserDomain<Dim> current;
    };

    template<std::size_t Dim>
    struct UserDomainCoordRange
    {
        UserDomainCoordRange(UserDomain<Dim> size) : size(size) {}

        auto begin() const -> UserDomainCoordIterator<Dim>
        {
            return {size, UserDomain<Dim>{}};
        }

        auto end() const -> UserDomainCoordIterator<Dim>
        {
            UserDomain<Dim> e{};
            e[0] = size[0];
            return {size, e};
        }

    private:
        UserDomain<Dim> size;
    };
}

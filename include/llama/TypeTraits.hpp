/* Copyright 2019 Rene Widera
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

#include <type_traits>

namespace llama
{

    /** remove const end reference
     *
     * This trait is equal to the C++20 remove_cvref.
     *
     * \attention This trait will not collide with the C++20 trait since it
     *            is defined in the llama namespace.
     */
    template< typename T >
    struct remove_cvref
    {
        using type = typename std::remove_cv<
            typename std::remove_reference< T >::type
        >::type;
    };

    //! short hand for remove_cvref
    template< typename T >
    using remove_cvref_t = typename remove_cvref< T >::type;

} // namespace llama

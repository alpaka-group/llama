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

#include "View.hpp"
#include "preprocessor/macros.hpp"

namespace llama
{
    /** Struct which acts like a \ref View, but shows only a smaller and/or
     * shifted part of a parental (virtual) view. A virtual view does not hold
     * any memory itself but has a reference to the parent view (and its
     * memory). \tparam T_ParentViewType type of the parent, can also be another
     * VirtualView
     */
    template<typename T_ParentViewType>
    struct VirtualView
    {
        /// type of parent view
        using ParentView = T_ParentViewType;
        /// blob type, gotten from parent view
        using BlobType = typename ParentView::BlobType;
        /// mapping type, gotten from parent view
        using Mapping = typename ParentView::Mapping;
        /// VirtualDatum type, gotten from parent view
        using VirtualDatumType = typename ParentView::VirtualDatumType;

        // VirtualView() = delete;
        VirtualView(VirtualView const &) = default;
        VirtualView(VirtualView &&) = default; // TODO noexcept
        ~VirtualView() = default;

        /** Unlike a \ref View, a VirtualView can be created without a factory
         *  directly from a parent view.
         * \param parentView a reference to the parental view. Meaning, the
         * parental view should not get out of scope before the virtual view.
         * \param position shifted position relative to the parental view
         * \param size size of the virtual view
         */
        LLAMA_NO_HOST_ACC_WARNING
        LLAMA_FN_HOST_ACC_INLINE
        VirtualView(
            ParentView & parentView,
            typename Mapping::UserDomain const position,
            typename Mapping::UserDomain const size) :
                parentView(parentView),
                position(position),
                size(size),
                blob(parentView.blob),
                mapping(parentView.mapping)
        {}

        /** Explicit access function taking the datum domain as tree index
         *  coordinate template arguments and the user domain as runtime
         * parameter. The operator() overloadings should be preferred as they
         * show a more array of struct like interface using \ref VirtualDatum.
         * \tparam T_coords... tree index coordinate
         * \param userDomain user domain as \ref UserDomain
         * \return reference to element
         */
        LLAMA_NO_HOST_ACC_WARNING
        template<std::size_t... T_coords>
        LLAMA_FN_HOST_ACC_INLINE auto
        accessor(typename Mapping::UserDomain const userDomain) -> auto &
        {
            return parentView.template accessor<T_coords...>(
                userDomain + position);
        }

        /** Explicit access function taking the datum domain as UID type list
         *  template arguments and the user domain as runtime parameter.
         *  The operator() overloadings should be preferred as they show a more
         *  array of struct like interface using \ref VirtualDatum.
         * \tparam T_UIDs... UID type list
         * \param userDomain user domain as \ref UserDomain
         * \return reference to element
         */
        LLAMA_NO_HOST_ACC_WARNING
        template<typename... T_UIDs>
        LLAMA_FN_HOST_ACC_INLINE auto
        accessor(typename Mapping::UserDomain const userDomain) -> auto &
        {
            return parentView.template accessor<T_UIDs...>(
                userDomain + position);
        }

        /** Operator overloading to reverse the order of compile time (datum
         * domain) and run time (user domain) parameter with a helper object
         *  (\ref VirtualDatum). Should be favoured to access data because of
         * the more array of struct like interface and the handy intermediate
         *  \ref VirtualDatum object.
         * \param userDomain user domain as \ref UserDomain
         * \return \ref VirtualDatum with bound user domain, which can be used
         * to access the datum domain
         */
        auto LLAMA_FN_HOST_ACC_INLINE operator()(
            typename Mapping::UserDomain const userDomain) -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(userDomain + position);
        }

        /** Operator overloading to reverse the order of compile time (datum
         * domain) and run time (user domain) parameter with a helper object
         *  (\ref VirtualDatum). Should be favoured to access data because of
         * the more array of struct like interface and the handy intermediate
         *  \ref VirtualDatum object.
         * \tparam T_Coord... types of user domain coordinates
         * \param coord user domain as list of numbers
         * \return \ref VirtualDatum with bound user domain, which can be used
         * to access the datum domain
         */
        template<typename... T_Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(T_Coord... coord)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(
                typename Mapping::UserDomain{coord...} + position);
        }

        template<typename... T_Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(T_Coord... coord) const
            -> const VirtualDatumType // TODO const value
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(
                typename Mapping::UserDomain{coord...} + position);
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto operator()(std::size_t coord = 0) -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(typename Mapping::UserDomain{coord} + position);
        }

        template<std::size_t... T_coord>
        LLAMA_FN_HOST_ACC_INLINE auto
        operator()(DatumCoord<T_coord...> && dc = DatumCoord<T_coord...>())
            -> auto &
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<T_coord...>(
                userDomainZero<Mapping::UserDomain::count>());
        }

        /// reference to parental view
        ParentView & parentView;
        /// shited position in parental view
        const typename Mapping::UserDomain position;
        /// shown size
        const typename Mapping::UserDomain size;
        /// reference to blob object of parent
        Array<BlobType, Mapping::blobCount> & blob;
        /// reference to mapping of parent
        const Mapping & mapping;
    };

} // namespace llama

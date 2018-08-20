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

namespace llama
{

template<
    typename T_ParentViewType
>
struct VirtualView
{
    using ParentView = T_ParentViewType;
    using BlobType = typename ParentView::BlobType;
    using Mapping = typename ParentView::Mapping;
    using VirtualDatumType = typename ParentView::VirtualDatumType;

    VirtualView() = delete;
    VirtualView( VirtualView const & ) = default;
    VirtualView( VirtualView && ) = default;
    ~VirtualView( ) = default;

    LLAMA_NO_HOST_ACC_WARNING
    LLAMA_FN_HOST_ACC_INLINE
    VirtualView(
        ParentView & parentView,
        typename Mapping::UserDomain const position,
        typename Mapping::UserDomain const size
    ) :
        parentView( parentView ),
        position( position ),
        size( size ),
        blob( parentView.blob ),
        mapping( parentView.mapping )
    { }

    LLAMA_NO_HOST_ACC_WARNING
    template< std::size_t... T_coords >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    accessor( typename Mapping::UserDomain const userDomain )
    -> GetType<
        typename Mapping::DatumDomain,
        T_coords...
    > &
    {
        return parentView.template accessor< T_coords... >(
            userDomain + position
        );
    }

    LLAMA_NO_HOST_ACC_WARNING
    template< typename... T_UIDs >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    accessor( typename Mapping::UserDomain const userDomain )
    -> GetTypeFromDatumCoord<
        typename Mapping::DatumDomain,
        GetCoordFromUID<
            typename Mapping::DatumDomain,
            T_UIDs...
        >
    > &
    {
        return parentView.template accessor< T_UIDs... >(
            userDomain + position
        );
    }

    auto
    LLAMA_FN_HOST_ACC_INLINE
    operator()( typename Mapping::UserDomain const userDomain )
    -> VirtualDatumType
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return parentView( userDomain + position );
    }

    template< typename... T_Coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Coord... coord )
    -> VirtualDatumType
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return parentView(
            typename Mapping::UserDomain{ coord... } + position
        );
    }

    template< typename... T_Coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( T_Coord... coord ) const
    -> const VirtualDatumType
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return parentView(
            typename Mapping::UserDomain{ coord... } + position
        );
    }

    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( std::size_t coord = 0 )
    -> VirtualDatumType
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return parentView(
            typename Mapping::UserDomain{ coord } + position
        );
    }

    template< std::size_t... T_coord >
    LLAMA_FN_HOST_ACC_INLINE
    auto
    operator()( DatumCoord< T_coord... > && dc= DatumCoord< T_coord... >() )
    -> GetType<
        typename Mapping::DatumDomain,
        T_coord...
    > &
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        return accessor< T_coord... >(
            userDomainZero< Mapping::UserDomain::count >()
        );
    }

    ParentView & parentView;
    const typename Mapping::UserDomain position;
    const typename Mapping::UserDomain size;
    Array<
        BlobType,
        Mapping::blobCount
    > & blob;
    const Mapping & mapping;
};

} // namespace llama

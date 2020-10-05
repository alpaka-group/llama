// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "DatumCoord.hpp"
#include "Functions.hpp"

#include <type_traits>

namespace llama
{
    namespace internal
    {
        template <typename T, std::size_t... Coords, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(T, DatumCoord<Coords...> coord, Functor&& functor)
        {
            functor(coord);
        };

        template <typename... Children, std::size_t... Coords, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE void applyFunctorForEachLeaf(
            DatumStruct<Children...>,
            DatumCoord<Coords...>,
            Functor&& functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof...(Children)>>([&](auto i) {
                constexpr auto childIndex = decltype(i)::value;
                using DatumElement = boost::mp11::mp_at_c<DatumStruct<Children...>, childIndex>;

                LLAMA_FORCE_INLINE_RECURSIVE
                applyFunctorForEachLeaf(
                    GetDatumElementType<DatumElement>{},
                    llama::DatumCoord<Coords..., childIndex>{},
                    std::forward<Functor>(functor));
            });
        }
    } // namespace internal

    /// Iterates over the datum domain tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have
    /// `operator()` with a template parameter for the \ref DatumCoord in the
    /// datum domain tree.
    /// \param baseCoord \ref DatumCoord at which the iteration should be
    /// started. The functor is called on elements beneath this coordinate.
    template <typename DatumDomain, typename Functor, std::size_t... Coords>
    LLAMA_FN_HOST_ACC_INLINE void forEach(Functor&& functor, DatumCoord<Coords...> baseCoord)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        internal::applyFunctorForEachLeaf(
            GetType<DatumDomain, DatumCoord<Coords...>>{},
            baseCoord,
            std::forward<Functor>(functor));
    }

    /// Iterates over the datum domain tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have
    /// `operator()` with a template parameter for the \ref DatumCoord in the
    /// datum domain tree.
    /// \param baseTags Tags used to define where the iteration should be
    /// started. The functor is called on elements beneath this coordinate.
    template <typename DatumDomain, typename Functor, typename... Tags>
    LLAMA_FN_HOST_ACC_INLINE void forEach(Functor&& functor, Tags... baseTags)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        forEach<DatumDomain>(std::forward<Functor>(functor), GetCoordFromTags<DatumDomain, Tags...>{});
    }
} // namespace llama

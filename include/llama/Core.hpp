// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "DatumCoord.hpp"

#include <boost/core/demangle.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mp11.hpp>
#include <iostream>
#include <type_traits>

namespace llama
{
    /// Anonymous naming for a \ref DatumElement. Especially used for a \ref
    /// DatumArray.
    struct NoName
    {
    };

    /// The run-time specified array domain.
    /// \tparam Dim compile time dimensionality of the array domain
    template <std::size_t Dim>
    struct ArrayDomain : Array<std::size_t, Dim>
    {
    };

    static_assert(
        std::is_trivially_default_constructible_v<ArrayDomain<1>>); // so ArrayDomain<1>{} will produce a zeroed
                                                                    // coord. Should hold for all dimensions,
                                                                    // but just checking for <1> here.

    template <typename... Args>
    ArrayDomain(Args...) -> ArrayDomain<sizeof...(Args)>;
} // namespace llama

template <size_t N>
struct std::tuple_size<llama::ArrayDomain<N>> : std::integral_constant<size_t, N>
{
};

template <size_t I, size_t N>
struct std::tuple_element<I, llama::ArrayDomain<N>>
{
    using type = size_t;
};

namespace llama
{
    /// A type list of \ref DatumElement which may be used to define a datum domain.
    template <typename... Leaves>
    struct DatumStruct
    {
    };

    /// Shortcut alias for \ref DatumStruct.
    template <typename... Leaves>
    using DS = DatumStruct<Leaves...>;

    /// Datum domain tree node which may either be a leaf or refer to a child
    /// tree presented as another \ref DatumStruct or \ref DatumArray.
    /// \tparam Tag Name of the node. May be any type (struct, class).
    /// \tparam Type Type of the node. May be either another sub tree consisting
    /// of a nested \ref DatumStruct or \ref DatumArray or any other type making
    /// it a leaf of this type.
    template <typename Tag, typename Type>
    struct DatumElement
    {
    };

    /// Shortcut alias for \ref DatumElement.
    template <typename Identifier, typename Type>
    using DE = DatumElement<Identifier, Type>;

    /// Tag describing an index. Used to access members of a \ref DatumArray.
    template <std::size_t I>
    using Index = boost::mp11::mp_size_t<I>;

    namespace internal
    {
        template <typename ChildType, std::size_t... Is>
        auto makeDatumArray(std::index_sequence<Is...>)
        {
            return DatumStruct<DatumElement<Index<Is>, ChildType>...>{};
        }
    } // namespace internal

    /// An array of identical \ref DatumElement with \ref Index specialized on
    /// consecutive numbers. Can be used anywhere where \ref DatumStruct may
    /// used.
    /// \tparam ChildType Type to repeat. May be either a nested \ref
    /// DatumStruct or DatumArray or any other type making it an array of leaves
    /// of this type.
    /// \tparam Count Number of repetitions of ChildType.
    template <typename ChildType, std::size_t Count>
    using DatumArray = decltype(internal::makeDatumArray<ChildType>(std::make_index_sequence<Count>{}));

    /// Shortcut alias for \ref DatumArray
    template <typename ChildType, std::size_t Count>
    using DA = DatumArray<ChildType, Count>;

    struct NrAndOffset
    {
        std::size_t nr;
        std::size_t offset;

        friend auto operator==(const NrAndOffset& a, const NrAndOffset& b) -> bool
        {
            return a.nr == b.nr && a.offset == b.offset;
        }

        friend auto operator!=(const NrAndOffset& a, const NrAndOffset& b) -> bool
        {
            return !(a == b);
        }

        friend auto operator<<(std::ostream& os, const NrAndOffset& value) -> std::ostream&
        {
            return os << "NrAndOffset{" << value.nr << ", " << value.offset << "}";
        }
    };

    /// Get the tag from a \ref DatumElement.
    template <typename DatumElement>
    using GetDatumElementTag = boost::mp11::mp_first<DatumElement>;

    /// Get the type from a \ref DatumElement.
    template <typename DatumElement>
    using GetDatumElementType = boost::mp11::mp_second<DatumElement>;

    template <typename T>
    static constexpr auto sizeOf = sizeof(T);

    /// The size a datum domain if it would be a normal struct.
    template <typename... DatumElements>
    static constexpr auto sizeOf<DatumStruct<DatumElements...>> = (sizeOf<GetDatumElementType<DatumElements>> + ...);

    namespace internal
    {
        template <typename T>
        constexpr auto offsetOfImpl(T*, DatumCoord<>)
        {
            return 0;
        }

        template <typename... DatumElements, std::size_t FirstCoord, std::size_t... Coords>
        constexpr auto offsetOfImpl(DatumStruct<DatumElements...>*, DatumCoord<FirstCoord, Coords...>)
        {
            std::size_t acc = 0;
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<FirstCoord>>([&](auto i) constexpr {
                constexpr auto index = decltype(i)::value;
                using Element = boost::mp11::mp_at_c<DatumStruct<DatumElements...>, index>;
                acc += sizeOf<GetDatumElementType<Element>>;
            });

            using Element = boost::mp11::mp_at_c<DatumStruct<DatumElements...>, FirstCoord>;
            acc += offsetOfImpl((GetDatumElementType<Element>*) nullptr, DatumCoord<Coords...>{});

            return acc;
        }
    } // namespace internal

    /// The byte offset of an element in a datum domain if it would be a normal
    /// struct.
    /// \tparam DatumDomain Datum domain tree.
    /// \tparam DatumCoord Datum coordinate of an element indatum domain tree.
    template <typename DatumDomain, typename DatumCoord>
    inline constexpr std::size_t offsetOf = internal::offsetOfImpl((DatumDomain*) nullptr, DatumCoord{});

    template <typename T>
    inline constexpr auto isDatumStruct = false;

    template <typename... DatumElements>
    inline constexpr auto isDatumStruct<DatumStruct<DatumElements...>> = true;

    namespace internal
    {
        template <typename CurrTag, typename DatumDomain, typename DatumCoord>
        struct GetTagsImpl;

        template <typename CurrTag, typename... DatumElements, std::size_t FirstCoord, std::size_t... Coords>
        struct GetTagsImpl<CurrTag, DatumStruct<DatumElements...>, DatumCoord<FirstCoord, Coords...>>
        {
            using DatumElement = boost::mp11::mp_at_c<boost::mp11::mp_list<DatumElements...>, FirstCoord>;
            using ChildTag = GetDatumElementTag<DatumElement>;
            using ChildType = GetDatumElementType<DatumElement>;
            using type = boost::mp11::
                mp_push_front<typename GetTagsImpl<ChildTag, ChildType, DatumCoord<Coords...>>::type, CurrTag>;
        };

        template <typename CurrTag, typename T>
        struct GetTagsImpl<CurrTag, T, DatumCoord<>>
        {
            using type = boost::mp11::mp_list<CurrTag>;
        };
    } // namespace internal

    /// Get the tags of all \ref DatumElement from the root of the datum domain
    /// tree until to the node identified by \ref DatumCoord.
    template <typename DatumDomain, typename DatumCoord>
    using GetTags = typename internal::GetTagsImpl<NoName, DatumDomain, DatumCoord>::type;

    /// Get the tag of the \ref DatumElement at a \ref DatumCoord inside the
    /// datum domain tree.
    template <typename DatumDomain, typename DatumCoord>
    using GetTag = boost::mp11::mp_back<GetTags<DatumDomain, DatumCoord>>;

    /// Is true if, starting at two coordinates in two datum domains, all
    /// subsequent nodes in the datum domain tree have the same tag.
    /// \tparam DatumDomainA First user domain.
    /// \tparam LocalA \ref DatumCoord based on StartA along which the tags are
    /// compared.
    /// \tparam DatumDomainB second user domain
    /// \tparam LocalB \ref DatumCoord based on StartB along which the tags are
    /// compared.
    template <typename DatumDomainA, typename LocalA, typename DatumDomainB, typename LocalB>
    inline constexpr auto hasSameTags = []() constexpr
    {
        if constexpr (LocalA::size != LocalB::size)
            return false;
        else if constexpr (LocalA::size == 0 && LocalB::size == 0)
            return true;
        else
            return std::is_same_v<GetTags<DatumDomainA, LocalA>, GetTags<DatumDomainB, LocalB>>;
    }
    ();

    namespace internal
    {
        template <typename DatumDomain, typename DatumCoord, typename... Tags>
        struct GetCoordFromTagsImpl
        {
            static_assert(boost::mp11::mp_size<DatumDomain>::value != 0, "Tag combination is not valid");
        };

        template <typename... DatumElements, std::size_t... ResultCoords, typename FirstTag, typename... Tags>
        struct GetCoordFromTagsImpl<DatumStruct<DatumElements...>, DatumCoord<ResultCoords...>, FirstTag, Tags...>
        {
            template <typename DatumElement>
            struct HasTag : std::is_same<GetDatumElementTag<DatumElement>, FirstTag>
            {
            };

            static constexpr auto tagIndex
                = boost::mp11::mp_find_if<boost::mp11::mp_list<DatumElements...>, HasTag>::value;
            static_assert(tagIndex < sizeof...(DatumElements), "FirstTag was not found inside this DatumStruct");

            using ChildType = GetDatumElementType<boost::mp11::mp_at_c<DatumStruct<DatumElements...>, tagIndex>>;

            using type = typename GetCoordFromTagsImpl<ChildType, DatumCoord<ResultCoords..., tagIndex>, Tags...>::type;
        };

        template <typename DatumDomain, typename DatumCoord>
        struct GetCoordFromTagsImpl<DatumDomain, DatumCoord>
        {
            using type = DatumCoord;
        };
    } // namespace internal

    /// Converts a series of tags navigating down a datum domain into a \ref
    /// DatumCoord.
    template <typename DatumDomain, typename... Tags>
    using GetCoordFromTags = typename internal::GetCoordFromTagsImpl<DatumDomain, DatumCoord<>, Tags...>::type;

    namespace internal
    {
        template <typename DatumDomain, typename... DatumCoordOrTags>
        struct GetTypeImpl;

        template <typename... Children, std::size_t HeadCoord, std::size_t... TailCoords>
        struct GetTypeImpl<DatumStruct<Children...>, DatumCoord<HeadCoord, TailCoords...>>
        {
            using ChildType = GetDatumElementType<boost::mp11::mp_at_c<DatumStruct<Children...>, HeadCoord>>;
            using type = typename GetTypeImpl<ChildType, DatumCoord<TailCoords...>>::type;
        };

        template <typename T>
        struct GetTypeImpl<T, DatumCoord<>>
        {
            using type = T;
        };

        template <typename DatumDomain, typename... DatumCoordOrTags>
        struct GetTypeImpl
        {
            using type = typename GetTypeImpl<DatumDomain, GetCoordFromTags<DatumDomain, DatumCoordOrTags...>>::type;
        };
    } // namespace internal

    /// Returns the type of a node in a datum domain tree identified by a given
    /// \ref DatumCoord or a series of tags.
    template <typename DatumDomain, typename... DatumCoordOrTags>
    using GetType = typename internal::GetTypeImpl<DatumDomain, DatumCoordOrTags...>::type;

    namespace internal
    {
        template <typename DatumDomain, typename BaseDatumCoord, typename... Tags>
        struct GetCoordFromTagsRelativeImpl
        {
            using AbsolutCoord = typename internal::
                GetCoordFromTagsImpl<GetType<DatumDomain, BaseDatumCoord>, BaseDatumCoord, Tags...>::type;
            // Only returning the datum coord relative to BaseDatumCoord
            using type = DatumCoordFromList<boost::mp11::mp_drop_c<typename AbsolutCoord::List, BaseDatumCoord::size>>;
        };
    } // namespace internal

    /// Converts a series of tags navigating down a datum domain, starting at a
    /// given \ref DatumCoord, into a \ref DatumCoord.
    template <typename DatumDomain, typename BaseDatumCoord, typename... Tags>
    using GetCoordFromTagsRelative =
        typename internal::GetCoordFromTagsRelativeImpl<DatumDomain, BaseDatumCoord, Tags...>::type;

    namespace internal
    {
        template <typename T, std::size_t... Coords, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE constexpr void forEachImpl(T, DatumCoord<Coords...> coord, Functor&& functor)
        {
            functor(coord);
        };

        template <typename... Children, std::size_t... Coords, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE constexpr void forEachImpl(
            DatumStruct<Children...>,
            DatumCoord<Coords...>,
            Functor&& functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof...(Children)>>([&](auto i) {
                constexpr auto childIndex = decltype(i)::value;
                using DatumElement = boost::mp11::mp_at_c<DatumStruct<Children...>, childIndex>;

                LLAMA_FORCE_INLINE_RECURSIVE
                forEachImpl(
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
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEach(Functor&& functor, DatumCoord<Coords...> baseCoord)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        internal::forEachImpl(GetType<DatumDomain, DatumCoord<Coords...>>{}, baseCoord, std::forward<Functor>(functor));
    }

    /// Iterates over the datum domain tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have
    /// `operator()` with a template parameter for the \ref DatumCoord in the
    /// datum domain tree.
    /// \param baseTags Tags used to define where the iteration should be
    /// started. The functor is called on elements beneath this coordinate.
    template <typename DatumDomain, typename Functor, typename... Tags>
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEach(Functor&& functor, Tags... baseTags)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        forEach<DatumDomain>(std::forward<Functor>(functor), GetCoordFromTags<DatumDomain, Tags...>{});
    }

    namespace internal
    {
        template <typename T>
        struct FlattenDatumDomainImpl
        {
            using type = boost::mp11::mp_list<T>;
        };

        template <typename... Elements>
        struct FlattenDatumDomainImpl<DatumStruct<Elements...>>
        {
            using type = boost::mp11::mp_append<typename FlattenDatumDomainImpl<GetDatumElementType<Elements>>::type...>;
        };
    } // namespace internal

    template <typename DatumDomain>
    using FlattenDatumDomain = typename internal::FlattenDatumDomainImpl<DatumDomain>::type;

    template <typename S>
    auto structName(S) -> std::string
    {
        auto s = boost::core::demangle(typeid(S).name());
        if (const auto pos = s.rfind(':'); pos != std::string::npos)
            s = s.substr(pos + 1);
        return s;
    }
} // namespace llama

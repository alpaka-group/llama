// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "DatumCoord.hpp"

#include <boost/core/demangle.hpp>
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

    // FIXME: the documented functionality currently only works through llama::DE, because the Type is not expanded in
    // case of arrays
    /// Datum domain tree node which may either be a leaf or refer to a child
    /// tree presented as another \ref DatumStruct or \ref DatumArray.
    /// \tparam Tag Name of the node. May be any type (struct, class).
    /// \tparam Type Type of the node. May be one of three cases. 1. another sub tree consisting
    /// of a nested \ref DatumStruct. 2. an array of any type, in which case a DatumStruct with as many \ref
    /// DatumElement as the array size is created, named \ref Index specialized on consecutive numbers.
    /// 3. A scalar type different from \ref DatumStruct, making this node a leaf of this type.
    template <typename Tag, typename Type>
    struct DatumElement
    {
        static_assert(!std::is_array_v<Type>, "DatumElement does not support array types. Please use DE instead!");
    };

    namespace internal
    {
        template <typename ChildType, std::size_t... Is>
        auto makeDatumArray(std::index_sequence<Is...>)
        {
            return DatumStruct<DatumElement<DatumCoord<Is>, ChildType>...>{};
        }
    } // namespace internal

    template <typename T>
    struct MakeDatumElementType
    {
        using type = T;
    };

    template <typename ChildType, std::size_t Count>
    struct MakeDatumElementType<ChildType[Count]>
    {
        using type = decltype(internal::makeDatumArray<typename MakeDatumElementType<ChildType>::type>(
            std::make_index_sequence<Count>{}));
    };

    /// Shortcut alias for \ref DatumElement.
    template <typename Identifier, typename Type>
    using DE = DatumElement<Identifier, typename MakeDatumElementType<Type>::type>;

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
        LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaveImpl(T*, DatumCoord<Coords...> coord, Functor&& functor)
        {
            functor(coord);
        };

        template <typename... Children, std::size_t... Coords, typename Functor>
        LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaveImpl(
            DatumStruct<Children...>*,
            DatumCoord<Coords...>,
            Functor&& functor)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof...(Children)>>([&](auto i) {
                constexpr auto childIndex = decltype(i)::value;
                using DatumElement = boost::mp11::mp_at_c<DatumStruct<Children...>, childIndex>;

                LLAMA_FORCE_INLINE_RECURSIVE
                forEachLeaveImpl(
                    static_cast<GetDatumElementType<DatumElement>*>(nullptr),
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
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeave(Functor&& functor, DatumCoord<Coords...> baseCoord)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        internal::forEachLeaveImpl(
            static_cast<GetType<DatumDomain, DatumCoord<Coords...>>*>(nullptr),
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
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeave(Functor&& functor, Tags... baseTags)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        forEachLeave<DatumDomain>(std::forward<Functor>(functor), GetCoordFromTags<DatumDomain, Tags...>{});
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
            using type
                = boost::mp11::mp_append<typename FlattenDatumDomainImpl<GetDatumElementType<Elements>>::type...>;
        };

        // TODO: MSVC fails to compile if we move this function into an IILE at the callsite
        template <typename DatumDomain, typename DatumCoord>
        constexpr auto flatDatumCoordImpl()
        {
            std::size_t c = 0;
            forEachLeave<DatumDomain>([&](auto coord) {
                if constexpr (DatumCoordCommonPrefixIsBigger<DatumCoord, decltype(coord)>)
                    c++;
            });
            return c;
        }
    } // namespace internal

    template <typename DatumDomain>
    using FlattenDatumDomain = typename internal::FlattenDatumDomainImpl<DatumDomain>::type;

    template <typename DatumDomain, typename DatumCoord>
    inline constexpr std::size_t flatDatumCoord = internal::flatDatumCoordImpl<DatumDomain, DatumCoord>();

    namespace internal
    {
        constexpr void roundUpToMultiple(std::size_t& value, std::size_t multiple)
        {
            value = ((value + multiple - 1) / multiple) * multiple;
        }

        // TODO: MSVC fails to compile if we move this function into an IILE at the callsite
        template <bool Align, typename... DatumElements>
        constexpr auto sizeOfDatumStructImpl()
        {
            using namespace boost::mp11;

            std::size_t size = 0;
            using FlatDD = FlattenDatumDomain<DatumStruct<DatumElements...>>;
            mp_for_each<mp_transform<mp_identity, FlatDD>>([&](auto e) constexpr {
                using T = typename decltype(e)::type;
                if constexpr (Align)
                    roundUpToMultiple(size, alignof(T));
                size += sizeof(T);
            });

            // final padding, so next struct can start right away
            if constexpr (Align)
                roundUpToMultiple(size, alignof(mp_first<FlatDD>));
            return size;
        }
    } // namespace internal

    template <typename T, bool Align = false>
    inline constexpr std::size_t sizeOf = sizeof(T);

    /// The size a datum domain if it would be a normal struct.
    template <typename... DatumElements, bool Align>
    inline constexpr std::size_t sizeOf<DatumStruct<DatumElements...>, Align> = internal::
        sizeOfDatumStructImpl<Align, DatumElements...>();

    /// The byte offset of an element in a datum domain if it would be a normal struct.
    /// \tparam DatumDomain Datum domain tree.
    /// \tparam DatumCoord Datum coordinate of an element indatum domain tree.
    template <typename DatumDomain, typename DatumCoord, bool Align = false>
    inline constexpr std::size_t offsetOf = []() constexpr
    {
        using namespace boost::mp11;

        using FlatDD = FlattenDatumDomain<DatumDomain>;
        constexpr auto flatCoord = flatDatumCoord<DatumDomain, DatumCoord>;

        std::size_t offset = 0;
        mp_for_each<mp_iota_c<flatCoord>>([&](auto i) constexpr {
            using T = mp_at<FlatDD, decltype(i)>;
            if constexpr (Align)
                internal::roundUpToMultiple(offset, alignof(T));
            offset += sizeof(T);
        });
        if constexpr (Align)
            internal::roundUpToMultiple(offset, alignof(mp_at_c<FlatDD, flatCoord>));
        return offset;
    }
    ();


    template <typename S>
    auto structName(S) -> std::string
    {
        auto s = boost::core::demangle(typeid(S).name());
        if (const auto pos = s.rfind(':'); pos != std::string::npos)
            s = s.substr(pos + 1);
        return s;
    }
} // namespace llama

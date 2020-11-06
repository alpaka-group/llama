// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "DatumCoord.hpp"
#include "Types.hpp"

#include <boost/core/demangle.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/mp11.hpp>
#include <type_traits>

namespace llama
{
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
    /// \tparam ElementCoords... Components of a coordinate of an element in
    /// datum domain tree.
    template <typename DatumDomain, std::size_t... ElementCoords>
    inline constexpr std::size_t offsetOf
        = internal::offsetOfImpl((DatumDomain*) nullptr, DatumCoord<ElementCoords...>{});

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

    /// Iterator supporting \ref ArrayDomainIndexRange.
    template <std::size_t Dim>
    struct ArrayDomainIndexIterator
        : boost::iterator_facade<
              ArrayDomainIndexIterator<Dim>,
              ArrayDomain<Dim>,
              std::forward_iterator_tag,
              ArrayDomain<Dim>>
    {
        ArrayDomainIndexIterator() = default;

        ArrayDomainIndexIterator(ArrayDomain<Dim> size, ArrayDomain<Dim> current) : size(size), current(current)
        {
        }

    private:
        friend class boost::iterator_core_access;

        auto dereference() const -> ArrayDomain<Dim>
        {
            return current;
        }

        void increment()
        {
            for (auto i = (int) Dim - 1; i >= 0; i--)
            {
                current[i]++;
                if (current[i] != size[i])
                    return;
                current[i] = 0;
            }
            // we reached the end
            current[0] = size[0];
        }

        auto equal(const ArrayDomainIndexIterator& other) const -> bool
        {
            return size == other.size && current == other.current;
        }

        ArrayDomain<Dim> size;
        ArrayDomain<Dim> current;
    };

    /// Range allowing to iterate over all indices in a \ref ArrayDomain.
    template <std::size_t Dim>
    struct ArrayDomainIndexRange
    {
        ArrayDomainIndexRange(ArrayDomain<Dim> size) : size(size)
        {
        }

        auto begin() const -> ArrayDomainIndexIterator<Dim>
        {
            return {size, ArrayDomain<Dim>{}};
        }

        auto end() const -> ArrayDomainIndexIterator<Dim>
        {
            ArrayDomain<Dim> e{};
            e[0] = size[0];
            return {size, e};
        }

    private:
        ArrayDomain<Dim> size;
    };

    template <typename S>
    auto structName(S) -> std::string
    {
        auto s = boost::core::demangle(typeid(S).name());
        if (const auto pos = s.rfind(':'); pos != std::string::npos)
            s = s.substr(pos + 1);
        return s;
    }
} // namespace llama

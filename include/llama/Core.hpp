// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "RecordCoord.hpp"

#include <boost/core/demangle.hpp>
#include <boost/mp11.hpp>
#include <iostream>
#include <type_traits>

namespace llama
{
    /// Anonymous naming for a \ref Field.
    struct NoName
    {
    };

    /// The run-time specified array dimensions.
    /// \tparam Dim Compile-time number of dimensions.
    template <std::size_t Dim>
    struct ArrayDims : Array<std::size_t, Dim>
    {
    };

    static_assert(std::is_trivially_default_constructible_v<ArrayDims<1>>); // so ArrayDims<1>{} will produce a zeroed
                                                                            // coord. Should hold for all dimensions,
                                                                            // but just checking for <1> here.

    template <typename... Args>
    ArrayDims(Args...) -> ArrayDims<sizeof...(Args)>;
} // namespace llama

template <size_t N>
struct std::tuple_size<llama::ArrayDims<N>> : std::integral_constant<size_t, N>
{
};

template <size_t I, size_t N>
struct std::tuple_element<I, llama::ArrayDims<N>>
{
    using type = size_t;
};

namespace llama
{
    /// A type list of \ref Field which may be used to define a record dimension.
    template <typename... Fields>
    struct Record
    {
    };

    /// @brief Tells whether the given type is allowed as a field type in LLAMA. Such types need to be trivially
    /// constructible and trivially destructible.
    template <typename T>
    inline constexpr bool isAllowedFieldType
        = std::is_trivially_constructible_v<T>&& std::is_trivially_destructible_v<T>;

    /// Record dimension tree node which may either be a leaf or refer to a child tree presented as another \ref
    /// Record.
    /// \tparam Tag Name of the node. May be any type (struct, class).
    /// \tparam Type Type of the node. May be one of three cases. 1. another sub tree consisting of a nested \ref
    /// Record. 2. an array of static size of any type, in which case a Record with as many \ref Field as the array
    /// size is created, named \ref RecordCoord specialized on consecutive numbers I. 3. A scalar type different from
    /// \ref Record, making this node a leaf of this type.
    template <typename Tag, typename Type>
    struct Field
    {
        static_assert(isAllowedFieldType<Type>, "This field's type is not allowed");
    };

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

    /// Get the tag from a \ref Field.
    template <typename Field>
    using GetFieldTag = boost::mp11::mp_first<Field>;

    namespace internal
    {
        template <typename ChildType, std::size_t... Is>
        auto makeRecordArray(std::index_sequence<Is...>)
        {
            return Record<Field<RecordCoord<Is>, ChildType>...>{};
        }

        template <typename T>
        struct ArrayToRecord
        {
            using type = T;
        };

        template <typename ChildType, std::size_t Count>
        struct ArrayToRecord<ChildType[Count]>
        {
            using type = decltype(internal::makeRecordArray<typename ArrayToRecord<ChildType>::type>(
                std::make_index_sequence<Count>{}));
        };
    } // namespace internal

    /// Get the type from a \ref Field.
    template <typename Field>
    using GetFieldType = typename internal::ArrayToRecord<boost::mp11::mp_second<Field>>::type;

    template <typename T>
    inline constexpr auto isRecord = false;

    template <typename... Fields>
    inline constexpr auto isRecord<Record<Fields...>> = true;

    namespace internal
    {
        template <typename CurrTag, typename RecordDim, typename RecordCoord>
        struct GetTagsImpl;

        template <typename CurrTag, typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
        struct GetTagsImpl<CurrTag, Record<Fields...>, RecordCoord<FirstCoord, Coords...>>
        {
            using Field = boost::mp11::mp_at_c<boost::mp11::mp_list<Fields...>, FirstCoord>;
            using ChildTag = GetFieldTag<Field>;
            using ChildType = GetFieldType<Field>;
            using type = boost::mp11::
                mp_push_front<typename GetTagsImpl<ChildTag, ChildType, RecordCoord<Coords...>>::type, CurrTag>;
        };

        template <typename CurrTag, typename T>
        struct GetTagsImpl<CurrTag, T, RecordCoord<>>
        {
            using type = boost::mp11::mp_list<CurrTag>;
        };
    } // namespace internal

    /// Get the tags of all \ref Field from the root of the record dimension
    /// tree until to the node identified by \ref RecordCoord.
    template <typename RecordDim, typename RecordCoord>
    using GetTags = typename internal::GetTagsImpl<NoName, RecordDim, RecordCoord>::type;

    /// Get the tag of the \ref Field at a \ref RecordCoord inside the
    /// record dimension tree.
    template <typename RecordDim, typename RecordCoord>
    using GetTag = boost::mp11::mp_back<GetTags<RecordDim, RecordCoord>>;

    /// Is true if, starting at two coordinates in two record dimensions, all
    /// subsequent nodes in the record dimension tree have the same tag.
    /// \tparam RecordDimA First record dimension.
    /// \tparam LocalA \ref RecordCoord based on StartA along which the tags are
    /// compared.
    /// \tparam RecordDimB second record dimension.
    /// \tparam LocalB \ref RecordCoord based on StartB along which the tags are
    /// compared.
    template <typename RecordDimA, typename LocalA, typename RecordDimB, typename LocalB>
    inline constexpr auto hasSameTags = []() constexpr
    {
        if constexpr (LocalA::size != LocalB::size)
            return false;
        else if constexpr (LocalA::size == 0 && LocalB::size == 0)
            return true;
        else
            return std::is_same_v<GetTags<RecordDimA, LocalA>, GetTags<RecordDimB, LocalB>>;
    }
    ();

    namespace internal
    {
        template <typename RecordDim, typename RecordCoord, typename... Tags>
        struct GetCoordFromTagsImpl
        {
            static_assert(boost::mp11::mp_size<RecordDim>::value != 0, "Tag combination is not valid");
        };

        template <typename... Fields, std::size_t... ResultCoords, typename FirstTag, typename... Tags>
        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<ResultCoords...>, FirstTag, Tags...>
        {
            template <typename Field>
            struct HasTag : std::is_same<GetFieldTag<Field>, FirstTag>
            {
            };

            static constexpr auto tagIndex = boost::mp11::mp_find_if<boost::mp11::mp_list<Fields...>, HasTag>::value;
            static_assert(
                tagIndex < sizeof...(Fields),
                "FirstTag was not found inside this DatumStruct. Does your datum domain contain the tag you access "
                "with?");

            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Fields...>, tagIndex>>;

            using type =
                typename GetCoordFromTagsImpl<ChildType, RecordCoord<ResultCoords..., tagIndex>, Tags...>::type;
        };

        template <typename RecordDim, typename RecordCoord>
        struct GetCoordFromTagsImpl<RecordDim, RecordCoord>
        {
            using type = RecordCoord;
        };
    } // namespace internal

    /// Converts a series of tags navigating down a record dimension into a \ref RecordCoord.
    template <typename RecordDim, typename... Tags>
    using GetCoordFromTags = typename internal::GetCoordFromTagsImpl<RecordDim, RecordCoord<>, Tags...>::type;

    namespace internal
    {
        template <typename RecordDim, typename... RecordCoordOrTags>
        struct GetTypeImpl;

        template <typename... Children, std::size_t HeadCoord, std::size_t... TailCoords>
        struct GetTypeImpl<Record<Children...>, RecordCoord<HeadCoord, TailCoords...>>
        {
            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Children...>, HeadCoord>>;
            using type = typename GetTypeImpl<ChildType, RecordCoord<TailCoords...>>::type;
        };

        template <typename T>
        struct GetTypeImpl<T, RecordCoord<>>
        {
            using type = T;
        };

        template <typename RecordDim, typename... RecordCoordOrTags>
        struct GetTypeImpl
        {
            using type = typename GetTypeImpl<RecordDim, GetCoordFromTags<RecordDim, RecordCoordOrTags...>>::type;
        };
    } // namespace internal

    /// Returns the type of a node in a record dimension tree identified by a given
    /// \ref RecordCoord or a series of tags.
    template <typename RecordDim, typename... RecordCoordOrTags>
    using GetType = typename internal::GetTypeImpl<RecordDim, RecordCoordOrTags...>::type;

    namespace internal
    {
        template <typename RecordDim, typename BaseRecordCoord, typename... Tags>
        struct GetCoordFromTagsRelativeImpl
        {
            using AbsolutCoord = typename internal::
                GetCoordFromTagsImpl<GetType<RecordDim, BaseRecordCoord>, BaseRecordCoord, Tags...>::type;
            // Only returning the record coord relative to BaseRecordCoord
            using type
                = RecordCoordFromList<boost::mp11::mp_drop_c<typename AbsolutCoord::List, BaseRecordCoord::size>>;
        };
    } // namespace internal

    /// Converts a series of tags navigating down a record dimension, starting at a
    /// given \ref RecordCoord, into a \ref RecordCoord.
    template <typename RecordDim, typename BaseRecordCoord, typename... Tags>
    using GetCoordFromTagsRelative =
        typename internal::GetCoordFromTagsRelativeImpl<RecordDim, BaseRecordCoord, Tags...>::type;

    namespace internal
    {
        template <typename RecordDim, typename RecordCoord>
        struct LeafRecordCoordsImpl;

        template <typename T, std::size_t... RCs>
        struct LeafRecordCoordsImpl<T, RecordCoord<RCs...>>
        {
            using type = boost::mp11::mp_list<RecordCoord<RCs...>>;
        };

        template <typename... Fields, std::size_t... RCs>
        struct LeafRecordCoordsImpl<Record<Fields...>, RecordCoord<RCs...>>
        {
            template <std::size_t... Is>
            static auto help(std::index_sequence<Is...>)
            {
                return boost::mp11::mp_append<
                    typename LeafRecordCoordsImpl<GetFieldType<Fields>, RecordCoord<RCs..., Is>>::type...>{};
            }
            using type = decltype(help(std::make_index_sequence<sizeof...(Fields)>{}));
        };
    } // namespace internal

    /// Returns a flat type list containing all record coordinates to all leaves of the given record dimension.
    template <typename RecordDim>
    using LeafRecordCoords = typename internal::LeafRecordCoordsImpl<RecordDim, RecordCoord<>>::type;

    namespace internal
    {
        // adapted from boost::mp11, but with LLAMA_FN_HOST_ACC_INLINE
        template <template <typename...> typename L, typename... T, typename F>
        LLAMA_FN_HOST_ACC_INLINE constexpr void mp_for_each_inlined(L<T...>, F&& f)
        {
            using A = int[sizeof...(T)];
            (void) A{((void) f(T{}), 0)...};
        }
    } // namespace internal

    /// Iterates over the record dimension tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have
    /// `operator()` with a template parameter for the \ref RecordCoord in the
    /// record dimension tree.
    /// \param baseCoord \ref RecordCoord at which the iteration should be
    /// started. The functor is called on elements beneath this coordinate.
    template <typename RecordDim, typename Functor, std::size_t... Coords>
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaf(Functor&& functor, RecordCoord<Coords...> baseCoord)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        internal::mp_for_each_inlined(
            LeafRecordCoords<GetType<RecordDim, RecordCoord<Coords...>>>{},
            [&](auto innerCoord) LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr)
            { std::forward<Functor>(functor)(cat(baseCoord, innerCoord)); });
    }

    /// Iterates over the record dimension tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have
    /// `operator()` with a template parameter for the \ref RecordCoord in the
    /// record dimension tree.
    /// \param baseTags Tags used to define where the iteration should be
    /// started. The functor is called on elements beneath this coordinate.
    template <typename RecordDim, typename Functor, typename... Tags>
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaf(Functor&& functor, Tags... baseTags)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        forEachLeaf<RecordDim>(std::forward<Functor>(functor), GetCoordFromTags<RecordDim, Tags...>{});
    }

    namespace internal
    {
        template <typename T>
        struct FlattenRecordDimImpl
        {
            using type = boost::mp11::mp_list<T>;
        };

        template <typename... Fields>
        struct FlattenRecordDimImpl<Record<Fields...>>
        {
            using type = boost::mp11::mp_append<typename FlattenRecordDimImpl<GetFieldType<Fields>>::type...>;
        };
    } // namespace internal

    /// Returns a flat type list containing all leaf field types of the given record dimension.
    template <typename RecordDim>
    using FlatRecordDim = typename internal::FlattenRecordDimImpl<RecordDim>::type;

    /// The total number of fields in the recursively expanded record dimension.
    template <typename RecordDim>
    inline constexpr std::size_t fieldCount = 1;

    template <typename... Children>
    inline constexpr std::size_t fieldCount<Record<Children...>> = (fieldCount<GetFieldType<Children>> + ... + 0);

    namespace internal
    {
        template <std::size_t I, typename RecordDim>
        inline constexpr std::size_t fieldCountBefore = 0;

        template <typename... Children>
        inline constexpr std::size_t fieldCountBefore<0, Record<Children...>> = 0;

        // recursive formulation to benefit from template instantiation memoization
        // this massively improves compilation time when this template is instantiated with a lot of different I
        template <std::size_t I, typename... Children>
        inline constexpr std::size_t fieldCountBefore<
            I,
            Record<
                Children...>> = fieldCountBefore<I - 1, Record<Children...>> + fieldCount<GetFieldType<boost::mp11::mp_at_c<Record<Children...>, I - 1>>>;

        template <typename T>
        constexpr auto flatRecordCoordImpl(T*, RecordCoord<>) -> std::size_t
        {
            return 0;
        }

        template <typename... Children, std::size_t I, std::size_t... Is>
        constexpr auto flatRecordCoordImpl(Record<Children...>*, RecordCoord<I, Is...>) -> std::size_t
        {
            return fieldCountBefore<
                       I,
                       Record<
                           Children...>> + flatRecordCoordImpl(static_cast<GetFieldType<boost::mp11::mp_at_c<Record<Children...>, I>>*>(nullptr), RecordCoord<Is...>{});
        }
    } // namespace internal

    /// The equivalent zero based index into a flat record dimension (\ref FlatRecordDim) of the given hierarchical
    /// record coordinate.
    template <typename RecordDim, typename RecordCoord>
    inline constexpr std::size_t flatRecordCoord
        = internal::flatRecordCoordImpl(static_cast<RecordDim*>(nullptr), RecordCoord{});

    namespace internal
    {
        constexpr void roundUpToMultiple(std::size_t& value, std::size_t multiple)
        {
            value = ((value + multiple - 1) / multiple) * multiple;
        }

        template <bool Align, typename TypeList>
        constexpr auto sizeOfImpl() -> std::size_t
        {
            using namespace boost::mp11;

            std::size_t size = 0;
            std::size_t maxAlign = 0;
            mp_for_each<mp_transform<mp_identity, TypeList>>([&](auto e) constexpr
                                                             {
                                                                 using T = typename decltype(e)::type;
                                                                 if constexpr (Align)
                                                                 {
                                                                     roundUpToMultiple(size, alignof(T));
                                                                     maxAlign = std::max(maxAlign, alignof(T));
                                                                 }
                                                                 size += sizeof(T);
                                                             });

            // final padding, so next struct can start right away
            if constexpr (Align)
                roundUpToMultiple(size, maxAlign);
            return size;
        }

        template <bool Align, typename TypeList, std::size_t I>
        constexpr std::size_t offsetOfImplWorkaround();

        // recursive formulation to benefit from template instantiation memoization
        // this massively improves compilation time when this template is instantiated with a lot of different I
        template <bool Align, typename TypeList, std::size_t I>
        inline constexpr std::size_t offsetOfImpl
            = offsetOfImplWorkaround<Align, TypeList, I>(); // FIXME: MSVC fails to compile an IILE here.

        template <bool Align, typename TypeList>
        inline constexpr std::size_t offsetOfImpl<Align, TypeList, 0> = 0;

        template <bool Align, typename TypeList, std::size_t I>
        constexpr std::size_t offsetOfImplWorkaround()
        {
            std::size_t offset = offsetOfImpl<Align, TypeList, I - 1> + sizeof(boost::mp11::mp_at_c<TypeList, I - 1>);
            if constexpr (Align)
                roundUpToMultiple(offset, alignof(boost::mp11::mp_at_c<TypeList, I>));
            return offset;
        }
    } // namespace internal

    /// The size of a type list if its elements would be in a normal struct.
    template <typename TypeList, bool Align>
    inline constexpr std::size_t flatSizeOf = internal::sizeOfImpl<Align, TypeList>();

    /// The size of a type T.
    template <typename T, bool Align = false>
    inline constexpr std::size_t sizeOf = sizeof(T);

    /// The size of a record dimension if its fields would be in a normal struct.
    template <typename... Fields, bool Align>
    inline constexpr std::size_t sizeOf<Record<Fields...>, Align> = flatSizeOf<FlatRecordDim<Record<Fields...>>, Align>;

    /// The byte offset of an element in a type list ifs elements would be in a normal struct.
    template <typename TypeList, std::size_t I, bool Align>
    inline constexpr std::size_t flatOffsetOf = internal::offsetOfImpl<Align, TypeList, I>;

    /// The byte offset of an element in a record dimension if it would be a normal struct.
    /// \tparam RecordDim Record dimension tree.
    /// \tparam RecordCoord Record coordinate of an element inrecord dimension tree.
    template <typename RecordDim, typename RecordCoord, bool Align = false>
    inline constexpr std::size_t offsetOf
        = flatOffsetOf<FlatRecordDim<RecordDim>, flatRecordCoord<RecordDim, RecordCoord>, Align>;

    template <typename S>
    auto structName(S = {}) -> std::string
    {
        auto s = boost::core::demangle(typeid(S).name());
        if (const auto pos = s.rfind(':'); pos != std::string::npos)
            s = s.substr(pos + 1);
        return s;
    }

    namespace internal
    {
        template <std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto popFront(ArrayDims<Dim> ad)
        {
            ArrayDims<Dim - 1> result;
            for (std::size_t i = 0; i < Dim - 1; i++)
                result[i] = ad[i + 1];
            return result;
        }
    } // namespace internal

    template <std::size_t Dim, typename Func, typename... OuterIndices>
    LLAMA_FN_HOST_ACC_INLINE void forEachADCoord(ArrayDims<Dim> adSize, Func&& func, OuterIndices... outerIndices)
    {
        for (std::size_t i = 0; i < adSize[0]; i++)
        {
            if constexpr (Dim > 1)
                forEachADCoord(internal::popFront(adSize), std::forward<Func>(func), outerIndices..., i);
            else
                std::forward<Func>(func)(ArrayDims<sizeof...(outerIndices) + 1>{outerIndices..., i});
        }
    }

    namespace internal
    {
        template <typename T>
        struct IndirectValue
        {
            T value;

            auto operator->() -> T*
            {
                return &value;
            }

            auto operator->() const -> const T*
            {
                return &value;
            }
        };
    } // namespace internal
} // namespace llama

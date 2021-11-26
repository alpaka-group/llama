// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "ArrayExtents.hpp"
#include "Meta.hpp"
#include "RecordCoord.hpp"

#include <boost/core/demangle.hpp>
#include <iostream>
#include <string>
#include <type_traits>

namespace llama
{
    /// Anonymous naming for a \ref Field.
    struct NoName
    {
    };

    /// A type list of \ref Field%s which may be used to define a record dimension.
    template<typename... Fields>
    struct Record
    {
    };

    /// @brief Tells whether the given type is allowed as a field type in LLAMA. Such types need to be trivially
    /// constructible and trivially destructible.
    template<typename T>
    inline constexpr bool isAllowedFieldType = std::is_trivially_destructible_v<T>;

    /// Record dimension tree node which may either be a leaf or refer to a child tree presented as another \ref
    /// Record.
    /// \tparam Tag Name of the node. May be any type (struct, class).
    /// \tparam Type Type of the node. May be one of three cases. 1. another sub tree consisting of a nested \ref
    /// Record. 2. an array of static size of any type, in which case a Record with as many \ref Field as the array
    /// size is created, named \ref RecordCoord specialized on consecutive numbers I. 3. A scalar type different from
    /// \ref Record, making this node a leaf of this type.
    template<typename Tag, typename Type>
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
    template<typename Field>
    using GetFieldTag = boost::mp11::mp_first<Field>;

    /// Get the type from a \ref Field.
    template<typename Field>
    using GetFieldType = boost::mp11::mp_second<Field>;

    template<typename T>
    inline constexpr auto isRecord = false;

    template<typename... Fields>
    inline constexpr auto isRecord<Record<Fields...>> = true;

    namespace internal
    {
        template<typename RecordDim, typename RecordCoord>
        struct GetTagsImpl;

        template<typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
        struct GetTagsImpl<Record<Fields...>, RecordCoord<FirstCoord, Coords...>>
        {
            using Field = boost::mp11::mp_at_c<boost::mp11::mp_list<Fields...>, FirstCoord>;
            using ChildTag = GetFieldTag<Field>;
            using ChildType = GetFieldType<Field>;
            using type
                = boost::mp11::mp_push_front<typename GetTagsImpl<ChildType, RecordCoord<Coords...>>::type, ChildTag>;
        };

        template<typename ChildType, std::size_t Count, std::size_t FirstCoord, std::size_t... Coords>
        struct GetTagsImpl<ChildType[Count], RecordCoord<FirstCoord, Coords...>>
        {
            using ChildTag = RecordCoord<FirstCoord>;
            using type
                = boost::mp11::mp_push_front<typename GetTagsImpl<ChildType, RecordCoord<Coords...>>::type, ChildTag>;
        };

        template<typename T>
        struct GetTagsImpl<T, RecordCoord<>>
        {
            using type = boost::mp11::mp_list<>;
        };
    } // namespace internal

    /// Get the tags of all \ref Field%s from the root of the record dimension tree until to the node identified by
    /// \ref RecordCoord.
    template<typename RecordDim, typename RecordCoord>
    using GetTags = typename internal::GetTagsImpl<RecordDim, RecordCoord>::type;

    namespace internal
    {
        template<typename RecordDim, typename RecordCoord>
        struct GetTagImpl
        {
            using type = boost::mp11::mp_back<GetTags<RecordDim, RecordCoord>>;
        };

        template<typename RecordDim>
        struct GetTagImpl<RecordDim, RecordCoord<>>
        {
            using type = NoName;
        };
    } // namespace internal

    /// Get the tag of the \ref Field at a \ref RecordCoord inside the record dimension tree.
    template<typename RecordDim, typename RecordCoord>
    using GetTag = typename internal::GetTagImpl<RecordDim, RecordCoord>::type;

    /// Is true if, starting at two coordinates in two record dimensions, all subsequent nodes in the record dimension
    /// tree have the same tag.
    /// \tparam RecordDimA First record dimension.
    /// \tparam LocalA \ref RecordCoord based on StartA along which the tags are compared.
    /// \tparam RecordDimB second record dimension.
    /// \tparam LocalB \ref RecordCoord based on StartB along which the tags are compared.
    template<typename RecordDimA, typename LocalA, typename RecordDimB, typename LocalB>
    inline constexpr auto hasSameTags = []() constexpr
    {
        if constexpr(LocalA::size != LocalB::size)
            return false;
        else if constexpr(LocalA::size == 0 && LocalB::size == 0)
            return true;
        else
            return std::is_same_v<GetTags<RecordDimA, LocalA>, GetTags<RecordDimB, LocalB>>;
    }
    ();

    namespace internal
    {
        template<typename FieldList, typename Tag>
        struct FindFieldByTag
        {
            template<typename Field>
            using HasTag = std::is_same<GetFieldTag<Field>, Tag>;

            static constexpr auto value = boost::mp11::mp_find_if<FieldList, HasTag>::value;
        };

        template<typename RecordDim, typename RecordCoord, typename... Tags>
        struct GetCoordFromTagsImpl
        {
            static_assert(boost::mp11::mp_size<RecordDim>::value != 0, "Tag combination is not valid");
        };

        template<typename... Fields, std::size_t... ResultCoords, typename FirstTag, typename... Tags>
        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<ResultCoords...>, FirstTag, Tags...>
        {
            static constexpr auto tagIndex = FindFieldByTag<boost::mp11::mp_list<Fields...>, FirstTag>::value;
            static_assert(
                tagIndex < sizeof...(Fields),
                "FirstTag was not found inside this Record. Does your record dimension contain the tag you access "
                "with?");

            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Fields...>, tagIndex>>;

            using type =
                typename GetCoordFromTagsImpl<ChildType, RecordCoord<ResultCoords..., tagIndex>, Tags...>::type;
        };

        template<
            typename ChildType,
            std::size_t Count,
            std::size_t... ResultCoords,
            typename FirstTag,
            typename... Tags>
        struct GetCoordFromTagsImpl<ChildType[Count], RecordCoord<ResultCoords...>, FirstTag, Tags...>
        {
            static_assert(isRecordCoord<FirstTag>, "Please use a RecordCoord<I> to index into static arrays");
            static_assert(FirstTag::size == 1, "Expected RecordCoord with 1 coordinate");
            static_assert(FirstTag::front < Count, "Index out of bounds");

            using type =
                typename GetCoordFromTagsImpl<ChildType, RecordCoord<ResultCoords..., FirstTag::front>, Tags...>::type;
        };

        template<typename RecordDim, typename RecordCoord>
        struct GetCoordFromTagsImpl<RecordDim, RecordCoord>
        {
            using type = RecordCoord;
        };

        // unpack a list of tags
        template<typename... Fields, typename... Tags>
        struct GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<>, boost::mp11::mp_list<Tags...>>
            : GetCoordFromTagsImpl<Record<Fields...>, RecordCoord<>, Tags...>
        {
        };
        template<typename ChildType, std::size_t Count, typename... Tags>
        struct GetCoordFromTagsImpl<ChildType[Count], RecordCoord<>, boost::mp11::mp_list<Tags...>>
            : GetCoordFromTagsImpl<ChildType[Count], RecordCoord<>, Tags...>
        {
        };
    } // namespace internal

    /// Converts a series of tags, or a list of tags, navigating down a record dimension into a \ref RecordCoord.
    template<typename RecordDim, typename... Tags>
    using GetCoordFromTags = typename internal::GetCoordFromTagsImpl<RecordDim, RecordCoord<>, Tags...>::type;

    namespace internal
    {
        template<typename RecordDim, typename... RecordCoordOrTags>
        struct GetTypeImpl
        {
            using type = typename GetTypeImpl<RecordDim, GetCoordFromTags<RecordDim, RecordCoordOrTags...>>::type;
        };

        template<typename... Children, std::size_t HeadCoord, std::size_t... TailCoords>
        struct GetTypeImpl<Record<Children...>, RecordCoord<HeadCoord, TailCoords...>>
        {
            using ChildType = GetFieldType<boost::mp11::mp_at_c<Record<Children...>, HeadCoord>>;
            using type = typename GetTypeImpl<ChildType, RecordCoord<TailCoords...>>::type;
        };

        template<typename ChildType, std::size_t N, std::size_t HeadCoord, std::size_t... TailCoords>
        struct GetTypeImpl<ChildType[N], RecordCoord<HeadCoord, TailCoords...>>
        {
            using type = typename GetTypeImpl<ChildType, RecordCoord<TailCoords...>>::type;
        };

        template<typename T>
        struct GetTypeImpl<T, RecordCoord<>>
        {
            static_assert(isAllowedFieldType<T>);
            using type = T;
        };
    } // namespace internal

    /// Returns the type of a node in a record dimension tree identified by a given \ref RecordCoord or a series of
    /// tags.
    template<typename RecordDim, typename... RecordCoordOrTags>
    using GetType = typename internal::GetTypeImpl<RecordDim, RecordCoordOrTags...>::type;

    namespace internal
    {
        template<typename RecordDim, typename RecordCoord>
        struct LeafRecordCoordsImpl;

        template<typename T, std::size_t... RCs>
        struct LeafRecordCoordsImpl<T, RecordCoord<RCs...>>
        {
            using type = boost::mp11::mp_list<RecordCoord<RCs...>>;
        };

        template<typename... Fields, std::size_t... RCs>
        struct LeafRecordCoordsImpl<Record<Fields...>, RecordCoord<RCs...>>
        {
            template<std::size_t... Is>
            static auto help(std::index_sequence<Is...>)
            {
                return boost::mp11::mp_append<
                    typename LeafRecordCoordsImpl<GetFieldType<Fields>, RecordCoord<RCs..., Is>>::type...>{};
            }
            using type = decltype(help(std::make_index_sequence<sizeof...(Fields)>{}));
        };

        template<typename Child, std::size_t N, std::size_t... RCs>
        struct LeafRecordCoordsImpl<Child[N], RecordCoord<RCs...>>
        {
            template<std::size_t... Is>
            static auto help(std::index_sequence<Is...>)
            {
                return boost::mp11::mp_append<
                    typename LeafRecordCoordsImpl<Child, RecordCoord<RCs..., Is>>::type...>{};
            }
            using type = decltype(help(std::make_index_sequence<N>{}));
        };
    } // namespace internal

    /// Returns a flat type list containing all record coordinates to all leaves of the given record dimension.
    template<typename RecordDim>
    using LeafRecordCoords = typename internal::LeafRecordCoordsImpl<RecordDim, RecordCoord<>>::type;

    namespace internal
    {
        // adapted from boost::mp11, but with LLAMA_FN_HOST_ACC_INLINE
        template<template<typename...> typename L, typename... T, typename F>
        LLAMA_FN_HOST_ACC_INLINE constexpr void mp_for_each_inlined(L<T...>, F&& f)
        {
            using A = int[sizeof...(T)];
            (void) A{((void) f(T{}), 0)...};
        }
    } // namespace internal

    /// Iterates over the record dimension tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have `operator()` with a template parameter for
    /// the \ref RecordCoord in the record dimension tree.
    /// \param baseCoord \ref RecordCoord at which the iteration should be started. The functor is called on elements
    /// beneath this coordinate.
    template<typename RecordDim, typename Functor, std::size_t... Coords>
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafCoord(Functor&& functor, RecordCoord<Coords...> baseCoord)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        internal::mp_for_each_inlined(
            LeafRecordCoords<GetType<RecordDim, RecordCoord<Coords...>>>{},
            [&](auto innerCoord) LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr)
            { std::forward<Functor>(functor)(cat(baseCoord, innerCoord)); });
    }

    /// Iterates over the record dimension tree and calls a functor on each element.
    /// \param functor Functor to execute at each element of. Needs to have `operator()` with a template parameter for
    /// the \ref RecordCoord in the record dimension tree.
    /// \param baseTags Tags used to define where the iteration should be started. The functor is called on elements
    /// beneath this coordinate.
    template<typename RecordDim, typename Functor, typename... Tags>
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeafCoord(Functor&& functor, Tags... /*baseTags*/)
    {
        LLAMA_FORCE_INLINE_RECURSIVE
        forEachLeafCoord<RecordDim>(std::forward<Functor>(functor), GetCoordFromTags<RecordDim, Tags...>{});
    }

    namespace internal
    {
        template<typename T>
        struct FlattenRecordDimImpl
        {
            using type = boost::mp11::mp_list<T>;
        };

        template<typename... Fields>
        struct FlattenRecordDimImpl<Record<Fields...>>
        {
            using type = boost::mp11::mp_append<typename FlattenRecordDimImpl<GetFieldType<Fields>>::type...>;
        };
        template<typename Child, std::size_t N>
        struct FlattenRecordDimImpl<Child[N]>
        {
            using type = boost::mp11::mp_repeat_c<typename FlattenRecordDimImpl<Child>::type, N>;
        };
    } // namespace internal

    /// Returns a flat type list containing all leaf field types of the given record dimension.
    template<typename RecordDim>
    using FlatRecordDim = typename internal::FlattenRecordDimImpl<RecordDim>::type;

    /// The total number of fields in the recursively expanded record dimension.
    template<typename RecordDim>
    inline constexpr std::size_t flatFieldCount = 1;

    template<typename... Children>
    inline constexpr std::size_t flatFieldCount<
        Record<Children...>> = (flatFieldCount<GetFieldType<Children>> + ... + 0);

    template<typename Child, std::size_t N>
    inline constexpr std::size_t flatFieldCount<Child[N]> = flatFieldCount<Child>* N;

    namespace internal
    {
        template<std::size_t I, typename RecordDim>
        inline constexpr std::size_t flatFieldCountBefore = 0;

        template<typename... Children>
        inline constexpr std::size_t flatFieldCountBefore<0, Record<Children...>> = 0;

        // recursive formulation to benefit from template instantiation memoization
        // this massively improves compilation time when this template is instantiated with a lot of different I
        template<std::size_t I, typename... Children>
        inline constexpr std::size_t flatFieldCountBefore<
            I,
            Record<
                Children...>> = flatFieldCountBefore<I - 1, Record<Children...>> + flatFieldCount<GetFieldType<boost::mp11::mp_at_c<Record<Children...>, I - 1>>>;
    } // namespace internal

    /// The equivalent zero based index into a flat record dimension (\ref FlatRecordDim) of the given hierarchical
    /// record coordinate.
    template<typename RecordDim, typename RecordCoord>
    inline constexpr std::size_t flatRecordCoord = 0;

    template<typename T>
    inline constexpr std::size_t flatRecordCoord<T, RecordCoord<>> = 0;

    template<typename... Children, std::size_t I, std::size_t... Is>
    inline constexpr std::size_t flatRecordCoord<
        Record<Children...>,
        RecordCoord<
            I,
            Is...>> = internal::
                          flatFieldCountBefore<
                              I,
                              Record<
                                  Children...>> + flatRecordCoord<GetFieldType<boost::mp11::mp_at_c<Record<Children...>, I>>, RecordCoord<Is...>>;

    template<typename Child, std::size_t N, std::size_t I, std::size_t... Is>
    inline constexpr std::size_t flatRecordCoord<Child[N], RecordCoord<I, Is...>> = flatFieldCount<Child>* I
        + flatRecordCoord<Child, RecordCoord<Is...>>;

    namespace internal
    {
        template<typename TypeList>
        constexpr auto flatAlignOfImpl()
        {
            using namespace boost::mp11;

            std::size_t maxAlign = 0;
            mp_for_each<mp_transform<mp_identity, TypeList>>([&](auto e) constexpr
                                                             {
                                                                 using T = typename decltype(e)::type;
                                                                 maxAlign = std::max(maxAlign, alignof(T));
                                                             });
            return maxAlign;
        }
    } // namespace internal

    /// The alignment of a type list if its elements would be in a normal struct.
    template<typename TypeList>
    inline constexpr std::size_t flatAlignOf = internal::flatAlignOfImpl<TypeList>();

    /// The alignment of a type T.
    template<typename T>
    inline constexpr std::size_t alignOf = alignof(T);

    /// The alignment of a record dimension if its fields would be in a normal struct.
    template<typename... Fields>
    inline constexpr std::size_t alignOf<Record<Fields...>> = flatAlignOf<FlatRecordDim<Record<Fields...>>>;

    /// Returns the integral n rounded up to be a multiple of mult.
    template<typename Integral>
    [[nodiscard]] LLAMA_FN_HOST_ACC_INLINE constexpr auto roundUpToMultiple(Integral n, Integral mult) -> Integral
    {
        return (n + mult - 1) / mult * mult;
    }

    namespace internal
    {
        template<typename TypeList, bool Align, bool IncludeTailPadding>
        constexpr auto sizeOfImpl() -> std::size_t
        {
            using namespace boost::mp11;

            std::size_t size = 0;
            std::size_t maxAlign = 0;
            mp_for_each<mp_transform<mp_identity, TypeList>>([&](auto e) constexpr
                                                             {
                                                                 using T = typename decltype(e)::type;
                                                                 if constexpr(Align)
                                                                 {
                                                                     size = roundUpToMultiple(size, alignof(T));
                                                                     maxAlign = std::max(maxAlign, alignof(T));
                                                                 }
                                                                 // NOLINTNEXTLINE(readability-misleading-indentation)
                                                                 size += sizeof(T);
                                                             });

            // final padding, so next struct can start right away
            if constexpr(Align && IncludeTailPadding)
                size = roundUpToMultiple(size, maxAlign); // TODO(bgruber): we could use flatAlignOf<TypeList> here, at
                                                          // the cost of more template instantiations
            return size;
        }

        template<bool Align, typename TypeList, std::size_t I>
        constexpr auto offsetOfImplWorkaround() -> std::size_t;

        // recursive formulation to benefit from template instantiation memoization
        // this massively improves compilation time when this template is instantiated with a lot of different I
        template<bool Align, typename TypeList, std::size_t I>
        inline constexpr std::size_t offsetOfImpl
            = offsetOfImplWorkaround<Align, TypeList, I>(); // FIXME: MSVC fails to compile an IILE here.

        template<bool Align, typename TypeList>
        inline constexpr std::size_t offsetOfImpl<Align, TypeList, 0> = 0;

        template<bool Align, typename TypeList, std::size_t I>
        constexpr auto offsetOfImplWorkaround() -> std::size_t
        {
            std::size_t offset = offsetOfImpl<Align, TypeList, I - 1> + sizeof(boost::mp11::mp_at_c<TypeList, I - 1>);
            if constexpr(Align)
                offset = roundUpToMultiple(offset, alignof(boost::mp11::mp_at_c<TypeList, I>));
            return offset;
        }
    } // namespace internal

    /// The size of a type list if its elements would be in a normal struct.
    template<typename TypeList, bool Align, bool IncludeTailPadding = true>
    inline constexpr std::size_t flatSizeOf = internal::sizeOfImpl<TypeList, Align, IncludeTailPadding>();

    /// The size of a type T.
    template<typename T, bool Align = false, bool IncludeTailPadding = true>
    inline constexpr std::size_t sizeOf = sizeof(T);

    /// The size of a record dimension if its fields would be in a normal struct.
    template<typename... Fields, bool Align, bool IncludeTailPadding>
    inline constexpr std::size_t sizeOf<Record<Fields...>, Align, IncludeTailPadding> = flatSizeOf<
        FlatRecordDim<Record<Fields...>>,
        Align,
        IncludeTailPadding>;

    /// The byte offset of an element in a type list ifs elements would be in a normal struct.
    template<typename TypeList, std::size_t I, bool Align>
    inline constexpr std::size_t flatOffsetOf = internal::offsetOfImpl<Align, TypeList, I>;

    /// The byte offset of an element in a record dimension if it would be a normal struct.
    /// \tparam RecordDim Record dimension tree.
    /// \tparam RecordCoord Record coordinate of an element inrecord dimension tree.
    template<typename RecordDim, typename RecordCoord, bool Align = false>
    inline constexpr std::size_t offsetOf
        = flatOffsetOf<FlatRecordDim<RecordDim>, flatRecordCoord<RecordDim, RecordCoord>, Align>;

    template<typename S>
    auto structName(S = {}) -> std::string
    {
        auto s = boost::core::demangle(typeid(S).name());
        if(const auto pos = s.rfind(':'); pos != std::string::npos)
            s = s.substr(pos + 1);
        return s;
    }

    namespace internal
    {
        template<typename T>
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

        // TODO(bgruber): replace in C++20
        template<class T>
        struct IsBoundedArray : std::false_type
        {
        };

        template<class T, std::size_t N>
        struct IsBoundedArray<T[N]> : std::true_type
        {
        };
    } // namespace internal

    namespace internal
    {
        template<typename T, template<typename> typename TypeFunctor>
        struct TransformLeavesImpl
        {
            using type = TypeFunctor<T>;
        };

        template<typename... Fields, template<typename> typename TypeFunctor>
        struct TransformLeavesImpl<Record<Fields...>, TypeFunctor>
        {
            using type = Record<
                Field<GetFieldTag<Fields>, typename TransformLeavesImpl<GetFieldType<Fields>, TypeFunctor>::type>...>;
        };
        template<typename Child, std::size_t N, template<typename> typename TypeFunctor>
        struct TransformLeavesImpl<Child[N], TypeFunctor>
        {
            using type = typename TransformLeavesImpl<Child, TypeFunctor>::type[N];
        };
    } // namespace internal

    /// Creates a new record dimension where each new leaf field's type is the result of applying FieldTypeFunctor to
    /// the original leaf field's type.
    template<typename RecordDim, template<typename> typename FieldTypeFunctor>
    using TransformLeaves = typename internal::TransformLeavesImpl<RecordDim, FieldTypeFunctor>::type;

    namespace internal
    {
        // TODO: we might implement this better by expanding a record dim into a list of tag lists and then computing a
        // real set union of the two tag list lists

        template<typename A, typename B>
        auto mergeRecordDimsImpl(boost::mp11::mp_identity<A> a, boost::mp11::mp_identity<B>)
        {
            static_assert(std::is_same_v<A, B>, "Cannot merge record and non-record or fields with different types");
            return a;
        }

        template<typename A, std::size_t NA, typename B, std::size_t NB>
        auto mergeRecordDimsImpl(
            [[maybe_unused]] boost::mp11::mp_identity<A[NA]> a,
            [[maybe_unused]] boost::mp11::mp_identity<B[NB]> b)
        {
            static_assert(std::is_same_v<A, B>, "Cannot merge arrays of different type");
            if constexpr(NA < NB)
                return b;
            else
                return a;
        }

        template<typename... FieldsA>
        auto mergeRecordDimsImpl(boost::mp11::mp_identity<Record<FieldsA...>> a, boost::mp11::mp_identity<Record<>>)
        {
            return a;
        }

        template<
            typename... FieldsA,
            typename FieldB,
            typename... FieldsB,
            auto pos = FindFieldByTag<Record<FieldsA...>, GetFieldTag<FieldB>>::value>
        auto mergeRecordDimsImpl(
            boost::mp11::mp_identity<Record<FieldsA...>>,
            boost::mp11::mp_identity<Record<FieldB, FieldsB...>>)
        {
            using namespace boost::mp11;
            if constexpr(pos == sizeof...(FieldsA))
            {
                return mergeRecordDimsImpl(
                    mp_identity<Record<FieldsA..., FieldB>>{},
                    mp_identity<Record<FieldsB...>>{});
            }
            else
            {
                using OldFieldA = mp_at_c<Record<FieldsA...>, pos>;
                using NewFieldA = Field<
                    GetFieldTag<OldFieldA>,
                    typename decltype(mergeRecordDimsImpl(
                        mp_identity<GetFieldType<OldFieldA>>{},
                        mp_identity<GetFieldType<FieldB>>{}))::type>;
                using NewRecordA = mp_replace_at_c<Record<FieldsA...>, pos, NewFieldA>;
                return mergeRecordDimsImpl(mp_identity<NewRecordA>{}, mp_identity<Record<FieldsB...>>{});
            }
        }
    } // namespace internal

    /// Creates a merged record dimension, where duplicated, nested fields are unified.
    template<typename RecordDimA, typename RecordDimB>
    using MergedRecordDims = typename decltype(internal::mergeRecordDimsImpl(
        boost::mp11::mp_identity<RecordDimA>{},
        boost::mp11::mp_identity<RecordDimB>{}))::type;

    /// Returns the tags interspersed by '.' represented by the given record coord in the given record dimension.
    template<typename RecordDim, std::size_t... Coords>
    auto recordCoordTags(RecordCoord<Coords...>) -> std::string
    {
        using Tags = GetTags<RecordDim, RecordCoord<Coords...>>;

        std::string r;
        boost::mp11::mp_for_each<Tags>(
            [&](auto tag)
            {
                using Tag = decltype(tag);
                if(!r.empty())
                    r += '.';
                if constexpr(isRecordCoord<Tag>)
                {
                    static_assert(Tag::size == 1);
                    r += std::to_string(Tag::front); // handle array indices
                }
                else
                    r += structName(tag);
            });
        return r;
    }
} // namespace llama

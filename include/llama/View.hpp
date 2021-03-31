// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "BlobAllocators.hpp"
#include "Concepts.hpp"
#include "Core.hpp"
#include "macros.hpp"
#include "mapping/One.hpp"

#include <boost/preprocessor/cat.hpp>
#include <type_traits>

namespace llama
{
#ifdef __cpp_concepts
    template <typename T_Mapping, Blob BlobType>
#else
    template <typename T_Mapping, typename BlobType>
#endif
    struct View;

    namespace internal
    {
        template <typename Allocator>
        using AllocatorBlobType = decltype(std::declval<Allocator>()(0));

        template <typename Allocator, typename Mapping, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeBlobArray(
            const Allocator& alloc,
            const Mapping& mapping,
            std::integer_sequence<std::size_t, Is...>) -> Array<AllocatorBlobType<Allocator>, Mapping::blobCount>
        {
            return {alloc(mapping.blobSize(Is))...};
        }
    } // namespace internal

    /// Creates a view based on the given mapping, e.g. \ref mapping::AoS or \ref mapping::SoA. For allocating the
    /// view's underlying memory, the specified allocator callable is used (or the default one, which is \ref
    /// bloballoc::Vector). The allocator callable is called with the size of bytes to allocate for each blob of the
    /// mapping. This function is the preferred way to create a \ref View.
#ifdef __cpp_concepts
    template <typename Mapping, BlobAllocator Allocator = bloballoc::Vector<>>
#else
    template <typename Mapping, typename Allocator = bloballoc::Vector<>>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator>>
    {
        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
        return {std::move(mapping), std::move(blobs)};
    }

    /// Allocates a \ref View holding a single record backed by stack memory (\ref bloballoc::Stack).
    /// \tparam Dim Dimension of the \ref ArrayDomain of the \ref View.
    template <std::size_t Dim, typename RecordDim>
    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
    {
        using Mapping = llama::mapping::One<ArrayDomain<Dim>, RecordDim>;
        return allocView(Mapping{}, llama::bloballoc::Stack<sizeOf<RecordDim>>{});
    }

    template <typename View, typename BoundRecordCoord = RecordCoord<>, bool OwnView = false>
    struct VirtualRecord;

    template <typename View>
    inline constexpr auto is_VirtualRecord = false;

    template <typename View, typename BoundRecordCoord, bool OwnView>
    inline constexpr auto is_VirtualRecord<VirtualRecord<View, BoundRecordCoord, OwnView>> = true;

    /// A \ref VirtualRecord that owns and holds a single value.
    template <typename RecordDim>
    using One = VirtualRecord<decltype(allocViewStack<1, RecordDim>()), RecordCoord<>, true>;

    /// Creates a single \ref VirtualRecord owning a view with stack memory and
    /// copies all values from an existing \ref VirtualRecord.
    template <typename VirtualRecord>
    LLAMA_FN_HOST_ACC_INLINE auto copyVirtualRecordStack(const VirtualRecord& vd) -> decltype(auto)
    {
        One<typename VirtualRecord::AccessibleRecordDim> temp;
        temp = vd;
        return temp;
    }

    namespace internal
    {
        template <
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(
            LeftRecord& left,
            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> LeftRecord&
        {
            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                [&](auto leftCoord)
                {
                    using LeftInnerCoord = decltype(leftCoord);
                    forEachLeaf<typename RightRecord::AccessibleRecordDim>(
                        [&](auto rightCoord)
                        {
                            using RightInnerCoord = decltype(rightCoord);
                            if constexpr (hasSameTags<
                                              typename LeftRecord::AccessibleRecordDim,
                                              LeftInnerCoord,
                                              typename RightRecord::AccessibleRecordDim,
                                              RightInnerCoord>)
                            {
                                Functor{}(left(leftCoord), right(rightCoord));
                            }
                        });
                });
            return left;
        }

        template <typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordArithOperator(LeftRecord& left, const T& right) -> LeftRecord&
        {
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>([&](auto leftCoord)
                                                                  { Functor{}(left(leftCoord), right); });
            return left;
        }

        template <
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(
            const LeftRecord& left,
            const VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>& right) -> bool
        {
            using RightRecord = VirtualRecord<RightView, RightBoundRecordDim, RightOwnView>;
            bool result = true;
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                [&](auto leftCoord)
                {
                    using LeftInnerCoord = decltype(leftCoord);
                    forEachLeaf<typename RightRecord::AccessibleRecordDim>(
                        [&](auto rightCoord)
                        {
                            using RightInnerCoord = decltype(rightCoord);
                            if constexpr (hasSameTags<
                                              typename LeftRecord::AccessibleRecordDim,
                                              LeftInnerCoord,
                                              typename RightRecord::AccessibleRecordDim,
                                              RightInnerCoord>)
                            {
                                result &= Functor{}(left(leftCoord), right(rightCoord));
                            }
                        });
                });
            return result;
        }

        template <typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualRecordRelOperator(const LeftRecord& left, const T& right) -> bool
        {
            bool result = true;
            forEachLeaf<typename LeftRecord::AccessibleRecordDim>(
                [&](auto leftCoord) {
                    result &= Functor{}(
                        left(leftCoord),
                        static_cast<std::remove_reference_t<decltype(left(leftCoord))>>(right));
                });
            return result;
        }

        struct Assign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a = b;
            }
        };

        struct PlusAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a += b;
            }
        };

        struct MinusAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a -= b;
            }
        };

        struct MultiplyAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a *= b;
            }
        };

        struct DivideAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a /= b;
            }
        };

        struct ModuloAssign
        {
            template <typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto) operator()(A& a, const B& b) const
            {
                return a %= b;
            }
        };

        template <typename TWithOptionalConst, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(TWithOptionalConst& leaf, T) -> std::
            enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::reference_wrapper<TWithOptionalConst>>
        {
            return leaf;
        }

        template <typename VirtualRecord, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(VirtualRecord&& vd, Record<Fields...>)
        {
            return std::make_tuple(asTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template <typename TWithOptionalConst, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(TWithOptionalConst& leaf, T)
            -> std::enable_if_t<!is_VirtualRecord<std::decay_t<TWithOptionalConst>>, std::tuple<TWithOptionalConst&>>
        {
            return {leaf};
        }

        template <typename VirtualRecord, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(VirtualRecord&& vd, Record<Fields...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template <typename T, typename = void>
        constexpr inline auto isTupleLike = false;

        // get<I>(t) and std::tuple_size<T> must be available
        using std::get; // make sure a get<0>() can be found, so the compiler can compile the trait
        template <typename T>
        constexpr inline auto
            isTupleLike<T, std::void_t<decltype(get<0>(std::declval<T>())), std::tuple_size<T>>> = true;

        template <typename... Ts>
        constexpr inline auto dependentFalse = false;

        template <typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>);

        template <typename T1, typename T2>
        LLAMA_FN_HOST_ACC_INLINE void assignTupleElement(T1&& dst, T2&& src)
        {
            if constexpr (isTupleLike<std::decay_t<T1>> && isTupleLike<std::decay_t<T2>>)
            {
                static_assert(std::tuple_size_v<std::decay_t<T1>> == std::tuple_size_v<std::decay_t<T2>>);
                assignTuples(dst, src, std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>{});
            }
            else if constexpr (!isTupleLike<std::decay_t<T1>> && !isTupleLike<std::decay_t<T2>>)
                std::forward<T1>(dst) = std::forward<T2>(src);
            else
                static_assert(dependentFalse<T1, T2>, "Elements to assign are not tuple/tuple or non-tuple/non-tuple.");
        }

        template <typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>)
        {
            static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>);
            using std::get;
            (assignTupleElement(get<Is>(std::forward<Tuple1>(dst)), get<Is>(std::forward<Tuple2>(src))), ...);
        }

        template <typename T, typename Tuple, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeFromTuple(Tuple&& src, std::index_sequence<Is...>)
        {
            using std::get;
            return T{get<Is>(std::forward<Tuple>(src))...};
        }

        template <typename T, typename SFINAE, typename... Args>
        constexpr inline auto isDirectListInitializableImpl = false;

        template <typename T, typename... Args>
        constexpr inline auto
            isDirectListInitializableImpl<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...> = true;

        template <typename T, typename... Args>
        constexpr inline auto isDirectListInitializable = isDirectListInitializableImpl<T, void, Args...>;

        template <typename T, typename Tuple>
        constexpr inline auto isDirectListInitializableFromTuple = false;

        template <typename T, template <typename...> typename Tuple, typename... Args>
        constexpr inline auto
            isDirectListInitializableFromTuple<T, Tuple<Args...>> = isDirectListInitializable<T, Args...>;
    } // namespace internal

    /// Virtual record type returned by \ref View after resolving an array domain
    /// coordinate or partially resolving a \ref RecordCoord. A virtual record
    /// does not hold data itself (thus named "virtual"), it just binds enough
    /// information (array domain coord and partial record coord) to retrieve it
    /// from a \ref View later. Virtual records should not be created by the
    /// user. They are returned from various access functions in \ref View and
    /// VirtualRecord itself.
    template <typename T_View, typename BoundRecordCoord, bool OwnView>
    struct VirtualRecord
    {
        using View = T_View; ///< View this virtual record points into.

    private:
        using ArrayDomain = typename View::Mapping::ArrayDomain;
        using RecordDim = typename View::Mapping::RecordDim;

        const ArrayDomain userDomainPos;
        std::conditional_t<OwnView, View, View&> view;

    public:
        /// Subtree of the record dimension of View starting at
        /// BoundRecordCoord. If BoundRecordCoord is `RecordCoord<>` (default)
        /// AccessibleRecordDim is the same as `Mapping::RecordDim`.
        using AccessibleRecordDim = GetType<RecordDim, BoundRecordCoord>;

        LLAMA_FN_HOST_ACC_INLINE VirtualRecord()
            /* requires(OwnView) */
            : userDomainPos({})
            , view{allocViewStack<1, RecordDim>()}
        {
            static_assert(OwnView, "The default constructor of VirtualRecord is only available if the ");
        }

        LLAMA_FN_HOST_ACC_INLINE
        VirtualRecord(ArrayDomain userDomainPos, std::conditional_t<OwnView, View&&, View&> view)
            : userDomainPos(userDomainPos)
            , view{static_cast<decltype(view)>(view)}
        {
        }

        VirtualRecord(const VirtualRecord&) = default;
        VirtualRecord(VirtualRecord&&) = default;

        /// Access a record in the record dimension underneath the current virtual
        /// record using a \ref RecordCoord. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual record.
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            if constexpr (isRecord<GetType<RecordDim, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<const View, AbsolutCoord>{userDomainPos, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, AbsolutCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> coord = {}) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            if constexpr (isRecord<GetType<RecordDim, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<View, AbsolutCoord>{userDomainPos, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, AbsolutCoord{});
            }
        }

        /// Access a record in the record dimension underneath the current virtual
        /// record using a series of tags. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual record.
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTagsRelative<RecordDim, BoundRecordCoord, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        // we need this one to disable the compiler generated copy assignment
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const VirtualRecord& other) -> VirtualRecord&
        {
            return this->operator=<VirtualRecord>(other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::Assign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator+=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::PlusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator-=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::MinusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator*=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::MultiplyAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator/=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::DivideAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator%=(const T& other) -> VirtualRecord&
        {
            return internal::virtualRecordArithOperator<internal::ModuloAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) += t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const T& t, const VirtualRecord& vd)
        {
            return vd + t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator-(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) -= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) *= t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const T& t, const VirtualRecord& vd)
        {
            return vd * t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator/(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) /= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator%(const VirtualRecord& vd, const T& t)
        {
            return copyVirtualRecordStack(vd) %= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd == t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::not_equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd != t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::less<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd > t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::less_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd >= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::greater<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd < t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const VirtualRecord& vd, const T& t) -> bool
        {
            return internal::virtualRecordRelOperator<std::greater_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualRecord<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const T& t, const VirtualRecord& vd) -> bool
        {
            return vd <= t;
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple()
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple() const
        {
            return internal::asTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple()
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple() const
        {
            return internal::asFlatTupleImpl(*this, AccessibleRecordDim{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() const -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this VirtualRecord recursively represents like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        struct Loader
        {
            VirtualRecord& vd;

            template <typename T>
            LLAMA_FN_HOST_ACC_INLINE operator T()
            {
                return vd.loadAs<T>();
            }
        };

        struct LoaderConst
        {
            const VirtualRecord& vd;

            template <typename T>
            LLAMA_FN_HOST_ACC_INLINE operator T() const
            {
                return vd.loadAs<T>();
            }
        };

        LLAMA_FN_HOST_ACC_INLINE auto load() -> Loader
        {
            return {*this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto load() const -> LoaderConst
        {
            return {*this};
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE void store(const TupleLike& t)
        {
            internal::assignTuples(asTuple(), t, std::make_index_sequence<std::tuple_size_v<TupleLike>>{});
        }
    };
} // namespace llama

template <typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_size<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
    : boost::mp11::mp_size<typename llama::VirtualRecord<View, BoundRecordCoord, OwnView>::AccessibleRecordDim>
{
};

template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
{
    using type = decltype(std::declval<llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
};

template <std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>
{
    using type
        = decltype(std::declval<const llama::VirtualRecord<View, BoundRecordCoord, OwnView>>().template get<I>());
};

//#include <boost/stl_interfaces/iterator_interface.hpp>

namespace llama
{
    // requires Boost 1.74 which is quite new
#if 0
    template <typename View>
    struct Iterator
        : boost::stl_interfaces::proxy_iterator_interface<
              Iterator<View>,
              std::random_access_iterator_tag,
              decltype(std::declval<View>()(ArrayDomain<1>{}))>
    {
        constexpr decltype(auto) operator*() const
        {
            return (*view)(coord);
        }

        constexpr auto operator+=(std::ptrdiff_t n) -> Iterator&
        {
            coord[0] += n;
            return *this;
        }

        friend constexpr auto operator-(const Iterator& a, const Iterator& b)
        {
            return a.coord[0] - b.coord[0];
        }

        friend constexpr bool operator==(const Iterator& a, const Iterator& b)
        {
            return a.coord == b.coord;
        }

        ArrayDomain<1> coord;
        View* view;
    };
#endif

    // Currently, only 1D iterators are supported, becaues higher dimensional iterators are difficult if we also
    // want to preserve good codegen. Multiple nested loops seem to be superior to a single iterator over multiple
    // dimensions. At least compilers are able to produce better code. std::mdspan also discovered similar
    // difficulties and there was a discussion in WG21 in Oulu 2016 to remove/postpone iterators from the design. In
    // std::mdspan's design, the iterator iterated over the co-domain.
    template <typename View>
    struct Iterator
    {
        static_assert(View::ArrayDomain::rank == 1, "Iterators for non-1D views are not implemented");

        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename View::VirtualRecordType;
        using difference_type = std::ptrdiff_t;
        using pointer = internal::IndirectValue<value_type>;
        using reference = value_type;

        constexpr auto operator++() -> Iterator&
        {
            ++coord[0];
            return *this;
        }
        constexpr auto operator++(int) -> Iterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        constexpr auto operator--() -> Iterator&
        {
            --coord[0];
            return *this;
        }

        constexpr auto operator--(int) -> Iterator
        {
            auto tmp{*this};
            --*this;
            return tmp;
        }

        constexpr auto operator*() const -> reference
        {
            return (*view)(coord);
        }

        constexpr auto operator->() const -> pointer
        {
            return internal::IndirectValue{**this};
        }

        constexpr auto operator[](difference_type i) const -> reference
        {
            return *(*this + i);
        }

        constexpr auto operator+=(difference_type n) -> Iterator&
        {
            coord[0] = static_cast<difference_type>(coord[0]) + n;
            return *this;
        }

        friend constexpr auto operator+(Iterator it, difference_type n) -> Iterator
        {
            it += n;
            return it;
        }

        friend constexpr auto operator+(difference_type n, Iterator it) -> Iterator
        {
            return it + n;
        }

        constexpr auto operator-=(difference_type n) -> Iterator&
        {
            coord[0] = static_cast<difference_type>(coord[0]) - n;
            return *this;
        }

        friend constexpr auto operator-(Iterator it, difference_type n) -> Iterator
        {
            it -= n;
            return it;
        }

        friend constexpr auto operator-(const Iterator& a, const Iterator& b) -> difference_type
        {
            return static_cast<std::ptrdiff_t>(a.coord[0]) - static_cast<std::ptrdiff_t>(b.coord[0]);
        }

        friend constexpr auto operator==(const Iterator& a, const Iterator& b) -> bool
        {
            return a.coord[0] == b.coord[0];
        }

        friend constexpr auto operator!=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a == b);
        }

        friend constexpr auto operator<(const Iterator& a, const Iterator& b) -> bool
        {
            return a.coord[0] < b.coord[0];
        }

        friend constexpr auto operator>(const Iterator& a, const Iterator& b) -> bool
        {
            return b < a;
        }

        friend constexpr auto operator<=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a > b);
        }

        friend constexpr auto operator>=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a < b);
        }

        ArrayDomain<1> coord;
        View* view;
    };

    namespace internal
    {
        template <typename Mapping, typename RecordCoord, typename = void>
        struct isComputed : std::false_type
        {
        };

        template <typename Mapping, typename RecordCoord>
        struct isComputed<Mapping, RecordCoord, std::void_t<decltype(Mapping::isComputed(RecordCoord{}))>>
            : std::bool_constant<Mapping::isComputed(RecordCoord{})>
        {
        };
    } // namespace internal

    /// Central LLAMA class holding memory for storage and giving access to
    /// values stored there defined by a mapping. A view should be created using
    /// \ref allocView.
    /// \tparam T_Mapping The mapping used by the view to map accesses into
    /// memory.
    /// \tparam BlobType The storage type used by the view holding
    /// memory.
#ifdef __cpp_concepts
    template <typename T_Mapping, Blob BlobType>
#else
    template <typename T_Mapping, typename BlobType>
#endif
    struct View
    {
        using Mapping = T_Mapping;
        using ArrayDomain = typename Mapping::ArrayDomain;
        using RecordDim = typename Mapping::RecordDim;
        using VirtualRecordType = VirtualRecord<View<Mapping, BlobType>>;
        using VirtualRecordTypeConst = VirtualRecord<const View<Mapping, BlobType>>;

        View() = default;

        LLAMA_FN_HOST_ACC_INLINE
        View(Mapping mapping, Array<BlobType, Mapping::blobCount> storageBlobs)
            : mapping(std::move(mapping))
            , storageBlobs(std::move(storageBlobs))
        {
        }

        /// Retrieves the \ref VirtualRecord at the given \ref ArrayDomain
        /// coordinate.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) const -> decltype(auto)
        {
            if constexpr (isRecord<RecordDim>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecordTypeConst{arrayDomain, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDomain, RecordCoord<>{});
            }
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) -> decltype(auto)
        {
            if constexpr (isRecord<RecordDim>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecordType{arrayDomain, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDomain, RecordCoord<>{});
            }
        }

        /// Retrieves the \ref VirtualRecord at the \ref ArrayDomain coordinate
        /// constructed from the passed component indices.
        template <typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Index) == ArrayDomain::rank,
                "Please specify as many indices as your array domain has dimensions");
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (ArrayDomain{indices...});
        }

        template <typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) -> decltype(auto)
        {
            static_assert(
                sizeof...(Index) == ArrayDomain::rank,
                "Please specify as many indices as your array domain has dimensions");
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (ArrayDomain{indices...});
        }

        /// Retrieves the \ref VirtualRecord at the \ref ArrayDomain coordinate
        /// constructed from the passed component indices.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDomain arrayDomain) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (arrayDomain);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDomain arrayDomain) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (arrayDomain);
        }

        /// Retrieves the \ref VirtualRecord at the 1D \ref ArrayDomain coordinate
        /// constructed from the passed index.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (index);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (index);
        }

        auto begin() -> Iterator<View>
        {
            return {ArrayDomain{}, this};
        }

        auto end() -> Iterator<View>
        {
            return {mapping.arrayDomainSize, this};
        }

        Mapping mapping;
        Array<BlobType, Mapping::blobCount> storageBlobs;

    private:
        template <typename T_View, typename T_BoundRecordCoord, bool OwnView>
        friend struct VirtualRecord;

        template <std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDomain arrayDomain, RecordCoord<Coords...> dc = {}) const
            -> decltype(auto)
        {
            if constexpr (internal::isComputed<Mapping, RecordCoord<Coords...>>::value)
                return mapping.compute(arrayDomain, dc, storageBlobs);
            else
            {
                const auto [nr, offset] = mapping.template blobNrAndOffset<Coords...>(arrayDomain);
                using Type = GetType<RecordDim, RecordCoord<Coords...>>;
                return reinterpret_cast<const Type&>(storageBlobs[nr][offset]);
            }
        }

        template <std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDomain arrayDomain, RecordCoord<Coords...> dc = {})
            -> decltype(auto)
        {
            if constexpr (internal::isComputed<Mapping, RecordCoord<Coords...>>::value)
                return mapping.compute(arrayDomain, dc, storageBlobs);
            else
            {
                const auto [nr, offset] = mapping.template blobNrAndOffset<Coords...>(arrayDomain);
                using Type = GetType<RecordDim, RecordCoord<Coords...>>;
                return reinterpret_cast<Type&>(storageBlobs[nr][offset]);
            }
        }
    };

    template <typename View>
    inline constexpr auto IsView = false;

    template <typename Mapping, typename BlobType>
    inline constexpr auto IsView<View<Mapping, BlobType>> = true;

    /// Acts like a \ref View, but shows only a smaller and/or shifted part of
    /// another view it references, the parent view.
    template <typename T_ParentViewType>
    struct VirtualView
    {
        using ParentView = T_ParentViewType; ///< type of the parent view
        using Mapping = typename ParentView::Mapping; ///< mapping of the parent view
        using ArrayDomain = typename Mapping::ArrayDomain; ///< array domain of the parent view
        using VirtualRecordType = typename ParentView::VirtualRecordType; ///< VirtualRecord type of the
                                                                          ///< parent view

        /// Creates a VirtualView given a parent \ref View, offset and size.
        LLAMA_FN_HOST_ACC_INLINE
        VirtualView(ParentView& parentView, ArrayDomain offset, ArrayDomain size)
            : parentView(parentView)
            , offset(offset)
            , size(size)
        {
        }

        template <std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDomain arrayDomain) const -> const auto&
        {
            return parentView.template accessor<Coords...>(ArrayDomain{arrayDomain + offset});
        }

        template <std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDomain arrayDomain) -> auto&
        {
            return parentView.template accessor<Coords...>(ArrayDomain{arrayDomain + offset});
        }

        /// Same as \ref View::operator()(ArrayDomain), but shifted by the offset
        /// of this \ref VirtualView.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) const -> VirtualRecordType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDomain{arrayDomain + offset});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) -> VirtualRecordType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDomain{arrayDomain + offset});
        }

        /// Same as corresponding operator in \ref View, but shifted by the
        /// offset of this \ref VirtualView.
        template <typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> VirtualRecordType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDomain{ArrayDomain{indices...} + offset});
        }

        template <typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> VirtualRecordType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDomain{ArrayDomain{indices...} + offset});
        }

        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>&& dc = {}) const -> const auto&
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayDomain{});
        }

        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>&& dc = {}) -> auto&
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayDomain{});
        }

        ParentView& parentView; ///< reference to parent view.
        const ArrayDomain offset; ///< offset this view's \ref ArrayDomain coordinates are
                                  ///< shifted to the parent view.
        const ArrayDomain size;
    };
} // namespace llama

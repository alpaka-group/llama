// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Concepts.hpp"
#include "HasRanges.hpp"
#include "ProxyRefOpMixin.hpp"
#include "StructName.hpp"
#include "View.hpp"

#include <iosfwd>
#include <type_traits>

namespace llama
{
    template<typename View, typename BoundRecordCoord, bool OwnView>
    struct RecordRef;

    template<typename View>
    inline constexpr auto isRecordRef = false;

    template<typename View, typename BoundRecordCoord, bool OwnView>
    inline constexpr auto isRecordRef<RecordRef<View, BoundRecordCoord, OwnView>> = true;

    /// Returns a \ref One with the same record dimension as the given record ref, with values copyied from rr.
    template<typename View, typename BoundRecordCoord, bool OwnView>
    LLAMA_FN_HOST_ACC_INLINE auto copyRecord(const RecordRef<View, BoundRecordCoord, OwnView>& rr)
    {
        using RecordDim = typename RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim;
        One<RecordDim> temp;
        temp = rr;
        return temp;
    }

    namespace internal
    {
        template<
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefArithOperator(
            LeftRecord& left,
            const RecordRef<RightView, RightBoundRecordDim, RightOwnView>& right) -> LeftRecord&
        {
            using RightRecord = RecordRef<RightView, RightBoundRecordDim, RightOwnView>;
            // if the record dimension left and right is the same, a single loop is enough and no tag check is needed.
            // this safes a lot of compilation time.
            if constexpr(std::is_same_v<
                             typename LeftRecord::AccessibleRecordDim,
                             typename RightRecord::AccessibleRecordDim>)
            {
                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE
                                                                           { Functor{}(left(rc), right(rc)); });
            }
            else
            {
                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
                    [&](auto leftRC) LLAMA_LAMBDA_INLINE
                    {
                        using LeftInnerCoord = decltype(leftRC);
                        forEachLeafCoord<typename RightRecord::AccessibleRecordDim>(
                            [&](auto rightRC) LLAMA_LAMBDA_INLINE
                            {
                                using RightInnerCoord = decltype(rightRC);
                                if constexpr(hasSameTags<
                                                 typename LeftRecord::AccessibleRecordDim,
                                                 LeftInnerCoord,
                                                 typename RightRecord::AccessibleRecordDim,
                                                 RightInnerCoord>)
                                {
                                    Functor{}(left(leftRC), right(rightRC));
                                }
                            });
                    });
            }
            return left;
        }

        template<typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefArithOperator(LeftRecord& left, const T& right) -> LeftRecord&
        {
            forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>([&](auto leftRC) LLAMA_LAMBDA_INLINE
                                                                       { Functor{}(left(leftRC), right); });
            return left;
        }

        template<
            typename Functor,
            typename LeftRecord,
            typename RightView,
            typename RightBoundRecordDim,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefRelOperator(
            const LeftRecord& left,
            const RecordRef<RightView, RightBoundRecordDim, RightOwnView>& right) -> bool
        {
            using RightRecord = RecordRef<RightView, RightBoundRecordDim, RightOwnView>;
            bool result = true;
            // if the record dimension left and right is the same, a single loop is enough and no tag check is needed.
            // this safes a lot of compilation time.
            if constexpr(std::is_same_v<
                             typename LeftRecord::AccessibleRecordDim,
                             typename RightRecord::AccessibleRecordDim>)
            {
                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
                    [&](auto rc) LLAMA_LAMBDA_INLINE { result &= Functor{}(left(rc), right(rc)); });
            }
            else
            {
                forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
                    [&](auto leftRC) LLAMA_LAMBDA_INLINE
                    {
                        using LeftInnerCoord = decltype(leftRC);
                        forEachLeafCoord<typename RightRecord::AccessibleRecordDim>(
                            [&](auto rightRC) LLAMA_LAMBDA_INLINE
                            {
                                using RightInnerCoord = decltype(rightRC);
                                if constexpr(hasSameTags<
                                                 typename LeftRecord::AccessibleRecordDim,
                                                 LeftInnerCoord,
                                                 typename RightRecord::AccessibleRecordDim,
                                                 RightInnerCoord>)
                                {
                                    result &= Functor{}(left(leftRC), right(rightRC));
                                }
                            });
                    });
            }
            return result;
        }

        template<typename Functor, typename LeftRecord, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto recordRefRelOperator(const LeftRecord& left, const T& right) -> bool
        {
            bool result = true;
            forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>([&](auto leftRC) LLAMA_LAMBDA_INLINE
                                                                       { result &= Functor{}(left(leftRC), right); });
            return result;
        }

        struct Assign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) = b;
            }
        };

        struct PlusAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) += b;
            }
        };

        struct MinusAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) -= b;
            }
        };

        struct MultiplyAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) *= b;
            }
        };

        struct DivideAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) /= b;
            }
        };

        struct ModuloAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE auto operator()(A&& a, const B& b) const -> decltype(auto)
            {
                return std::forward<A>(a) %= b;
            }
        };

        template<
            typename ProxyReference,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<ProxyReference>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(ProxyReference&& leaf, T) -> ProxyReference
        {
            return leaf;
        }

        template<
            typename TWithOptionalConst,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<TWithOptionalConst>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(TWithOptionalConst& leaf, T)
            -> std::reference_wrapper<TWithOptionalConst>
        {
            return leaf;
        }

        template<typename RecordRef, typename T, std::size_t N, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImplForArray(RecordRef&& vd, T (&&)[N], std::index_sequence<Is...>)
        {
            return std::make_tuple(asTupleImpl(vd(RecordCoord<Is>{}), T{})...);
        }

        template<typename RecordRef, typename T, std::size_t N>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(RecordRef&& vd, T (&&a)[N])
        {
            return asTupleImplForArray(std::forward<RecordRef>(vd), std::move(a), std::make_index_sequence<N>{});
        }

        template<typename RecordRef, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(RecordRef&& vd, Record<Fields...>)
        {
            return std::make_tuple(asTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template<
            typename ProxyReference,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<ProxyReference>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(ProxyReference&& leaf, T) -> std::tuple<ProxyReference>
        {
            static_assert(!std::is_reference_v<ProxyReference>);
            return {std::move(leaf)}; // NOLINT(bugprone-move-forwarding-reference)
        }

        template<
            typename TWithOptionalConst,
            typename T,
            std::enable_if_t<!isRecordRef<std::decay_t<TWithOptionalConst>>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(TWithOptionalConst& leaf, T) -> std::tuple<TWithOptionalConst&>
        {
            return {leaf};
        }

        template<typename RecordRef, typename T, std::size_t N, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImplForArray(RecordRef&& vd, T (&&)[N], std::index_sequence<Is...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(RecordCoord<Is>{}), T{})...);
        }

        template<typename RecordRef, typename T, std::size_t N>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(RecordRef&& vd, T (&&a)[N])
        {
            return asFlatTupleImplForArray(std::forward<RecordRef>(vd), std::move(a), std::make_index_sequence<N>{});
        }

        template<typename RecordRef, typename... Fields>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(RecordRef&& vd, Record<Fields...>)
        {
            return std::tuple_cat(asFlatTupleImpl(vd(GetFieldTag<Fields>{}), GetFieldType<Fields>{})...);
        }

        template<typename T, typename = void>
        constexpr inline auto isTupleLike = false;

        // get<I>(t) and std::tuple_size<T> must be available
        using std::get; // make sure a get<0>() can be found, so the compiler can compile the trait
        template<typename T>
        constexpr inline auto
            isTupleLike<T, std::void_t<decltype(get<0>(std::declval<T>())), std::tuple_size<T>>> = true;

        template<typename... Ts>
        constexpr inline auto dependentFalse = false;

        template<typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>);

        template<typename T1, typename T2>
        LLAMA_FN_HOST_ACC_INLINE void assignTupleElement(T1&& dst, T2&& src)
        {
            if constexpr(isTupleLike<std::decay_t<T1>> && isTupleLike<std::decay_t<T2>>)
            {
                static_assert(std::tuple_size_v<std::decay_t<T1>> == std::tuple_size_v<std::decay_t<T2>>);
                assignTuples(dst, src, std::make_index_sequence<std::tuple_size_v<std::decay_t<T1>>>{});
            }
            else if constexpr(!isTupleLike<std::decay_t<T1>> && !isTupleLike<std::decay_t<T2>>)
                std::forward<T1>(dst) = std::forward<T2>(src);
            else
                static_assert(
                    dependentFalse<T1, T2>,
                    "Elements to assign are not tuple/tuple or non-tuple/non-tuple.");
        }

        template<typename Tuple1, typename Tuple2, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE void assignTuples(Tuple1&& dst, Tuple2&& src, std::index_sequence<Is...>)
        {
            static_assert(std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>);
            using std::get;
            (assignTupleElement(get<Is>(std::forward<Tuple1>(dst)), get<Is>(std::forward<Tuple2>(src))), ...);
        }

        template<typename T, typename Tuple, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeFromTuple(Tuple&& src, std::index_sequence<Is...>)
        {
            using std::get;
            return T{get<Is>(src)...}; // no forward of src, since we call get multiple times on it
        }

        template<typename T, typename SFINAE, typename... Args>
        constexpr inline auto isDirectListInitializableImpl = false;

        template<typename T, typename... Args>
        constexpr inline auto
            isDirectListInitializableImpl<T, std::void_t<decltype(T{std::declval<Args>()...})>, Args...> = true;

        template<typename T, typename... Args>
        constexpr inline auto isDirectListInitializable = isDirectListInitializableImpl<T, void, Args...>;

        template<typename T, typename Tuple>
        constexpr inline auto isDirectListInitializableFromTuple = false;

        template<typename T, template<typename...> typename Tuple, typename... Args>
        constexpr inline auto
            isDirectListInitializableFromTuple<T, Tuple<Args...>> = isDirectListInitializable<T, Args...>;
    } // namespace internal

    /// Record reference type returned by \ref View after resolving an array dimensions coordinate or partially
    /// resolving a \ref RecordCoord. A record reference does not hold data itself, it just binds enough information
    /// (array dimensions coord and partial record coord) to retrieve it later from a \ref View. Records references
    /// should not be created by the user. They are returned from various access functions in \ref View and RecordRef
    /// itself.
    template<typename TView, typename TBoundRecordCoord, bool OwnView>
    struct RecordRef : private TView::Mapping::ArrayIndex
    {
        using View = TView; ///< View this record reference points into.
        using BoundRecordCoord
            = TBoundRecordCoord; ///< Record coords into View::RecordDim which are already bound by this RecordRef.

    private:
        using ArrayIndex = typename View::Mapping::ArrayIndex;
        using RecordDim = typename View::Mapping::RecordDim;

        std::conditional_t<OwnView, View, View&> view;

    public:
        /// Subtree of the record dimension of View starting at BoundRecordCoord. If BoundRecordCoord is
        /// `RecordCoord<>` (default) AccessibleRecordDim is the same as `Mapping::RecordDim`.
        using AccessibleRecordDim = GetType<RecordDim, BoundRecordCoord>;

        /// Creates an empty RecordRef. Only available for if the view is owned. Used by llama::One.
        LLAMA_FN_HOST_ACC_INLINE RecordRef()
            /* requires(OwnView) */
            : ArrayIndex{}
            , view{allocViewStack<0, RecordDim>()}
        {
            static_assert(OwnView, "The default constructor of RecordRef is only available if it owns the view.");
        }

        LLAMA_FN_HOST_ACC_INLINE
        RecordRef(ArrayIndex ai, std::conditional_t<OwnView, View&&, View&> view)
            : ArrayIndex{ai}
            , view{static_cast<decltype(view)>(view)}
        {
        }

        RecordRef(const RecordRef&) = default;

        // NOLINTNEXTLINE(cert-oop54-cpp)
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const RecordRef& other) -> RecordRef&
        {
            // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
            return this->operator=<RecordRef>(other);
        }

        RecordRef(RecordRef&&) noexcept = default;
        auto operator=(RecordRef&&) noexcept -> RecordRef& = default;

        ~RecordRef() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr auto arrayIndex() const -> ArrayIndex
        {
            return static_cast<const ArrayIndex&>(*this);
        }

        /// Create a RecordRef from a different RecordRef. Only available for if the view is owned. Used by
        /// llama::One.
        template<typename OtherView, typename OtherBoundRecordCoord, bool OtherOwnView>
        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        LLAMA_FN_HOST_ACC_INLINE RecordRef(const RecordRef<OtherView, OtherBoundRecordCoord, OtherOwnView>& recordRef)
            /* requires(OwnView) */
            : RecordRef()
        {
            static_assert(
                OwnView,
                "The copy constructor of RecordRef from a different RecordRef is only available if it owns "
                "the "
                "view.");
            *this = recordRef;
        }

        // TODO(bgruber): unify with previous in C++20 and use explicit(cond)
        /// Create a RecordRef from a scalar. Only available for if the view is owned. Used by llama::One.
        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE explicit RecordRef(const T& scalar)
            /* requires(OwnView) */
            : RecordRef()
        {
            static_assert(
                OwnView,
                "The constructor of RecordRef from a scalar is only available if it owns the view.");
            *this = scalar;
        }

        /// Access a record in the record dimension underneath the current record reference using a \ref RecordCoord.
        /// If the access resolves to a leaf, an l-value reference to a variable inside the \ref View storage is
        /// returned, otherwise another RecordRef.
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            using AccessedType = GetType<RecordDim, AbsolutCoord>;
            if constexpr(isRecord<AccessedType> || internal::IsBoundedArray<AccessedType>::value)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return RecordRef<const View, AbsolutCoord>{arrayIndex(), this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayIndex(), AbsolutCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...>) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundRecordCoord, RecordCoord<Coord...>>;
            using AccessedType = GetType<RecordDim, AbsolutCoord>;
            if constexpr(isRecord<AccessedType> || internal::IsBoundedArray<AccessedType>::value)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return RecordRef<View, AbsolutCoord>{arrayIndex(), this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(arrayIndex(), AbsolutCoord{});
            }
        }

        /// Access a record in the record dimension underneath the current record reference using a series of tags. If
        /// the access resolves to a leaf, an l-value reference to a variable inside the \ref View storage is returned,
        /// otherwise another RecordRef.
        template<typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template<typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
        {
            using RecordCoord = GetCoordFromTags<AccessibleRecordDim, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(RecordCoord{});
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const T& other) -> RecordRef&
        {
            // NOLINTNEXTLINE(cppcoreguidelines-c-copy-assignment-signature,misc-unconventional-assign-operator)
            return internal::recordRefArithOperator<internal::Assign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator+=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::PlusAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator-=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::MinusAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator*=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::MultiplyAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator/=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::DivideAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator%=(const T& other) -> RecordRef&
        {
            return internal::recordRefArithOperator<internal::ModuloAssign>(*this, other);
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) += t;
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const T& t, const RecordRef& vd)
        {
            return vd + t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator-(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) -= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) *= t;
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const T& t, const RecordRef& vd)
        {
            return vd * t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator/(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) /= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator%(const RecordRef& vd, const T& t)
        {
            return copyRecord(vd) %= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::equal_to<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const T& t, const RecordRef& vd) -> bool
        {
            return vd == t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::not_equal_to<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const RecordRef& vd) -> bool
        {
            return vd != t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::less<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const T& t, const RecordRef& vd) -> bool
        {
            return vd > t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::less_equal<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const T& t, const RecordRef& vd) -> bool
        {
            return vd >= t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::greater<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const T& t, const RecordRef& vd) -> bool
        {
            return vd < t;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const RecordRef& vd, const T& t) -> bool
        {
            return internal::recordRefRelOperator<std::greater_equal<>>(vd, t);
        }

        template<typename T, typename = std::enable_if_t<!isRecordRef<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const T& t, const RecordRef& vd) -> bool
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

        template<std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template<std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> decltype(auto)
        {
            return operator()(RecordCoord<I>{});
        }

        template<typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this RecordRef recursively represents "
                "like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        template<typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() const -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this RecordRef recursively represents "
                "like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        struct Loader
        {
            RecordRef& vd;

            template<typename T>
            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE operator T()
            {
                return vd.loadAs<T>();
            }
        };

        struct LoaderConst
        {
            const RecordRef& vd;

            template<typename T>
            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
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

        template<typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE void store(const TupleLike& t)
        {
            internal::assignTuples(asTuple(), t, std::make_index_sequence<std::tuple_size_v<TupleLike>>{});
        }

        // swap for equal RecordRef
        LLAMA_FN_HOST_ACC_INLINE friend void swap(
            std::conditional_t<OwnView, RecordRef&, RecordRef> a,
            std::conditional_t<OwnView, RecordRef&, RecordRef> b) noexcept
        {
            forEachLeafCoord<AccessibleRecordDim>(
                [&](auto rc) LLAMA_LAMBDA_INLINE
                {
                    using std::swap;
                    swap(a(rc), b(rc));
                });
        }
    };

    // swap for heterogeneous RecordRef
    template<
        typename ViewA,
        typename BoundRecordDimA,
        bool OwnViewA,
        typename ViewB,
        typename BoundRecordDimB,
        bool OwnViewB>
    LLAMA_FN_HOST_ACC_INLINE auto swap(
        RecordRef<ViewA, BoundRecordDimA, OwnViewA>& a,
        RecordRef<ViewB, BoundRecordDimB, OwnViewB>& b) noexcept
        -> std::enable_if_t<std::is_same_v<
            typename RecordRef<ViewA, BoundRecordDimA, OwnViewA>::AccessibleRecordDim,
            typename RecordRef<ViewB, BoundRecordDimB, OwnViewB>::AccessibleRecordDim>>
    {
        using LeftRecord = RecordRef<ViewA, BoundRecordDimA, OwnViewA>;
        forEachLeafCoord<typename LeftRecord::AccessibleRecordDim>(
            [&](auto rc) LLAMA_LAMBDA_INLINE
            {
                using std::swap;
                swap(a(rc), b(rc));
            });
    }

    template<typename View, typename BoundRecordCoord, bool OwnView>
    auto operator<<(std::ostream& os, const RecordRef<View, BoundRecordCoord, OwnView>& vr) -> std::ostream&
    {
        using RecordDim = typename RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim;
        os << "{";
        if constexpr(std::is_array_v<RecordDim>)
        {
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<std::extent_v<RecordDim>>>(
                [&](auto ic)
                {
                    constexpr std::size_t i = decltype(ic)::value;
                    if(i > 0)
                        os << ", ";
                    os << '[' << i << ']' << ": " << vr(RecordCoord<i>{});
                });
        }
        else
        {
            boost::mp11::mp_for_each<boost::mp11::mp_iota<boost::mp11::mp_size<RecordDim>>>(
                [&](auto ic)
                {
                    constexpr std::size_t i = decltype(ic)::value;
                    if(i > 0)
                        os << ", ";
                    using Field = boost::mp11::mp_at_c<RecordDim, i>;
                    os << structName<GetFieldTag<Field>>() << ": " << vr(RecordCoord<i>{});
                });
        }
        os << "}";
        return os;
    }

    template<typename RecordRefFwd, typename Functor>
    LLAMA_FN_HOST_ACC_INLINE constexpr void forEachLeaf(RecordRefFwd&& vr, Functor&& functor)
    {
        using RecordRef = std::remove_reference_t<RecordRefFwd>;
        LLAMA_FORCE_INLINE_RECURSIVE
        forEachLeafCoord<typename RecordRef::AccessibleRecordDim>(
            [functor = std::forward<Functor>(functor), &vr = vr](auto rc)
                LLAMA_LAMBDA_INLINE_WITH_SPECIFIERS(constexpr mutable) { std::forward<Functor>(functor)(vr(rc)); });
    }

    namespace internal
    {
        // gets the value type for a given T, where T models a reference type. T is either an l-value reference, a
        // proxy reference or a RecordRef
        template<typename T, typename = void>
        struct ValueOf
        {
            static_assert(sizeof(T) == 0, "T does not model a reference");
        };

        template<typename T>
        struct ValueOf<T, std::enable_if_t<isRecordRef<T>>>
        {
            using type = One<typename T::AccessibleRecordDim>;
        };

#ifdef __cpp_lib_concepts
        template<ProxyReference T>
#else
        template<typename T>
#endif
        struct ValueOf<T, std::enable_if_t<isProxyReference<T>>>
        {
            using type = typename T::value_type;
        };

        template<typename T>
        struct ValueOf<T&>
        {
            using type = T;
        };
    } // namespace internal

    /// Scope guard type. ScopedUpdate takes a copy of a value through a reference and stores it internally during
    /// construction. The stored value is written back when ScopedUpdate is destroyed. ScopedUpdate tries to act like
    /// the stored value as much as possible, exposing member functions of the stored value and acting like a proxy
    /// reference if the stored value is a primitive type.
    template<typename Reference, typename = void>
    struct ScopedUpdate : internal::ValueOf<Reference>::type
    {
        using value_type = typename internal::ValueOf<Reference>::type;

        /// Loads a copy of the value referenced by r. Stores r and the loaded value.
        LLAMA_FN_HOST_ACC_INLINE explicit ScopedUpdate(Reference r) : value_type(r), ref(r)
        {
        }

        ScopedUpdate(const ScopedUpdate&) = delete;
        auto operator=(const ScopedUpdate&) -> ScopedUpdate& = delete;

        ScopedUpdate(ScopedUpdate&&) noexcept = default;
        auto operator=(ScopedUpdate&&) noexcept -> ScopedUpdate& = default;

        using value_type::operator=;

        /// Stores the internally stored value back to the referenced value.
        LLAMA_FN_HOST_ACC_INLINE ~ScopedUpdate()
        {
            ref = static_cast<value_type&>(*this);
        }

        /// Get access to the stored value.
        LLAMA_FN_HOST_ACC_INLINE auto get() -> value_type&
        {
            return *this;
        }

        /// Get access to the stored value.
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> const value_type&
        {
            return *this;
        }

    private:
        Reference ref;
    };

    template<typename Reference>
    struct ScopedUpdate<
        Reference,
        std::enable_if_t<std::is_fundamental_v<typename internal::ValueOf<Reference>::type>>>
        : ProxyRefOpMixin<ScopedUpdate<Reference>, typename internal::ValueOf<Reference>::type>
    {
        using value_type = typename internal::ValueOf<Reference>::type;

        LLAMA_FN_HOST_ACC_INLINE explicit ScopedUpdate(Reference r) : value(r), ref(r)
        {
        }

        ScopedUpdate(const ScopedUpdate&) = delete;
        auto operator=(const ScopedUpdate&) -> ScopedUpdate& = delete;

        ScopedUpdate(ScopedUpdate&&) noexcept = default;
        auto operator=(ScopedUpdate&&) noexcept -> ScopedUpdate& = default;

        LLAMA_FN_HOST_ACC_INLINE auto get() -> value_type&
        {
            return value;
        }

        LLAMA_FN_HOST_ACC_INLINE auto get() const -> const value_type&
        {
            return value;
        }

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        LLAMA_FN_HOST_ACC_INLINE operator const value_type&() const
        {
            return value;
        }

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        LLAMA_FN_HOST_ACC_INLINE operator value_type&()
        {
            return value;
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type v) -> ScopedUpdate&
        {
            value = v;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE ~ScopedUpdate()
        {
            ref = value;
        }

    private:
        value_type value;
        Reference ref;
    };

    namespace internal
    {
        template<typename T, typename = void>
        struct ReferenceTo
        {
            using type = T&;
        };

        template<typename T>
        struct ReferenceTo<T, std::enable_if_t<isRecordRef<T> && !isOne<T>>>
        {
            using type = T;
        };

#ifdef __cpp_lib_concepts
        template<ProxyReference T>
#else
        template<typename T>
#endif
        struct ReferenceTo<T, std::enable_if_t<isProxyReference<T>>>
        {
            using type = T;
        };
    } // namespace internal

    template<typename T>
    ScopedUpdate(T) -> ScopedUpdate<typename internal::ReferenceTo<std::remove_reference_t<T>>::type>;
} // namespace llama

template<typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_size<llama::RecordRef<View, BoundRecordCoord, OwnView>>
    : boost::mp11::mp_size<typename llama::RecordRef<View, BoundRecordCoord, OwnView>::AccessibleRecordDim>
{
};

template<std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, llama::RecordRef<View, BoundRecordCoord, OwnView>>
{
    using type = decltype(std::declval<llama::RecordRef<View, BoundRecordCoord, OwnView>>().template get<I>());
};

template<std::size_t I, typename View, typename BoundRecordCoord, bool OwnView>
struct std::tuple_element<I, const llama::RecordRef<View, BoundRecordCoord, OwnView>>
{
    using type = decltype(std::declval<const llama::RecordRef<View, BoundRecordCoord, OwnView>>().template get<I>());
};

#if CAN_USE_RANGES
template<
    typename ViewA,
    typename BoundA,
    bool OwnA,
    typename ViewB,
    typename BoundB,
    bool OwnB,
    template<class>
    class TQual,
    template<class>
    class UQual>
struct std::
    basic_common_reference<llama::RecordRef<ViewA, BoundA, OwnA>, llama::RecordRef<ViewB, BoundB, OwnB>, TQual, UQual>
{
    using type = std::enable_if_t<
        std::is_same_v<
            typename llama::RecordRef<ViewA, BoundA, OwnA>::AccessibleRecordDim,
            typename llama::RecordRef<ViewB, BoundB, OwnB>::AccessibleRecordDim>,
        llama::One<typename ViewA::RecordDim>>;
};
#endif

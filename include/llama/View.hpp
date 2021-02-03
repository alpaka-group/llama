// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Allocators.hpp"
#include "Array.hpp"
#include "Concepts.hpp"
#include "Core.hpp"
#include "macros.hpp"
#include "mapping/One.hpp"

#include <boost/preprocessor/cat.hpp>
#include <type_traits>

namespace llama
{
#ifdef __cpp_concepts
    template <typename T_Mapping, StorageBlob BlobType>
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
            return {alloc(mapping.getBlobSize(Is))...};
        }
    } // namespace internal

    /// Creates a view based on the given mapping, e.g. \ref mapping::AoS or \ref mapping::SoA. For allocating the
    /// view's underlying memory, the specified allocator callable is used (or the default one, which is \ref
    /// allocator::Vector). The allocator callable is called with the size of bytes to allocate for each blob of the
    /// mapping. This function is the preferred way to create a \ref View.
#ifdef __cpp_concepts
    template <typename Mapping, BlobAllocator Allocator = allocator::Vector<>>
#else
    template <typename Mapping, typename Allocator = allocator::Vector<>>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator>>
    {
        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
        return {std::move(mapping), std::move(blobs)};
    }

    /// Allocates a \ref View holding a single datum backed by stack memory
    /// (\ref allocator::Stack).
    /// \tparam Dim Dimension of the \ref ArrayDomain of the \ref View.
    template <std::size_t Dim, typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
    {
        using Mapping = llama::mapping::One<ArrayDomain<Dim>, DatumDomain>;
        return allocView(Mapping{}, llama::allocator::Stack<sizeOf<DatumDomain>>{});
    }

    template <typename View, typename BoundDatumDomain = DatumCoord<>, bool OwnView = false>
    struct VirtualDatum;

    template <typename View>
    inline constexpr auto is_VirtualDatum = false;

    template <typename View, typename BoundDatumDomain, bool OwnView>
    inline constexpr auto is_VirtualDatum<VirtualDatum<View, BoundDatumDomain, OwnView>> = true;

    /// A \ref VirtualDatum that owns and holds a single value.
    template <typename DatumDomain>
    using One = VirtualDatum<decltype(allocViewStack<1, DatumDomain>()), DatumCoord<>, true>;

    /// Creates a single \ref VirtualDatum owning a view with stack memory and
    /// copies all values from an existing \ref VirtualDatum.
    template <typename VirtualDatum>
    LLAMA_FN_HOST_ACC_INLINE auto copyVirtualDatumStack(const VirtualDatum& vd) -> decltype(auto)
    {
        One<typename VirtualDatum::AccessibleDatumDomain> temp;
        temp = vd;
        return temp;
    }

    namespace internal
    {
        template <
            typename Functor,
            typename LeftDatum,
            typename RightView,
            typename RightBoundDatumDomain,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumArithOperator(
            LeftDatum& left,
            const VirtualDatum<RightView, RightBoundDatumDomain, RightOwnView>& right) -> LeftDatum&
        {
            using RightDatum = VirtualDatum<RightView, RightBoundDatumDomain, RightOwnView>;
            forEachLeave<typename LeftDatum::AccessibleDatumDomain>([&](auto leftCoord) {
                using LeftInnerCoord = decltype(leftCoord);
                forEachLeave<typename RightDatum::AccessibleDatumDomain>([&](auto rightCoord) {
                    using RightInnerCoord = decltype(rightCoord);
                    if constexpr (hasSameTags<
                                      typename LeftDatum::AccessibleDatumDomain,
                                      LeftInnerCoord,
                                      typename RightDatum::AccessibleDatumDomain,
                                      RightInnerCoord>)
                    {
                        Functor{}(left(leftCoord), right(rightCoord));
                    }
                });
            });
            return left;
        }

        template <typename Functor, typename LeftDatum, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumArithOperator(LeftDatum& left, const T& right) -> LeftDatum&
        {
            forEachLeave<typename LeftDatum::AccessibleDatumDomain>(
                [&](auto leftCoord) { Functor{}(left(leftCoord), right); });
            return left;
        }

        template <
            typename Functor,
            typename LeftDatum,
            typename RightView,
            typename RightBoundDatumDomain,
            bool RightOwnView>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumRelOperator(
            const LeftDatum& left,
            const VirtualDatum<RightView, RightBoundDatumDomain, RightOwnView>& right) -> bool
        {
            using RightDatum = VirtualDatum<RightView, RightBoundDatumDomain, RightOwnView>;
            bool result = true;
            forEachLeave<typename LeftDatum::AccessibleDatumDomain>([&](auto leftCoord) {
                using LeftInnerCoord = decltype(leftCoord);
                forEachLeave<typename RightDatum::AccessibleDatumDomain>([&](auto rightCoord) {
                    using RightInnerCoord = decltype(rightCoord);
                    if constexpr (hasSameTags<
                                      typename LeftDatum::AccessibleDatumDomain,
                                      LeftInnerCoord,
                                      typename RightDatum::AccessibleDatumDomain,
                                      RightInnerCoord>)
                    {
                        result &= Functor{}(left(leftCoord), right(rightCoord));
                    }
                });
            });
            return result;
        }

        template <typename Functor, typename LeftDatum, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumRelOperator(const LeftDatum& left, const T& right) -> bool
        {
            bool result = true;
            forEachLeave<typename LeftDatum::AccessibleDatumDomain>([&](auto leftCoord) {
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
            enable_if_t<!is_VirtualDatum<std::decay_t<TWithOptionalConst>>, std::reference_wrapper<TWithOptionalConst>>
        {
            return leaf;
        }

        template <typename VirtualDatum, typename... Elements>
        LLAMA_FN_HOST_ACC_INLINE auto asTupleImpl(VirtualDatum&& vd, DatumStruct<Elements...>)
        {
            return std::make_tuple(asTupleImpl(vd(GetDatumElementTag<Elements>{}), GetDatumElementType<Elements>{})...);
        }

        template <typename TWithOptionalConst, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(TWithOptionalConst& leaf, T)
            -> std::enable_if_t<!is_VirtualDatum<std::decay_t<TWithOptionalConst>>, std::tuple<TWithOptionalConst&>>
        {
            return {leaf};
        }

        template <typename VirtualDatum, typename... Elements>
        LLAMA_FN_HOST_ACC_INLINE auto asFlatTupleImpl(VirtualDatum&& vd, DatumStruct<Elements...>)
        {
            return std::tuple_cat(
                asFlatTupleImpl(vd(GetDatumElementTag<Elements>{}), GetDatumElementType<Elements>{})...);
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

    /// Virtual data type returned by \ref View after resolving a user domain
    /// coordinate or partially resolving a \ref DatumCoord. A virtual datum
    /// does not hold data itself (thus named "virtual"), it just binds enough
    /// information (user domain coord and partial datum coord) to retrieve it
    /// from a \ref View later. Virtual datums should not be created by the
    /// user. They are returned from various access functions in \ref View and
    /// VirtualDatum itself.
    template <typename T_View, typename BoundDatumDomain, bool OwnView>
    struct VirtualDatum
    {
        using View = T_View; ///< View this virtual datum points into.

    private:
        using ArrayDomain = typename View::Mapping::ArrayDomain;
        using DatumDomain = typename View::Mapping::DatumDomain;

        const ArrayDomain userDomainPos;
        std::conditional_t<OwnView, View, View&> view;

    public:
        /// Subtree of the datum domain of View starting at
        /// BoundDatumDomain. If BoundDatumDomain is `DatumCoord<>` (default)
        /// AccessibleDatumDomain is the same as `Mapping::DatumDomain`.
        using AccessibleDatumDomain = GetType<DatumDomain, BoundDatumDomain>;

        LLAMA_FN_HOST_ACC_INLINE VirtualDatum()
            /* requires(OwnView) */
            : userDomainPos({})
            , view{allocViewStack<1, DatumDomain>()}
        {
            static_assert(OwnView, "The default constructor of VirtualDatum is only available if the ");
        }

        LLAMA_FN_HOST_ACC_INLINE
        VirtualDatum(ArrayDomain userDomainPos, std::conditional_t<OwnView, View&&, View&> view)
            : userDomainPos(userDomainPos)
            , view{static_cast<decltype(view)>(view)}
        {
        }

        VirtualDatum(const VirtualDatum&) = default;
        VirtualDatum(VirtualDatum&&) = default;

        /// Access a datum in the datum domain underneath the current virtual
        /// datum using a \ref DatumCoord. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual datum.
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoord<Coord...> = {}) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
            if constexpr (isDatumStruct<GetType<DatumDomain, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<const View, AbsolutCoord>{userDomainPos, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, AbsolutCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoord<Coord...> coord = {}) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
            if constexpr (isDatumStruct<GetType<DatumDomain, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<View, AbsolutCoord>{userDomainPos, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, AbsolutCoord{});
            }
        }

        /// Access a datum in the datum domain underneath the current virtual
        /// datum using a series of tags. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual datum.
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) const -> decltype(auto)
        {
            using DatumCoord = GetCoordFromTagsRelative<DatumDomain, BoundDatumDomain, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(DatumCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Tags...) -> decltype(auto)
        {
            using DatumCoord = GetCoordFromTagsRelative<DatumDomain, BoundDatumDomain, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return operator()(DatumCoord{});
        }

        // we need this one to disable the compiler generated copy assignment
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const VirtualDatum& other) -> VirtualDatum&
        {
            return this->operator=<VirtualDatum>(other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const T& other) -> VirtualDatum&
        {
            return internal::virtualDatumArithOperator<internal::Assign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator+=(const T& other) -> VirtualDatum&
        {
            return internal::virtualDatumArithOperator<internal::PlusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator-=(const T& other) -> VirtualDatum&
        {
            return internal::virtualDatumArithOperator<internal::MinusAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator*=(const T& other) -> VirtualDatum&
        {
            return internal::virtualDatumArithOperator<internal::MultiplyAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator/=(const T& other) -> VirtualDatum&
        {
            return internal::virtualDatumArithOperator<internal::DivideAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator%=(const T& other) -> VirtualDatum&
        {
            return internal::virtualDatumArithOperator<internal::ModuloAssign>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const VirtualDatum& vd, const T& t)
        {
            return copyVirtualDatumStack(vd) += t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator+(const T& t, const VirtualDatum& vd)
        {
            return vd + t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator-(const VirtualDatum& vd, const T& t)
        {
            return copyVirtualDatumStack(vd) -= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const VirtualDatum& vd, const T& t)
        {
            return copyVirtualDatumStack(vd) *= t;
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator*(const T& t, const VirtualDatum& vd)
        {
            return vd * t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator/(const VirtualDatum& vd, const T& t)
        {
            return copyVirtualDatumStack(vd) /= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator%(const VirtualDatum& vd, const T& t)
        {
            return copyVirtualDatumStack(vd) %= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const VirtualDatum& vd, const T& t) -> bool
        {
            return internal::virtualDatumRelOperator<std::equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator==(const T& t, const VirtualDatum& vd) -> bool
        {
            return vd == t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const VirtualDatum& vd, const T& t) -> bool
        {
            return internal::virtualDatumRelOperator<std::not_equal_to<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator!=(const T& t, const VirtualDatum& vd) -> bool
        {
            return vd != t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const VirtualDatum& vd, const T& t) -> bool
        {
            return internal::virtualDatumRelOperator<std::less<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<(const T& t, const VirtualDatum& vd) -> bool
        {
            return vd > t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const VirtualDatum& vd, const T& t) -> bool
        {
            return internal::virtualDatumRelOperator<std::less_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator<=(const T& t, const VirtualDatum& vd) -> bool
        {
            return vd >= t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const VirtualDatum& vd, const T& t) -> bool
        {
            return internal::virtualDatumRelOperator<std::greater<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>(const T& t, const VirtualDatum& vd) -> bool
        {
            return vd < t;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const VirtualDatum& vd, const T& t) -> bool
        {
            return internal::virtualDatumRelOperator<std::greater_equal<>>(vd, t);
        }

        template <typename T, typename = std::enable_if_t<!is_VirtualDatum<T>>>
        LLAMA_FN_HOST_ACC_INLINE friend auto operator>=(const T& t, const VirtualDatum& vd) -> bool
        {
            return vd <= t;
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple()
        {
            return internal::asTupleImpl(*this, AccessibleDatumDomain{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asTuple() const
        {
            return internal::asTupleImpl(*this, AccessibleDatumDomain{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple()
        {
            return internal::asFlatTupleImpl(*this, AccessibleDatumDomain{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto asFlatTuple() const
        {
            return internal::asFlatTupleImpl(*this, AccessibleDatumDomain{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() -> decltype(auto)
        {
            return operator()(DatumCoord<I>{});
        }

        template <std::size_t I>
        LLAMA_FN_HOST_ACC_INLINE auto get() const -> decltype(auto)
        {
            return operator()(DatumCoord<I>{});
        }

        template <typename TupleLike>
        LLAMA_FN_HOST_ACC_INLINE auto loadAs() -> TupleLike
        {
            static_assert(
                internal::isDirectListInitializableFromTuple<TupleLike, decltype(asFlatTuple())>,
                "TupleLike must be constructible from as many values as this VirtualDatum recursively represents like "
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
                "TupleLike must be constructible from as many values as this VirtualDatum recursively represents like "
                "this: TupleLike{values...}");
            return internal::makeFromTuple<TupleLike>(
                asFlatTuple(),
                std::make_index_sequence<std::tuple_size_v<decltype(asFlatTuple())>>{});
        }

        struct Loader
        {
            VirtualDatum& vd;

            template <typename T>
            LLAMA_FN_HOST_ACC_INLINE operator T()
            {
                return vd.loadAs<T>();
            }
        };

        struct LoaderConst
        {
            const VirtualDatum& vd;

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

template <typename View, typename BoundDatumDomain, bool OwnView>
struct std::tuple_size<llama::VirtualDatum<View, BoundDatumDomain, OwnView>>
    : boost::mp11::mp_size<typename llama::VirtualDatum<View, BoundDatumDomain, OwnView>::AccessibleDatumDomain>
{
};

template <std::size_t I, typename View, typename BoundDatumDomain, bool OwnView>
struct std::tuple_element<I, llama::VirtualDatum<View, BoundDatumDomain, OwnView>>
{
    using type = decltype(std::declval<llama::VirtualDatum<View, BoundDatumDomain, OwnView>>().template get<I>());
};

template <std::size_t I, typename View, typename BoundDatumDomain, bool OwnView>
struct std::tuple_element<I, const llama::VirtualDatum<View, BoundDatumDomain, OwnView>>
{
    using type = decltype(std::declval<const llama::VirtualDatum<View, BoundDatumDomain, OwnView>>().template get<I>());
};

namespace llama
{
    /// Central LLAMA class holding memory for storage and giving access to
    /// values stored there defined by a mapping. A view should be created using
    /// \ref allocView.
    /// \tparam T_Mapping The mapping used by the view to map accesses into
    /// memory.
    /// \tparam BlobType The storage type used by the view holding
    /// memory.
#ifdef __cpp_concepts
    template <typename T_Mapping, StorageBlob BlobType>
#else
    template <typename T_Mapping, typename BlobType>
#endif
    struct View
    {
        using Mapping = T_Mapping;
        using ArrayDomain = typename Mapping::ArrayDomain;
        using DatumDomain = typename Mapping::DatumDomain;
        using VirtualDatumType = VirtualDatum<View<Mapping, BlobType>>;
        using VirtualDatumTypeConst = VirtualDatum<const View<Mapping, BlobType>>;

        View() = default;

        LLAMA_FN_HOST_ACC_INLINE
        View(Mapping mapping, Array<BlobType, Mapping::blobCount> storageBlobs)
            : mapping(std::move(mapping))
            , storageBlobs(std::move(storageBlobs))
        {
        }

        /// Retrieves the \ref VirtualDatum at the given \ref ArrayDomain
        /// coordinate.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) const -> decltype(auto)
        {
            if constexpr (isDatumStruct<DatumDomain>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatumTypeConst{arrayDomain, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDomain, DatumCoord<>{});
            }
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) -> decltype(auto)
        {
            if constexpr (isDatumStruct<DatumDomain>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatumType{arrayDomain, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDomain, DatumCoord<>{});
            }
        }

        /// Retrieves the \ref VirtualDatum at the \ref ArrayDomain coordinate
        /// constructed from the passed component indices.
        template <typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ArrayDomain{indices...});
        }

        template <typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ArrayDomain{indices...});
        }

        /// Retrieves the \ref VirtualDatum at the \ref ArrayDomain coordinate
        /// constructed from the passed component indices.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDomain arrayDomain) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(arrayDomain);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDomain arrayDomain) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(arrayDomain);
        }

        /// Retrieves the \ref VirtualDatum at the 1D \ref ArrayDomain coordinate
        /// constructed from the passed index.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(index);
        }

        Mapping mapping;
        Array<BlobType, Mapping::blobCount> storageBlobs;

    private:
        template <typename T_View, typename T_BoundDatumDomain, bool OwnView>
        friend struct VirtualDatum;

        template <std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDomain arrayDomain, DatumCoord<Coords...> = {}) const -> const auto&
        {
            const auto [nr, offset] = mapping.template getBlobNrAndOffset<Coords...>(arrayDomain);
            using Type = GetType<DatumDomain, DatumCoord<Coords...>>;
            return reinterpret_cast<const Type&>(storageBlobs[nr][offset]);
        }

        template <std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDomain arrayDomain, DatumCoord<Coords...> coord = {}) -> auto&
        {
            const auto [nr, offset] = mapping.template getBlobNrAndOffset<Coords...>(arrayDomain);
            using Type = GetType<DatumDomain, DatumCoord<Coords...>>;
            return reinterpret_cast<Type&>(storageBlobs[nr][offset]);
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
        using ArrayDomain = typename Mapping::ArrayDomain; ///< user domain of the parent view
        using VirtualDatumType = typename ParentView::VirtualDatumType; ///< VirtualDatum type of the
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
            return parentView.template accessor<Coords...>(arrayDomain + offset);
        }

        template <std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDomain arrayDomain) -> auto&
        {
            return parentView.template accessor<Coords...>(arrayDomain + offset);
        }

        /// Same as \ref View::operator()(ArrayDomain), but shifted by the offset
        /// of this \ref VirtualView.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) const -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(arrayDomain + offset);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(arrayDomain + offset);
        }

        /// Same as corresponding operator in \ref View, but shifted by the
        /// offset of this \ref VirtualView.
        template <typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDomain{indices...} + offset);
        }

        template <typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDomain{indices...} + offset);
        }

        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoord<Coord...>&& dc = {}) const -> const auto&
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayDomain{});
        }

        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoord<Coord...>&& dc = {}) -> auto&
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

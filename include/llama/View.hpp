// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Allocators.hpp"
#include "Array.hpp"
#include "ForEach.hpp"
#include "Functions.hpp"
#include "macros.hpp"
#include "mapping/One.hpp"

#include <boost/preprocessor/cat.hpp>
#include <type_traits>

namespace llama
{
    template<typename Mapping, typename BlobType>
    struct View;

    namespace internal
    {
        template<typename Allocator>
        using AllocatorBlobType = decltype(std::declval<Allocator>()(0));

        template<typename Allocator, typename Mapping, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE static auto makeBlobArray(
            const Allocator & alloc,
            const Mapping & mapping,
            std::integer_sequence<std::size_t, Is...>)
            -> Array<AllocatorBlobType<Allocator>, Mapping::blobCount>
        {
            return {alloc(mapping.getBlobSize(Is))...};
        }
    }

    /// Creates a view based on the given mapping, e.g. \ref mapping::AoS or
    /// \ref mapping::SoA. For allocating the view's underlying memory, the
    /// specified allocator is used (or the default one, which is \ref
    /// allocator::Vector). This function is the preferred way to create a \ref
    /// View.
    template<typename Mapping, typename Allocator = allocator::Vector<>>
    LLAMA_FN_HOST_ACC_INLINE auto
    allocView(Mapping mapping = {}, const Allocator & alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator>>
    {
        auto blobs = internal::makeBlobArray<Allocator>(
            alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
        return {std::move(mapping), std::move(blobs)};
    }

    /// Allocates a \ref View holding a single datum backed by stack memory
    /// (\ref allocator::Stack).
    /// \tparam Dim Dimension of the \ref UserDomain of the \ref View.
    template<std::size_t Dim, typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
    {
        using Mapping = llama::mapping::One<UserDomain<Dim>, DatumDomain>;
        return allocView(
            Mapping{}, llama::allocator::Stack<sizeOf<DatumDomain>>{});
    }

    template<typename View>
    inline constexpr auto IsView = false;

    template<typename Mapping, typename BlobType>
    inline constexpr auto IsView<View<Mapping, BlobType>> = true;

    namespace internal
    {
        template<typename View>
        struct ViewByRefHolder
        {
            View & view;
        };

        template<typename View>
        struct ViewByValueHolder
        {
            View view;
        };
    }

    template<
        typename View,
        typename BoundDatumDomain = DatumCoord<>,
        bool OwnView = false>
    struct VirtualDatum;

    template<typename View>
    inline constexpr auto is_VirtualDatum = false;

    template<typename View, typename BoundDatumDomain, bool OwnView>
    inline constexpr auto
        is_VirtualDatum<VirtualDatum<View, BoundDatumDomain, OwnView>> = true;

    /// Creates a single \ref VirtualDatum owning a view with stack memory.
    template<typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto allocVirtualDatumStack() -> VirtualDatum<
        decltype(llama::allocViewStack<1, DatumDomain>()),
        DatumCoord<>,
        true>
    {
        return {UserDomain<1>{}, llama::allocViewStack<1, DatumDomain>()};
    }

    /// Creates a single \ref VirtualDatum owning a view with stack memory and
    /// copies all values from an existing \ref VirtualDatum.
    template<typename VirtualDatum>
    LLAMA_FN_HOST_ACC_INLINE auto copyVirtualDatumStack(const VirtualDatum & vd)
        -> decltype(auto)
    {
        auto temp = allocVirtualDatumStack<
            typename VirtualDatum::AccessibleDatumDomain>();
        temp = vd;
        return temp;
    }

    namespace internal
    {
        template<
            typename LeftDatum,
            typename RightDatum,
            typename Source,
            typename LeftOuterCoord,
            typename LeftInnerCoord,
            typename OP>
        struct GenericInnerFunctor
        {
            template<typename RightOuterCoord, typename RightInnerCoord>
            LLAMA_FN_HOST_ACC_INLINE void
            operator()(RightOuterCoord, RightInnerCoord)
            {
                if constexpr(hasSameTags<
                                 typename LeftDatum::AccessibleDatumDomain,
                                 LeftOuterCoord,
                                 LeftInnerCoord,
                                 typename RightDatum::AccessibleDatumDomain,
                                 RightOuterCoord,
                                 RightInnerCoord>)
                {
                    using Dst = Cat<LeftOuterCoord, LeftInnerCoord>;
                    using Src = Cat<RightOuterCoord, RightInnerCoord>;
                    OP{}(left(Dst()), right(Src()));
                }
            }
            LeftDatum & left;
            const RightDatum & right;
        };

        template<
            typename LeftDatum,
            typename RightDatum,
            typename SourceDatumCoord,
            typename OP>
        struct GenericFunctor
        {
            template<typename LeftOuterCoord, typename LeftInnerCoord>
            LLAMA_FN_HOST_ACC_INLINE void
            operator()(LeftOuterCoord, LeftInnerCoord)
            {
                forEach<typename RightDatum::AccessibleDatumDomain>(
                    GenericInnerFunctor<
                        LeftDatum,
                        RightDatum,
                        SourceDatumCoord,
                        LeftOuterCoord,
                        LeftInnerCoord,
                        OP>{left, right},
                    SourceDatumCoord{});
            }
            LeftDatum & left;
            const RightDatum & right;
        };

        template<typename LeftDatum, typename RightType, typename OP>
        struct GenericTypeFunctor
        {
            template<typename OuterCoord, typename InnerCoord>
            LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
            {
                using Dst = Cat<OuterCoord, InnerCoord>;
                OP{}(left(Dst()), right);
            }
            LeftDatum & left;
            const RightType & right;
        };
    }

    namespace internal
    {
        template<
            typename LeftDatum,
            typename LeftBase,
            typename LeftLocal,
            typename RightDatum,
            typename OP>
        struct GenericBoolInnerFunctor
        {
            template<typename OuterCoord, typename InnerCoord>
            LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
            {
                if constexpr(hasSameTags<
                                 typename LeftDatum::View::Mapping::DatumDomain,
                                 LeftBase,
                                 LeftLocal,
                                 typename RightDatum::View::Mapping::
                                     DatumDomain,
                                 OuterCoord,
                                 InnerCoord>)
                {
                    using Dst = Cat<LeftBase, LeftLocal>;
                    using Src = Cat<OuterCoord, InnerCoord>;
                    result &= OP{}(left(Dst()), right(Src()));
                }
            }
            const LeftDatum & left;
            const RightDatum & right;
            bool result;
        };

        template<
            typename LeftDatum,
            typename RightDatum,
            typename SourceDatumCoord,
            typename OP>
        struct GenericBoolFunctor
        {
            template<typename OuterCoord, typename InnerCoord>
            LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
            {
                GenericBoolInnerFunctor<
                    LeftDatum,
                    OuterCoord,
                    InnerCoord,
                    RightDatum,
                    OP>
                    functor{left, right, true};
                forEach<typename RightDatum::AccessibleDatumDomain>(
                    functor, SourceDatumCoord{});
                result &= functor.result;
            }
            const LeftDatum & left;
            const RightDatum & right;
            bool result;
        };

        template<typename LeftDatum, typename RightType, typename OP>
        struct GenericBoolTypeFunctor
        {
            template<typename OuterCoord, typename InnerCoord>
            LLAMA_FN_HOST_ACC_INLINE void operator()(OuterCoord, InnerCoord)
            {
                using Dst = Cat<OuterCoord, InnerCoord>;
                result &= OP{}(
                    left(Dst()),
                    static_cast<std::remove_reference_t<decltype(left(Dst()))>>(
                        right));
            }
            const LeftDatum & left;
            const RightType & right;
            bool result;
        };
    }

    namespace internal
    {
        struct Assign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto)
            operator()(A & a, const B & b) const
            {
                return a = b;
            }
        };

        struct PlusAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto)
            operator()(A & a, const B & b) const
            {
                return a += b;
            }
        };

        struct MinusAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto)
            operator()(A & a, const B & b) const
            {
                return a -= b;
            }
        };

        struct MultiplyAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto)
            operator()(A & a, const B & b) const
            {
                return a *= b;
            }
        };

        struct DivideAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto)
            operator()(A & a, const B & b) const
            {
                return a /= b;
            }
        };

        struct ModuloAssign
        {
            template<typename A, typename B>
            LLAMA_FN_HOST_ACC_INLINE decltype(auto)
            operator()(A & a, const B & b) const
            {
                return a %= b;
            }
        };
    }

    /// Virtual data type returned by \ref View after resolving a user domain
    /// coordinate or partially resolving a \ref DatumCoord. A virtual datum
    /// does not hold data itself (thus named "virtual"), it just binds enough
    /// information (user domain coord and partial datum coord) to retrieve it
    /// from a \ref View later. Virtual datums should not be created by the
    /// user. They are returned from various access functions in \ref View and
    /// VirtualDatum itself.
    template<typename T_View, typename BoundDatumDomain, bool OwnView>
    struct VirtualDatum
    {
        using View = T_View; ///< View this virtual datum points into.

    private:
        using UserDomain = typename View::Mapping::UserDomain;
        using DatumDomain = typename View::Mapping::DatumDomain;

        const UserDomain userDomainPos;
        std::conditional_t<OwnView, View, View &> view;

    public:
        /// Subtree of the datum domain of View starting at
        /// BoundDatumDomain. If BoundDatumDomain is `DatumCoord<>` (default)
        /// AccessibleDatumDomain is the same as `Mapping::DatumDomain`.
        using AccessibleDatumDomain = GetType<DatumDomain, BoundDatumDomain>;

        LLAMA_FN_HOST_ACC_INLINE
        VirtualDatum(
            UserDomain userDomainPos,
            std::conditional_t<OwnView, View &&, View &> view) :
                userDomainPos(userDomainPos),
                view{static_cast<decltype(view)>(view)}
        {}

        VirtualDatum(const VirtualDatum &) = default;
        VirtualDatum(VirtualDatum &&) = default;

        /// Access a datum in the datum domain underneath the current virtual
        /// datum using a \ref DatumCoord. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual datum.
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto access(DatumCoord<Coord...> = {}) const
            -> decltype(auto)
        {
            if constexpr(isDatumStruct<GetType<
                             DatumDomain,
                             Cat<BoundDatumDomain, DatumCoord<Coord...>>>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<
                    const View,
                    Cat<BoundDatumDomain, DatumCoord<Coord...>>>{
                    userDomainPos, this->view};
            }
            else
            {
                using DatumCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, DatumCoord{});
            }
        }

        // FIXME(bgruber): remove redundancy
        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto access(DatumCoord<Coord...> coord = {})
            -> decltype(auto)
        {
            if constexpr(isDatumStruct<GetType<
                             DatumDomain,
                             Cat<BoundDatumDomain, DatumCoord<Coord...>>>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<
                    View,
                    Cat<BoundDatumDomain, DatumCoord<Coord...>>>{
                    userDomainPos, this->view};
            }
            else
            {
                using DatumCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, DatumCoord{});
            }
        }

        /// Access a datum in the datum domain underneath the current virtual
        /// datum using a series of tags. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual datum.
        template<typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto access(Tags...) const -> decltype(auto)
        {
            using DatumCoord = GetCoordFromTagsRelative<
                DatumDomain,
                BoundDatumDomain,
                Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoord{});
        }

        // FIXME(bgruber): remove redundancy
        template<typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto access(Tags... uids) -> decltype(auto)
        {
            using DatumCoord = GetCoordFromTagsRelative<
                DatumDomain,
                BoundDatumDomain,
                Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoord{});
        }

        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access() const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access() -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

        /// Calls \ref access with the passed arguments and returns the result.
        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoordOrUIDs...) const
            -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

        template<typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoordOrUIDs...)
            -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs{}...);
        }

#define __LLAMA_VIRTUALDATUM_OPERATOR(OP, FUNCTOR) \
    template< \
        typename OtherView, \
        typename OtherBoundDatumDomain, \
        bool OtherOwnView> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView> & \
            other) \
        ->VirtualDatum & \
    { \
        internal::GenericFunctor< \
            std::remove_reference_t<decltype(*this)>, \
            VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView>, \
            DatumCoord<>, \
            FUNCTOR> \
            functor{*this, other}; \
        forEach<AccessibleDatumDomain>(functor); \
        return *this; \
    } \
\
    template<typename OtherMapping, typename OtherBlobType> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const llama::View<OtherMapping, OtherBlobType> & other) \
        ->VirtualDatum & \
    { \
        return *this OP other( \
            llama::UserDomain<OtherMapping::UserDomain::rank>{}); \
    } \
\
    template< \
        typename OtherType, \
        typename \
        = std::enable_if_t<!IsView<OtherType> && !is_VirtualDatum<OtherType>>> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP(const OtherType & other) \
        ->VirtualDatum & \
    { \
        internal::GenericTypeFunctor<decltype(*this), OtherType, FUNCTOR> \
            functor{*this, other}; \
        forEach<AccessibleDatumDomain>(functor); \
        return *this; \
    }

        __LLAMA_VIRTUALDATUM_OPERATOR(=, internal::Assign)
        __LLAMA_VIRTUALDATUM_OPERATOR(+=, internal::PlusAssign)
        __LLAMA_VIRTUALDATUM_OPERATOR(-=, internal::MinusAssign)
        __LLAMA_VIRTUALDATUM_OPERATOR(*=, internal::MultiplyAssign)
        __LLAMA_VIRTUALDATUM_OPERATOR(/=, internal::DivideAssign)
        __LLAMA_VIRTUALDATUM_OPERATOR(%=, internal::ModuloAssign)

#undef __LLAMA_VIRTUALDATUM_OPERATOR

        // we need this one to disable the compiler generated copy assignment
        LLAMA_FN_HOST_ACC_INLINE auto operator=(const VirtualDatum & other)
            -> VirtualDatum &
        {
            return this->operator=<>(other);
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator+(const OtherType & other)
        {
            return copyVirtualDatumStack(*this) += other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator-(const OtherType & other)
        {
            return copyVirtualDatumStack(*this) -= other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator*(const OtherType & other)
        {
            return copyVirtualDatumStack(*this) *= other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator/(const OtherType & other)
        {
            return copyVirtualDatumStack(*this) /= other;
        }

        template<typename OtherType>
        LLAMA_FN_HOST_ACC_INLINE auto operator%(const OtherType & other)
        {
            return copyVirtualDatumStack(*this) %= other;
        }

#define __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(OP, FUNCTOR) \
    template< \
        typename OtherView, \
        typename OtherBoundDatumDomain, \
        bool OtherOwnView> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView> & \
            other) const->bool \
    { \
        internal::GenericBoolFunctor< \
            std::remove_reference_t<decltype(*this)>, \
            VirtualDatum<OtherView, OtherBoundDatumDomain, OtherOwnView>, \
            DatumCoord<>, \
            FUNCTOR> \
            functor{*this, other, true}; \
        forEach<AccessibleDatumDomain>(functor); \
        return functor.result; \
    } \
\
    template<typename OtherMapping, typename OtherBlobType> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP( \
        const llama::View<OtherMapping, OtherBlobType> & other) const->bool \
    { \
        return *this OP other( \
            llama::UserDomain<OtherMapping::UserDomain::rank>{}); \
    } \
\
    template<typename OtherType> \
    LLAMA_FN_HOST_ACC_INLINE auto operator OP(const OtherType & other) \
        const->bool \
    { \
        internal::GenericBoolTypeFunctor<decltype(*this), OtherType, FUNCTOR> \
            functor{*this, other, true}; \
        forEach<AccessibleDatumDomain>(functor); \
        return functor.result; \
    }

        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(==, std::equal_to<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(!=, std::not_equal_to<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(<, std::less<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(<=, std::less_equal<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(>, std::greater<>)
        __LLAMA_VIRTUALDATUM_BOOL_OPERATOR(>=, std::greater_equal<>)

#undef __LLAMA_VIRTUALDATUM_BOOL_OPERATOR
    };

    /// Central LLAMA class holding memory for storage and giving access to
    /// values stored there defined by a mapping. A view should be created using
    /// \ref allocView.
    /// \tparam T_Mapping The mapping used by the view to map accesses into
    /// memory.
    /// \tparam BlobType The storage type used by the view holding
    /// memory.
    template<typename T_Mapping, typename BlobType>
    struct View
    {
        using Mapping = T_Mapping;
        using UserDomain = typename Mapping::UserDomain;
        using DatumDomain = typename Mapping::DatumDomain;
        using VirtualDatumType = VirtualDatum<View<Mapping, BlobType>>;
        using VirtualDatumTypeConst
            = VirtualDatum<const View<Mapping, BlobType>>;

        View() = default;

        LLAMA_FN_HOST_ACC_INLINE
        View(
            Mapping mapping,
            Array<BlobType, Mapping::blobCount> storageBlobs) :
                mapping(std::move(mapping)), storageBlobs(storageBlobs)
        {}

        /// Retrieves the \ref VirtualDatum at the given \ref UserDomain
        /// coordinate.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {userDomain, *this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {userDomain, *this};
        }

        /// Retrieves the \ref VirtualDatum at the \ref UserDomain coordinate
        /// constructed from the passed component indices.
        template<typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {UserDomain{indices...}, *this};
        }

        template<typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return {UserDomain{indices...}, *this};
        }

        /// Retrieves the \ref VirtualDatum at the \ref UserDomain coordinate
        /// constructed from the passed component indices.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](UserDomain userDomain) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(userDomain);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](UserDomain userDomain)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(userDomain);
        }

        /// Retrieves the \ref VirtualDatum at the 1D \ref UserDomain coordinate
        /// constructed from the passed index.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index) const
            -> VirtualDatumTypeConst
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](std::size_t index)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(index);
        }

        Mapping mapping;
        Array<BlobType, Mapping::blobCount> storageBlobs;

    private:
        template<typename T_View, typename T_BoundDatumDomain, bool OwnView>
        friend struct VirtualDatum;

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto
        accessor(UserDomain userDomain, DatumCoord<Coords...> = {}) const
            -> const auto &
        {
            const auto [nr, offset]
                = mapping.template getBlobNrAndOffset<Coords...>(userDomain);
            using Type = GetType<DatumDomain, DatumCoord<Coords...>>;
            return reinterpret_cast<const Type &>(storageBlobs[nr][offset]);
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto
        accessor(UserDomain userDomain, DatumCoord<Coords...> coord = {})
            -> auto &
        {
            const auto [nr, offset]
                = mapping.template getBlobNrAndOffset<Coords...>(userDomain);
            using Type = GetType<DatumDomain, DatumCoord<Coords...>>;
            return reinterpret_cast<Type &>(storageBlobs[nr][offset]);
        }
    };

    /// Acts like a \ref View, but shows only a smaller and/or shifted part of
    /// another view it references, the parent view.
    template<typename T_ParentViewType>
    struct VirtualView
    {
        using ParentView = T_ParentViewType; ///< type of the parent view
        using Mapping =
            typename ParentView::Mapping; ///< mapping of the parent view
        using UserDomain =
            typename Mapping::UserDomain; ///< user domain of the parent view
        using VirtualDatumType =
            typename ParentView::VirtualDatumType; ///< VirtualDatum type of the
                                                   ///< parent view

        /// Creates a VirtualView given a parent \ref View, offset and size.
        LLAMA_FN_HOST_ACC_INLINE
        VirtualView(
            ParentView & parentView,
            UserDomain offset,
            UserDomain size) :
                parentView(parentView), offset(offset), size(size)
        {}

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(UserDomain userDomain) const
            -> const auto &
        {
            return parentView.template accessor<Coords...>(userDomain + offset);
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(UserDomain userDomain) -> auto &
        {
            return parentView.template accessor<Coords...>(userDomain + offset);
        }

        /// Same as \ref View::operator()(UserDomain), but shifted by the offset
        /// of this \ref VirtualView.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain) const
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(userDomain + offset);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(UserDomain userDomain)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(userDomain + offset);
        }

        /// Same as corresponding operator in \ref View, but shifted by the
        /// offset of this \ref VirtualView.
        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(UserDomain{indices...} + offset);
        }

        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices)
            -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(UserDomain{indices...} + offset);
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto
        operator()(DatumCoord<Coord...> && dc = {}) const -> const auto &
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(UserDomain{});
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto
        operator()(DatumCoord<Coord...> && dc = {}) -> auto &
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(UserDomain{});
        }

        ParentView & parentView; ///< reference to parent view.
        const UserDomain
            offset; ///< offset this view's \ref UserDomain coordinates are
                    ///< shifted to the parent view.
        const UserDomain size;
    };
}

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
    template <typename Mapping, typename BlobType>
    struct View;

    namespace internal
    {
        template <typename Allocator>
        using AllocatorBlobType = decltype(std::declval<Allocator>()(0));

        template <typename Allocator, typename Mapping, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE static auto makeBlobArray(
            const Allocator& alloc,
            const Mapping& mapping,
            std::integer_sequence<std::size_t, Is...>) -> Array<AllocatorBlobType<Allocator>, Mapping::blobCount>
        {
            return {alloc(mapping.getBlobSize(Is))...};
        }
    } // namespace internal

    /// Creates a view based on the given mapping, e.g. \ref mapping::AoS or
    /// \ref mapping::SoA. For allocating the view's underlying memory, the
    /// specified allocator is used (or the default one, which is \ref
    /// allocator::Vector). This function is the preferred way to create a \ref
    /// View.
    template <typename Mapping, typename Allocator = allocator::Vector<>>
    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator>>
    {
        auto blobs
            = internal::makeBlobArray<Allocator>(alloc, mapping, std::make_index_sequence<Mapping::blobCount> {});
        return {std::move(mapping), std::move(blobs)};
    }

    /// Allocates a \ref View holding a single datum backed by stack memory
    /// (\ref allocator::Stack).
    /// \tparam Dim Dimension of the \ref ArrayDomain of the \ref View.
    template <std::size_t Dim, typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
    {
        using Mapping = llama::mapping::One<ArrayDomain<Dim>, DatumDomain>;
        return allocView(Mapping {}, llama::allocator::Stack<sizeOf<DatumDomain>> {});
    }

    template <typename View>
    inline constexpr auto IsView = false;

    template <typename Mapping, typename BlobType>
    inline constexpr auto IsView<View<Mapping, BlobType>> = true;

    template <typename View, typename BoundDatumDomain = DatumCoord<>, bool OwnView = false>
    struct VirtualDatum;

    template <typename View>
    inline constexpr auto is_VirtualDatum = false;

    template <typename View, typename BoundDatumDomain, bool OwnView>
    inline constexpr auto is_VirtualDatum<VirtualDatum<View, BoundDatumDomain, OwnView>> = true;

    /// Creates a single \ref VirtualDatum owning a view with stack memory.
    template <typename DatumDomain>
    LLAMA_FN_HOST_ACC_INLINE auto allocVirtualDatumStack()
        -> VirtualDatum<decltype(llama::allocViewStack<1, DatumDomain>()), DatumCoord<>, true>
    {
        return {ArrayDomain<1> {}, llama::allocViewStack<1, DatumDomain>()};
    }

    /// Creates a single \ref VirtualDatum owning a view with stack memory and
    /// copies all values from an existing \ref VirtualDatum.
    template <typename VirtualDatum>
    LLAMA_FN_HOST_ACC_INLINE auto copyVirtualDatumStack(const VirtualDatum& vd) -> decltype(auto)
    {
        auto temp = allocVirtualDatumStack<typename VirtualDatum::AccessibleDatumDomain>();
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
            forEach<typename LeftDatum::AccessibleDatumDomain>([&](auto leftCoord) {
                using LeftInnerCoord = decltype(leftCoord);
                forEach<typename RightDatum::AccessibleDatumDomain>([&](auto rightCoord) {
                    using RightInnerCoord = decltype(rightCoord);
                    if constexpr (hasSameTags<
                                      typename LeftDatum::AccessibleDatumDomain,
                                      LeftInnerCoord,
                                      typename RightDatum::AccessibleDatumDomain,
                                      RightInnerCoord>)
                    {
                        Functor {}(left(leftCoord), right(rightCoord));
                    }
                });
            });
            return left;
        }

        template <typename Functor, typename LeftDatum, typename RightMapping, typename RightBlobType>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumArithOperator(
            LeftDatum& left,
            const View<RightMapping, RightBlobType>& right) -> LeftDatum&
        {
            return virtualDatumArithOperator(left, right(ArrayDomain<RightMapping::ArrayDomain::rank> {}));
        }

        template <typename Functor, typename LeftDatum, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumArithOperator(LeftDatum& left, const T& right) -> LeftDatum&
        {
            forEach<typename LeftDatum::AccessibleDatumDomain>(
                [&](auto leftCoord) { Functor {}(left(leftCoord), right); });
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
            forEach<typename LeftDatum::AccessibleDatumDomain>([&](auto leftCoord) {
                using LeftInnerCoord = decltype(leftCoord);
                forEach<typename RightDatum::AccessibleDatumDomain>([&](auto rightCoord) {
                    using RightInnerCoord = decltype(rightCoord);
                    if constexpr (hasSameTags<
                                      typename LeftDatum::AccessibleDatumDomain,
                                      LeftInnerCoord,
                                      typename RightDatum::AccessibleDatumDomain,
                                      RightInnerCoord>)
                    {
                        result &= Functor {}(left(leftCoord), right(rightCoord));
                    }
                });
            });
            return result;
        }

        template <typename Functor, typename LeftDatum, typename RightMapping, typename RightBlobType>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumRelOperator(
            const LeftDatum& left,
            const View<RightMapping, RightBlobType>& right) -> bool
        {
            return virtualDatumRelOperator(left, right(ArrayDomain<RightMapping::ArrayDomain::rank> {}));
        }

        template <typename Functor, typename LeftDatum, typename T>
        LLAMA_FN_HOST_ACC_INLINE auto virtualDatumRelOperator(const LeftDatum& left, const T& right) -> bool
        {
            bool result = true;
            forEach<typename LeftDatum::AccessibleDatumDomain>([&](auto leftCoord) {
                result &= Functor {}(
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

        LLAMA_FN_HOST_ACC_INLINE
        VirtualDatum(ArrayDomain userDomainPos, std::conditional_t<OwnView, View&&, View&> view)
            : userDomainPos(userDomainPos)
            , view {static_cast<decltype(view)>(view)}
        {
        }

        VirtualDatum(const VirtualDatum&) = default;
        VirtualDatum(VirtualDatum&&) = default;

        /// Access a datum in the datum domain underneath the current virtual
        /// datum using a \ref DatumCoord. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual datum.
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto access(DatumCoord<Coord...> = {}) const -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
            if constexpr (isDatumStruct<GetType<DatumDomain, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<const View, AbsolutCoord> {userDomainPos, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, AbsolutCoord {});
            }
        }

        // FIXME(bgruber): remove redundancy
        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto access(DatumCoord<Coord...> coord = {}) -> decltype(auto)
        {
            using AbsolutCoord = Cat<BoundDatumDomain, DatumCoord<Coord...>>;
            if constexpr (isDatumStruct<GetType<DatumDomain, AbsolutCoord>>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatum<View, AbsolutCoord> {userDomainPos, this->view};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return this->view.accessor(userDomainPos, AbsolutCoord {});
            }
        }

        /// Access a datum in the datum domain underneath the current virtual
        /// datum using a series of tags. If the access resolves to a leaf, a
        /// reference to a variable inside the \ref View storage is returned,
        /// otherwise another virtual datum.
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto access(Tags...) const -> decltype(auto)
        {
            using DatumCoord = GetCoordFromTagsRelative<DatumDomain, BoundDatumDomain, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoord {});
        }

        // FIXME(bgruber): remove redundancy
        template <typename... Tags>
        LLAMA_FN_HOST_ACC_INLINE auto access(Tags... uids) -> decltype(auto)
        {
            using DatumCoord = GetCoordFromTagsRelative<DatumDomain, BoundDatumDomain, Tags...>;

            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoord {});
        }

        template <typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access() const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs {}...);
        }

        template <typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto access() -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs {}...);
        }

        /// Calls \ref access with the passed arguments and returns the result.
        template <typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoordOrUIDs...) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs {}...);
        }

        template <typename... DatumCoordOrUIDs>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoordOrUIDs...) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return access(DatumCoordOrUIDs {}...);
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
        LLAMA_FN_HOST_ACC_INLINE auto operator+(const T& other)
        {
            return copyVirtualDatumStack(*this) += other;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator-(const T& other)
        {
            return copyVirtualDatumStack(*this) -= other;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator*(const T& other)
        {
            return copyVirtualDatumStack(*this) *= other;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator/(const T& other)
        {
            return copyVirtualDatumStack(*this) /= other;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator%(const T& other)
        {
            return copyVirtualDatumStack(*this) %= other;
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator==(const T& other) const -> bool
        {
            return internal::virtualDatumRelOperator<std::equal_to<>>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator!=(const T& other) const -> bool
        {
            return internal::virtualDatumRelOperator<std::not_equal_to<>>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator<(const T& other) const -> bool
        {
            return internal::virtualDatumRelOperator<std::less<>>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator<=(const T& other) const -> bool
        {
            return internal::virtualDatumRelOperator<std::less_equal<>>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator>(const T& other) const -> bool
        {
            return internal::virtualDatumRelOperator<std::greater<>>(*this, other);
        }

        template <typename T>
        LLAMA_FN_HOST_ACC_INLINE auto operator>=(const T& other) const -> bool
        {
            return internal::virtualDatumRelOperator<std::greater_equal<>>(*this, other);
        }
    };

    /// Central LLAMA class holding memory for storage and giving access to
    /// values stored there defined by a mapping. A view should be created using
    /// \ref allocView.
    /// \tparam T_Mapping The mapping used by the view to map accesses into
    /// memory.
    /// \tparam BlobType The storage type used by the view holding
    /// memory.
    template <typename T_Mapping, typename BlobType>
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
            , storageBlobs(storageBlobs)
        {
        }

        /// Retrieves the \ref VirtualDatum at the given \ref ArrayDomain
        /// coordinate.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) const -> decltype(auto)
        {
            if constexpr (isDatumStruct<DatumDomain>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatumTypeConst {arrayDomain, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDomain, DatumCoord<> {});
            }
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDomain arrayDomain) -> decltype(auto)
        {
            if constexpr (isDatumStruct<DatumDomain>)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualDatumType {arrayDomain, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDomain, DatumCoord<> {});
            }
        }

        /// Retrieves the \ref VirtualDatum at the \ref ArrayDomain coordinate
        /// constructed from the passed component indices.
        template <typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ArrayDomain {indices...});
        }

        template <typename... Index>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Index... indices) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ArrayDomain {indices...});
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
            return parentView(ArrayDomain {indices...} + offset);
        }

        template <typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> VirtualDatumType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDomain {indices...} + offset);
        }

        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoord<Coord...>&& dc = {}) const -> const auto&
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayDomain {});
        }

        template <std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(DatumCoord<Coord...>&& dc = {}) -> auto&
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayDomain {});
        }

        ParentView& parentView; ///< reference to parent view.
        const ArrayDomain offset; ///< offset this view's \ref ArrayDomain coordinates are
                                 ///< shifted to the parent view.
        const ArrayDomain size;
    };
} // namespace llama

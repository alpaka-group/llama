// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "ArrayDimsIndexRange.hpp"
#include "BlobAllocators.hpp"
#include "Concepts.hpp"
#include "Core.hpp"
#include "macros.hpp"
#include "mapping/One.hpp"

#include <type_traits>

namespace llama
{
#ifdef __cpp_lib_concepts
    template<typename TMapping, Blob BlobType>
#else
    template<typename TMapping, typename BlobType>
#endif
    struct View;

    namespace internal
    {
        template<typename Allocator, typename RecordDim>
        using AllocatorBlobType
            = decltype(std::declval<Allocator>()(std::integral_constant<std::size_t, alignOf<RecordDim>>{}, 0));

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<typename Allocator, typename Mapping, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeBlobArray(
            const Allocator& alloc,
            const Mapping& mapping,
            std::integer_sequence<std::size_t, Is...>)
            -> Array<AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Mapping::blobCount>
        {
            constexpr auto alignment = alignOf<typename Mapping::RecordDim>;
            return {alloc(std::integral_constant<std::size_t, alignment>{}, mapping.blobSize(Is))...};
        }
    } // namespace internal

    /// Creates a view based on the given mapping, e.g. \ref AoS or \ref :SoA. For allocating the view's underlying
    /// memory, the specified allocator callable is used (or the default one, which is \ref bloballoc::Vector). The
    /// allocator callable is called with the size of bytes to allocate for each blob of the mapping. This function is
    /// the preferred way to create a \ref View.
#ifdef __cpp_lib_concepts
    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector>
#else
    template<typename Mapping, typename Allocator = bloballoc::Vector>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>>
    {
        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
        return {std::move(mapping), std::move(blobs)};
    }

    /// Allocates a \ref View holding a single record backed by stack memory (\ref bloballoc::Stack).
    /// \tparam Dim Dimension of the \ref ArrayDims of the \ref View.
    template<std::size_t Dim, typename RecordDim>
    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
    {
        constexpr auto mapping = mapping::MinAlignedOne<ArrayDims<Dim>, RecordDim>{};
        return allocView(mapping, bloballoc::Stack<mapping.blobSize(0)>{});
    }

    template<typename View, typename BoundRecordCoord = RecordCoord<>, bool OwnView = false>
    struct VirtualRecord;

    /// A \ref VirtualRecord that owns and holds a single value.
    template<typename RecordDim>
    using One = VirtualRecord<decltype(allocViewStack<0, RecordDim>()), RecordCoord<>, true>;

    // TODO(bgruber): Higher dimensional iterators might not have good codegen. Multiple nested loops seem to be
    // superior to a single iterator over multiple dimensions. At least compilers are able to produce better code.
    // std::mdspan also discovered similar difficulties and there was a discussion in WG21 in Oulu 2016 to
    // remove/postpone iterators from the design. In std::mdspan's design, the iterator iterated over the co-domain.
    template<typename View>
    struct Iterator
    {
        using ADIterator = ArrayDimsIndexIterator<View::ArrayDims::rank>;

        using iterator_category = std::random_access_iterator_tag;
        using value_type = One<typename View::RecordDim>;
        using difference_type = typename ADIterator::difference_type;
        using pointer = internal::IndirectValue<VirtualRecord<View>>;
        using reference = VirtualRecord<View>;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++() -> Iterator&
        {
            ++adIndex;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++(int) -> Iterator
        {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--() -> Iterator&
        {
            --adIndex;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator--(int) -> Iterator
        {
            auto tmp{*this};
            --*this;
            return tmp;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator*() const -> reference
        {
            return (*view)(*adIndex);
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator->() const -> pointer
        {
            return {**this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator[](difference_type i) const -> reference
        {
            return *(*this + i);
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator+=(difference_type n) -> Iterator&
        {
            adIndex += n;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(Iterator it, difference_type n) -> Iterator
        {
            it += n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator+(difference_type n, Iterator it) -> Iterator
        {
            return it + n;
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator-=(difference_type n) -> Iterator&
        {
            adIndex -= n;
            return *this;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(Iterator it, difference_type n) -> Iterator
        {
            it -= n;
            return it;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator-(const Iterator& a, const Iterator& b) -> difference_type
        {
            return static_cast<std::ptrdiff_t>(a.adIndex - b.adIndex);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator==(const Iterator& a, const Iterator& b) -> bool
        {
            return a.adIndex == b.adIndex;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator!=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a == b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<(const Iterator& a, const Iterator& b) -> bool
        {
            return a.adIndex < b.adIndex;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>(const Iterator& a, const Iterator& b) -> bool
        {
            return b < a;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a > b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator>=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a < b);
        }

        ADIterator adIndex;
        View* view;
    };

    namespace internal
    {
        template<typename Mapping, typename RecordCoord, typename = void>
        struct IsComputed : std::false_type
        {
        };

        template<typename Mapping, typename RecordCoord>
        struct IsComputed<Mapping, RecordCoord, std::void_t<decltype(Mapping::isComputed(RecordCoord{}))>>
            : std::bool_constant<Mapping::isComputed(RecordCoord{})>
        {
        };
    } // namespace internal

    /// Returns true if the field accessed via the given mapping and record coordinate is a computed value.
    template<typename Mapping, typename RecordCoord>
    inline constexpr bool isComputed = internal::IsComputed<Mapping, RecordCoord>::value;

    /// Central LLAMA class holding memory for storage and giving access to values stored there defined by a mapping. A
    /// view should be created using \ref allocView.
    /// \tparam TMapping The mapping used by the view to map accesses into memory.
    /// \tparam BlobType The storage type used by the view holding memory.
#ifdef __cpp_lib_concepts
    template<typename TMapping, Blob BlobType>
#else
    template<typename TMapping, typename BlobType>
#endif
    struct View
        : private TMapping
#if CAN_USE_RANGES
        , std::ranges::view_base
#endif
    {
        using Mapping = TMapping;
        using ArrayDims = typename Mapping::ArrayDims;
        using RecordDim = typename Mapping::RecordDim;
        using VirtualRecordType = VirtualRecord<View>;
        using VirtualRecordTypeConst = VirtualRecord<const View>;
        using iterator = Iterator<View>;
        using const_iterator = Iterator<const View>;

        static_assert(
            !std::is_reference_v<ArrayDims>,
            "Mapping::ArrayDims must not be a reference. Are you using decltype(...) as mapping template argument?");

        View() = default;

        LLAMA_FN_HOST_ACC_INLINE
        View(Mapping mapping, Array<BlobType, Mapping::blobCount> storageBlobs)
            : Mapping(std::move(mapping))
            , storageBlobs(std::move(storageBlobs))
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() -> Mapping&
        {
            return static_cast<Mapping&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() const -> const Mapping&
        {
            return static_cast<const Mapping&>(*this);
        }

        /// Retrieves the \ref VirtualRecord at the given \ref ArrayDims coordinate.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) const -> decltype(auto)
        {
            if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecordTypeConst{arrayDims, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDims, RecordCoord<>{});
            }
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) -> decltype(auto)
        {
            if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecordType{arrayDims, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(arrayDims, RecordCoord<>{});
            }
        }

        /// Retrieves the \ref VirtualRecord at the \ref ArrayDims coordinate constructed from the passed component
        /// indices.
        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayDims::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, std::size_t>...>,
                "Indices must be convertible to std::size_t");
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (ArrayDims{static_cast<typename ArrayDims::value_type>(indices)...});
        }

        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayDims::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, std::size_t>...>,
                "Indices must be convertible to std::size_t");
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (ArrayDims{static_cast<typename ArrayDims::value_type>(indices)...});
        }

        /// Retrieves the \ref VirtualRecord at the \ref ArrayDims coordinate constructed from the passed component
        /// indices.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDims arrayDims) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (arrayDims);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayDims arrayDims) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this) (arrayDims);
        }

        /// Retrieves the \ref VirtualRecord at the 1D \ref ArrayDims coordinate constructed from the passed index.
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

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() -> iterator
        {
            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping().arrayDims()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() const -> const_iterator
        {
            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping().arrayDims()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() -> iterator
        {
            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping().arrayDims()}.end(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() const -> const_iterator
        {
            return {ArrayDimsIndexRange<ArrayDims::rank>{mapping().arrayDims()}.end(), this};
        }

        Array<BlobType, Mapping::blobCount> storageBlobs;

    private:
        template<typename TView, typename TBoundRecordCoord, bool OwnView>
        friend struct VirtualRecord;

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims, RecordCoord<Coords...> rc = {}) const
            -> decltype(auto)
        {
            if constexpr(isComputed<Mapping, RecordCoord<Coords...>>)
                return mapping().compute(arrayDims, rc, storageBlobs);
            else
            {
                const auto [nr, offset] = mapping().blobNrAndOffset(arrayDims, rc);
                using Type = GetType<RecordDim, RecordCoord<Coords...>>;
                return reinterpret_cast<const Type&>(storageBlobs[nr][offset]);
            }
        }

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims, RecordCoord<Coords...> rc = {}) -> decltype(auto)
        {
            if constexpr(isComputed<Mapping, RecordCoord<Coords...>>)
                return mapping().compute(arrayDims, rc, storageBlobs);
            else
            {
                const auto [nr, offset] = mapping().blobNrAndOffset(arrayDims, rc);
                using Type = GetType<RecordDim, RecordCoord<Coords...>>;
                return reinterpret_cast<Type&>(storageBlobs[nr][offset]);
            }
        }
    };

    template<typename View>
    inline constexpr auto IsView = false;

    template<typename Mapping, typename BlobType>
    inline constexpr auto IsView<View<Mapping, BlobType>> = true;

    /// Acts like a \ref View, but shows only a smaller and/or shifted part of another view it references, the parent
    /// view.
    template<typename TParentViewType>
    struct VirtualView
    {
        using ParentView = TParentViewType; ///< type of the parent view
        using Mapping = typename ParentView::Mapping; ///< mapping of the parent view
        using ArrayDims = typename Mapping::ArrayDims; ///< array dimensions of the parent view
        using VirtualRecordType = typename ParentView::VirtualRecordType; ///< VirtualRecord type of the
                                                                          ///< parent view

        /// Creates a VirtualView given a parent \ref View, offset and size.
        LLAMA_FN_HOST_ACC_INLINE
        VirtualView(ParentView& parentView, ArrayDims offset) : parentView(parentView), offset(offset)
        {
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims) const -> const auto&
        {
            return parentView.template accessor<Coords...>(ArrayDims{arrayDims + offset});
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayDims arrayDims) -> auto&
        {
            return parentView.template accessor<Coords...>(ArrayDims{arrayDims + offset});
        }

        /// Same as \ref View::operator()(ArrayDims), but shifted by the offset of this \ref VirtualView.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) const -> VirtualRecordType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDims{arrayDims + offset});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayDims arrayDims) -> VirtualRecordType
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDims{arrayDims + offset});
        }

        /// Same as corresponding operator in \ref View, but shifted by the offset of this \ref VirtualView.
        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> VirtualRecordType
        {
            static_assert(
                sizeof...(Indices) == ArrayDims::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, std::size_t>...>,
                "Indices must be convertible to std::size_t");
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDims{ArrayDims{static_cast<typename ArrayDims::value_type>(indices)...} + offset});
        }

        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> VirtualRecordType
        {
            static_assert(
                sizeof...(Indices) == ArrayDims::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, std::size_t>...>,
                "Indices must be convertible to std::size_t");
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayDims{ArrayDims{static_cast<typename ArrayDims::value_type>(indices)...} + offset});
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayDims{});
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayDims{});
        }

        ParentView& parentView; ///< reference to parent view.
        const ArrayDims offset; ///< offset this view's \ref ArrayDims coordinates are shifted to the parent view.
    };
} // namespace llama

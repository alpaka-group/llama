// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "Array.hpp"
#include "ArrayIndexRange.hpp"
#include "BlobAllocators.hpp"
#include "Concepts.hpp"
#include "Core.hpp"
#include "HasRanges.hpp"
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
            [[maybe_unused]] constexpr auto alignment
                = alignOf<typename Mapping::RecordDim>; // g++-12 warns that alignment is unsed
            return {alloc(std::integral_constant<std::size_t, alignment>{}, mapping.blobSize(Is))...};
        }
    } // namespace internal

    /// Same as \ref allocView but does not run field constructors.
#ifdef __cpp_lib_concepts
    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector>
#else
    template<typename Mapping, typename Allocator = bloballoc::Vector>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocViewUninitialized(Mapping mapping = {}, const Allocator& alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>>
    {
        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
        return {std::move(mapping), std::move(blobs)};
    }

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

    /// Runs the constructor of all fields reachable through the given view. Computed fields are not constructed.
    template<typename Mapping, typename BlobType>
    LLAMA_FN_HOST_ACC_INLINE void constructFields(View<Mapping, BlobType>& view)
    {
        using View = View<Mapping, BlobType>;
        using RecordDim = typename View::RecordDim;
        forEachADCoord(
            view.mapping().extents(),
            [&]([[maybe_unused]] typename View::ArrayIndex ai)
            {
                if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
                    forEachLeafCoord<RecordDim>(
                        [&](auto rc)
                        {
                            // TODO(bgruber): we could initialize computed fields if we can write to those. We could
                            // test if the returned value can be cast to a T& and then attempt to write.
                            if constexpr(!isComputed<Mapping, decltype(rc)>)
                                new(&view(ai)(rc)) GetType<RecordDim, decltype(rc)>;
                        });
                else if constexpr(!isComputed<Mapping, RecordCoord<>>)
                    new(&view(ai)) RecordDim;
            });
    }

    /// Creates a view based on the given mapping, e.g. \ref AoS or \ref :SoA. For allocating the view's underlying
    /// memory, the specified allocator callable is used (or the default one, which is \ref bloballoc::Vector). The
    /// allocator callable is called with the alignment and size of bytes to allocate for each blob of the mapping.
    /// The constructors are run for all fields by calling \ref constructFields. This function is the preferred way to
    /// create a \ref View. See also \ref allocViewUninitialized.
#ifdef __cpp_lib_concepts
    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector>
#else
    template<typename Mapping, typename Allocator = bloballoc::Vector>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>>
    {
        auto view = allocViewUninitialized(std::move(mapping), alloc);
        constructFields(view);
        return view;
    }

    /// Allocates a \ref View holding a single record backed by stack memory (\ref bloballoc::Stack).
    /// \tparam Dim Dimension of the \ref ArrayExtents of the \ref View.
    template<std::size_t Dim, typename RecordDim>
    LLAMA_FN_HOST_ACC_INLINE auto allocViewStack() -> decltype(auto)
    {
        constexpr auto mapping = mapping::MinAlignedOne<ArrayExtentsNCube<int, Dim, 1>, RecordDim>{};
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
        using ArrayIndexIterator = llama::ArrayIndexIterator<typename View::ArrayExtents>;

        using iterator_category = std::random_access_iterator_tag;
        using value_type = One<typename View::RecordDim>;
        using difference_type = typename ArrayIndexIterator::difference_type;
        using pointer = internal::IndirectValue<VirtualRecord<View>>;
        using reference = VirtualRecord<View>;

        constexpr Iterator() = default;

        LLAMA_FN_HOST_ACC_INLINE constexpr Iterator(ArrayIndexIterator arrayIndex, View* view)
            : arrayIndex(arrayIndex)
            , view(view)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto operator++() -> Iterator&
        {
            ++arrayIndex;
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
            --arrayIndex;
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
            return (*view)(*arrayIndex);
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
            arrayIndex += n;
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
            arrayIndex -= n;
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
            assert(a.view == b.view);
            return static_cast<std::ptrdiff_t>(a.arrayIndex - b.arrayIndex);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator==(const Iterator& a, const Iterator& b) -> bool
        {
            assert(a.view == b.view);
            return a.arrayIndex == b.arrayIndex;
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator!=(const Iterator& a, const Iterator& b) -> bool
        {
            return !(a == b);
        }

        LLAMA_FN_HOST_ACC_INLINE
        friend constexpr auto operator<(const Iterator& a, const Iterator& b) -> bool
        {
            assert(a.view == b.view);
            return a.arrayIndex < b.arrayIndex;
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

        ArrayIndexIterator arrayIndex;
        View* view;
    };

    /// Using a mapping, maps the given array index and record coordinate to a memory reference onto the given blobs.
    /// \return Either an l-value reference if the record coord maps to a physical field or a proxy reference if mapped
    /// to a computed field.
    template<typename Mapping, std::size_t... Coords, typename Blobs>
    LLAMA_FN_HOST_ACC_INLINE auto mapToMemory(
        Mapping& mapping,
        typename Mapping::ArrayIndex ai,
        RecordCoord<Coords...> rc,
        Blobs& blobs) -> decltype(auto)
    {
        if constexpr(llama::isComputed<Mapping, RecordCoord<Coords...>>)
            return mapping.compute(ai, rc, blobs);
        else
        {
            const auto [nr, offset] = mapping.blobNrAndOffset(ai, rc);
            using Type = GetType<typename Mapping::RecordDim, RecordCoord<Coords...>>;


#ifdef __NVCC__
            // suppress: calling a __host__ function from a __host__ __device__ function is not allowed
            // suppress: calling a __host__ function("...") from a __host__ __device__ function("...") is not allowed
#    pragma push
#    ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#        pragma nv_diag_suppress 20011
#        pragma nv_diag_suppress 20014
#    else
#        pragma diag_suppress 20011
#        pragma diag_suppress 20014
#    endif
#endif
            return reinterpret_cast<CopyConst<std::remove_reference_t<decltype(blobs[nr][offset])>, Type>&>(
                blobs[nr][offset]);
#ifdef __NVCC__
#    pragma pop
#endif
        }
    }

    /// Central LLAMA class holding memory for storage and giving access to values stored there defined by a mapping. A
    /// view should be created using \ref allocView.
    /// \tparam TMapping The mapping used by the view to map accesses into memory.
    /// \tparam BlobType The storage type used by the view holding memory.
#ifdef __cpp_lib_concepts
    template<typename TMapping, Blob BlobType>
#else
    template<typename TMapping, typename BlobType>
#endif
    struct LLAMA_DECLSPEC_EMPTY_BASES View
        : private TMapping
#if CAN_USE_RANGES
        , std::ranges::view_base
#endif
    {
        static_assert(!std::is_const_v<TMapping>);
        using Mapping = TMapping;
        using ArrayExtents = typename Mapping::ArrayExtents;
        using ArrayIndex = typename Mapping::ArrayIndex;
        using RecordDim = typename Mapping::RecordDim;
        using iterator = Iterator<View>;
        using const_iterator = Iterator<const View>;
        using size_type = typename ArrayExtents::value_type;

        static_assert(
            std::is_same_v<Mapping, std::decay_t<Mapping>>,
            "Mapping must not be const qualified or a reference. Are you using decltype(...) as View template "
            "argument?");
        static_assert(
            std::is_same_v<ArrayExtents, std::decay_t<ArrayExtents>>,
            "Mapping::ArrayExtents must not be const qualified or a reference. Are you using decltype(...) as mapping "
            "template argument?");

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

#if !(defined(_MSC_VER) && defined(__NVCC__))
        template<typename V>
        auto operator()(llama::ArrayIndex<V, ArrayIndex::rank>) const
        {
            static_assert(!sizeof(V), "Passed ArrayIndex with SizeType different than Mapping::ArrayExtent");
        }
#endif

        /// Retrieves the \ref VirtualRecord at the given \ref ArrayIndex index.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) const -> decltype(auto)
        {
            if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<const View>{ai, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(ai, RecordCoord<>{});
            }
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
        {
            if constexpr(isRecord<RecordDim> || internal::IsBoundedArray<RecordDim>::value)
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return VirtualRecord<View>{ai, *this};
            }
            else
            {
                LLAMA_FORCE_INLINE_RECURSIVE
                return accessor(ai, RecordCoord<>{});
            }
        }

        /// Retrieves the \ref VirtualRecord at the \ref ArrayIndex index constructed from the passed component
        /// indices.
        template<
            typename... Indices,
            std::enable_if_t<std::conjunction_v<std::is_convertible<Indices, size_type>...>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...});
        }

        template<
            typename... Indices,
            std::enable_if_t<std::conjunction_v<std::is_convertible<Indices, size_type>...>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...});
        }

        /// Retrieves the \ref VirtualRecord at the \ref ArrayIndex index constructed from the passed component
        /// indices.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ai);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(ai);
        }

#if !(defined(_MSC_VER) && defined(__NVCC__))
        template<typename V>
        auto operator[](llama::ArrayIndex<V, ArrayIndex::rank>) const
        {
            static_assert(!sizeof(V), "Passed ArrayIndex with SizeType different than Mapping::ArrayExtent");
        }
#endif

        /// Retrieves the \ref VirtualRecord at the 1D \ref ArrayIndex index constructed from the passed index.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type index) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type index) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() -> iterator
        {
            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() const -> const_iterator
        {
            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() -> iterator
        {
            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.end(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() const -> const_iterator
        {
            return {ArrayIndexRange<ArrayExtents>{mapping().extents()}.end(), this};
        }

        Array<BlobType, Mapping::blobCount> storageBlobs;

    private:
        template<typename TView, typename TBoundRecordCoord, bool OwnView>
        friend struct VirtualRecord;

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayIndex ai, RecordCoord<Coords...> rc = {}) const -> decltype(auto)
        {
            return mapToMemory(mapping(), ai, rc, storageBlobs);
        }

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayIndex ai, RecordCoord<Coords...> rc = {}) -> decltype(auto)
        {
            return mapToMemory(mapping(), ai, rc, storageBlobs);
        }
    };

    template<typename View>
    inline constexpr auto IsView = false;

    template<typename Mapping, typename BlobType>
    inline constexpr auto IsView<View<Mapping, BlobType>> = true;

    /// Like a \ref View, but array indices are shifted.
    /// @tparam TStoredParentView Type of the underlying view. May be cv qualified and/or a reference type.
    template<typename TStoredParentView>
    struct VirtualView
    {
        using StoredParentView = TStoredParentView;
        using ParentView = std::remove_const_t<std::remove_reference_t<StoredParentView>>; ///< type of the parent view
        using Mapping = typename ParentView::Mapping; ///< mapping of the parent view
        using ArrayExtents = typename Mapping::ArrayExtents; ///< array extents of the parent view
        using ArrayIndex = typename Mapping::ArrayIndex; ///< array index of the parent view

        using size_type = typename ArrayExtents::value_type;

        /// Creates a VirtualView given a parent \ref View and offset.
        template<typename StoredParentViewFwd>
        LLAMA_FN_HOST_ACC_INLINE VirtualView(StoredParentViewFwd&& parentView, ArrayIndex offset)
            : parentView(std::forward<StoredParentViewFwd>(parentView))
            , offset(offset)
        {
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayIndex ai) const -> const auto&
        {
            return parentView.template accessor<Coords...>(ArrayIndex{ai + offset});
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto accessor(ArrayIndex ai) -> auto&
        {
            return parentView.template accessor<Coords...>(ArrayIndex{ai + offset});
        }

        /// Same as \ref View::operator()(ArrayIndex), but shifted by the offset of this \ref VirtualView.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayIndex{ai + offset});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(ArrayIndex{ai + offset});
        }

        /// Same as corresponding operator in \ref View, but shifted by the offset of this \ref VirtualView.
        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, size_type>...>,
                "Indices must be convertible to ArrayExtents::size_type");
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(
                ArrayIndex{ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...} + offset});
        }

        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, size_type>...>,
                "Indices must be convertible to ArrayExtents::size_type");
            LLAMA_FORCE_INLINE_RECURSIVE
            return parentView(
                ArrayIndex{ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...} + offset});
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) const -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayIndex{});
        }

        template<std::size_t... Coord>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coord...> = {}) -> decltype(auto)
        {
            LLAMA_FORCE_INLINE_RECURSIVE
            return accessor<Coord...>(ArrayIndex{});
        }

        StoredParentView parentView;
        const ArrayIndex offset; ///< offset by which this view's \ref ArrayIndex indices are shifted when passed to
                                 ///< the parent view.
    };

    /// VirtualView vview(view); will store a reference to view.
    /// VirtualView vview(std::move(view)); will store the view.
    template<typename TStoredParentView>
    VirtualView(TStoredParentView&&, typename std::remove_reference_t<TStoredParentView>::Mapping::ArrayIndex)
        -> VirtualView<TStoredParentView>;
} // namespace llama

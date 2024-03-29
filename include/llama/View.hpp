// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "Accessors.hpp"
#include "Array.hpp"
#include "ArrayIndexRange.hpp"
#include "BlobAllocators.hpp"
#include "Concepts.hpp"
#include "Core.hpp"
#include "macros.hpp"
#include "mapping/One.hpp"

#include <type_traits>

namespace llama
{
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename TMapping, Blob BlobType, typename TAccessor>
#else
    template<typename TMapping, typename BlobType, typename TAccessor>
#endif
    struct View;

    namespace internal
    {
        template<typename Allocator, typename RecordDim>
        using AllocatorBlobType
            = decltype(std::declval<Allocator>()(std::integral_constant<std::size_t, alignOf<RecordDim>>{}, 0));

        template<typename Allocator, typename Mapping, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeBlobArray(
            const Allocator& alloc,
            const Mapping& mapping,
            std::integer_sequence<std::size_t, Is...>)
            -> Array<AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Mapping::blobCount>
        {
            [[maybe_unused]] constexpr auto alignment
                = alignOf<typename Mapping::RecordDim>; // g++-12 warns that alignment is unused
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return {alloc(std::integral_constant<std::size_t, alignment>{}, mapping.blobSize(Is))...};
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        } // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
    } // namespace internal

    /// Same as \ref allocView but does not run field constructors.
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#else
    template<typename Mapping, typename Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocViewUninitialized(
        Mapping mapping = {},
        const Allocator& alloc = {},
        Accessor accessor = {})
    {
        auto blobs = internal::makeBlobArray(alloc, mapping, std::make_index_sequence<Mapping::blobCount>{});
        return View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Accessor>{
            std::move(mapping),
            std::move(blobs),
            std::move(accessor)};
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
    LLAMA_EXPORT
    template<typename Mapping, typename RecordCoord>
    inline constexpr bool isComputed = internal::IsComputed<Mapping, RecordCoord>::value;

    /// Returns true if any field accessed via the given mapping is a computed value.
    // TODO(bgruber): harmonize this with LLAMA's concepts from Concepts.hpp
    LLAMA_EXPORT
    template<typename Mapping>
    inline constexpr bool hasAnyComputedField = mp_any_of<
        LeafRecordCoords<typename Mapping::RecordDim>,
        mp_bind_front<internal::IsComputed, Mapping>::template fn>::value;

    LLAMA_EXPORT
    template<typename Mapping, typename BlobType, typename Accessor, std::size_t... RCs>
    LLAMA_FN_HOST_ACC_INLINE void constructField(
        View<Mapping, BlobType, Accessor>& view,
        typename Mapping::ArrayExtents::Index ai,
        RecordCoord<RCs...> rc)
    {
        auto init = [](auto&& ref) LLAMA_LAMBDA_INLINE
        {
            using FieldType = GetType<typename Mapping::RecordDim, RecordCoord<RCs...>>;
            using RefType = decltype(ref);
            if constexpr(isProxyReference<std::remove_reference_t<RefType>>)
            {
                ref = FieldType{};
            }
            else if constexpr(
                std::is_lvalue_reference_v<RefType> && !std::is_const_v<std::remove_reference_t<RefType>>)
            {
                new(&ref) FieldType{};
            }
        };

        // this handles physical and computed mappings
        if constexpr(sizeof...(RCs) == 0)
        {
            init(view(ai));
        }
        else
        {
            init(view(ai)(rc));
        }
    }

    /// Value-initializes all fields reachable through the given view. That is, constructors are run and fundamental
    /// types are zero-initialized. Computed fields are constructed if they return l-value references and assigned a
    /// default constructed value if they return a proxy reference.
    LLAMA_EXPORT
    template<typename Mapping, typename BlobType, typename Accessor>
    LLAMA_FN_HOST_ACC_INLINE void constructFields(View<Mapping, BlobType, Accessor>& view)
    {
        using View = View<Mapping, BlobType, Accessor>;
        using RecordDim = typename View::RecordDim;
        forEachArrayIndex(
            view.extents(),
            [&](typename View::ArrayIndex ai) LLAMA_LAMBDA_INLINE
            { forEachLeafCoord<RecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE { constructField(view, ai, rc); }); });
    }

    /// Creates a view based on the given mapping, e.g. \ref mapping::AoS or \ref mapping::SoA. For allocating the
    /// view's underlying memory, the specified allocator callable is used (or the default one, which is
    /// \ref bloballoc::Vector). The allocator callable is called with the alignment and size of bytes to allocate for
    /// each blob of the mapping. Value-initialization is performed for all fields by calling \ref constructFields.
    /// This function is the preferred way to create a \ref View. See also \ref allocViewUninitialized.
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename Mapping, BlobAllocator Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#else
    template<typename Mapping, typename Allocator = bloballoc::Vector, typename Accessor = accessor::Default>
#endif
    LLAMA_FN_HOST_ACC_INLINE auto allocView(Mapping mapping = {}, const Allocator& alloc = {}, Accessor accessor = {})
        -> View<Mapping, internal::AllocatorBlobType<Allocator, typename Mapping::RecordDim>, Accessor>
    {
        auto view = allocViewUninitialized(std::move(mapping), alloc, std::move(accessor));
        constructFields(view);
        return view;
    }

    /// Same as \ref allocScalarView but does not run field constructors.
    LLAMA_EXPORT
    template<std::size_t Dim, typename RecordDim>
    LLAMA_FN_HOST_ACC_INLINE auto allocScalarViewUninitialized() -> decltype(auto)
    {
        constexpr auto mapping = mapping::MinAlignedOne<ArrayExtentsNCube<int, Dim, 1>, RecordDim>{};
        return allocViewUninitialized(mapping, bloballoc::Array<mapping.blobSize(0)>{});
    }

    /// Allocates a \ref View holding a single record backed by a byte array (\ref bloballoc::Array).
    /// \tparam Dim Dimension of the \ref ArrayExtents of the \ref View.
    LLAMA_EXPORT
    template<std::size_t Dim, typename RecordDim>
    LLAMA_FN_HOST_ACC_INLINE auto allocScalarView() -> decltype(auto)
    {
        auto view = allocScalarViewUninitialized<Dim, RecordDim>();
        constructFields(view);
        return view;
    }

    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord = RecordCoord<>, bool OwnView = false>
    struct RecordRef;

    /// A \ref RecordRef that owns and holds a single value.
    LLAMA_EXPORT
    template<typename RecordDim>
    using One = RecordRef<decltype(allocScalarView<0, RecordDim>()), RecordCoord<>, true>;

    /// Is true, if T is an instance of \ref One.
    LLAMA_EXPORT
    template<typename T>
    inline constexpr bool isOne = false;

    LLAMA_EXPORT
    template<typename View, typename BoundRecordCoord>
    inline constexpr bool isOne<RecordRef<View, BoundRecordCoord, true>> = true;

    // TODO(bgruber): Higher dimensional iterators might not have good codegen. Multiple nested loops seem to be
    // superior to a single iterator over multiple dimensions. At least compilers are able to produce better code.
    // std::mdspan also discovered similar difficulties and there was a discussion in WG21 in Oulu 2016 to
    // remove/postpone iterators from the design. In std::mdspan's design, the iterator iterated over the co-domain.
    LLAMA_EXPORT
    template<typename View>
    struct Iterator
    {
        using ArrayIndexIterator = llama::ArrayIndexIterator<typename View::ArrayExtents>;

        using iterator_category = std::random_access_iterator_tag;
        using value_type = One<typename View::RecordDim>;
        using difference_type = typename ArrayIndexIterator::difference_type;
        using pointer = internal::IndirectValue<RecordRef<View>>;
        using reference = RecordRef<View>;

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
    LLAMA_EXPORT
    template<typename Mapping, typename RecordCoord, typename Blobs>
    LLAMA_FN_HOST_ACC_INLINE auto mapToMemory(
        Mapping& mapping,
        typename Mapping::ArrayExtents::Index ai,
        RecordCoord rc,
        Blobs& blobs) -> decltype(auto)
    {
        if constexpr(llama::isComputed<Mapping, RecordCoord>)
            return mapping.compute(ai, rc, blobs);
        else
        {
            const auto [nr, offset] = mapping.blobNrAndOffset(ai, rc);
            using Type = GetType<typename Mapping::RecordDim, RecordCoord>;
            LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
            return reinterpret_cast<CopyConst<std::remove_reference_t<decltype(blobs[nr][offset])>, Type>&>(
                blobs[nr][offset]);
            LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
        }
    }

    /// Central LLAMA class holding memory for storage and giving access to values stored there defined by a
    /// mapping. A view should be created using \ref allocView. \tparam TMapping The mapping used by the view to
    /// map accesses into memory. \tparam TBlobType The storage type used by the view holding memory. \tparam
    /// TAccessor The accessor to use when an access is made through this view.
    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename TMapping, Blob TBlobType, typename TAccessor = accessor::Default>
#else
    template<typename TMapping, typename TBlobType, typename TAccessor = accessor::Default>
#endif
    struct LLAMA_DECLSPEC_EMPTY_BASES View
        : private TMapping
        , private TAccessor
#if CAN_USE_RANGES
        , std::ranges::view_base
#endif
    {
        static_assert(!std::is_const_v<TMapping>);
        using Mapping = TMapping;
        using BlobType = TBlobType;
        using ArrayExtents = typename Mapping::ArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = typename Mapping::RecordDim;
        using Accessor = TAccessor;
        using iterator = Iterator<View>;
        using const_iterator = Iterator<const View>;

    private:
        using size_type = typename ArrayExtents::value_type;

    public:
        static_assert(
            std::is_same_v<Mapping, std::decay_t<Mapping>>,
            "Mapping must not be const qualified or a reference. Are you using decltype(...) as View template "
            "argument?");
        static_assert(
            std::is_same_v<ArrayExtents, std::decay_t<ArrayExtents>>,
            "Mapping::ArrayExtents must not be const qualified or a reference. Are you using decltype(...) as "
            "mapping "
            "template argument?");

        /// Performs default initialization of the blob array.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,hicpp-member-init)
        View() = default;

        /// Creates a LLAMA View manually. Prefer the allocations functions \ref allocView and \ref
        /// allocViewUninitialized if possible.
        /// \param mapping The mapping used by the view to map accesses into memory.
        /// \param blobs An array of blobs providing storage space for the mapped data.
        /// \param accessor The accessor to use when an access is made through this view.
        LLAMA_FN_HOST_ACC_INLINE
        explicit View(Mapping mapping, Array<BlobType, Mapping::blobCount> blobs = {}, Accessor accessor = {})
            : Mapping(std::move(mapping))
            , Accessor(std::move(accessor))
            , m_blobs(std::move(blobs))
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

        LLAMA_FN_HOST_ACC_INLINE auto accessor() -> Accessor&
        {
            return static_cast<Accessor&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE auto accessor() const -> const Accessor&
        {
            return static_cast<const Accessor&>(*this);
        }

        LLAMA_FN_HOST_ACC_INLINE auto extents() const -> ArrayExtents
        {
            return mapping().extents();
        }

#if !(defined(_MSC_VER) && defined(__NVCC__))
        template<typename V>
        auto operator()(llama::ArrayIndex<V, ArrayIndex::rank>) const
        {
            static_assert(!sizeof(V), "Passed ArrayIndex with SizeType different than Mapping::ArrayExtent");
        }
#endif

        /// Retrieves the \ref RecordRef at the given \ref ArrayIndex index.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) const -> decltype(auto)
        {
            if constexpr(isRecordDim<RecordDim>)
                return RecordRef<const View>{ai, *this};
            else
                return access(ai, RecordCoord<>{});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
        {
            if constexpr(isRecordDim<RecordDim>)
                return RecordRef<View>{ai, *this};
            else
                return access(ai, RecordCoord<>{});
        }

        /// Retrieves the \ref RecordRef at the \ref ArrayIndex index constructed from the passed component
        /// indices.
        template<
            typename... Indices,
            std::enable_if_t<std::conjunction_v<std::is_convertible<Indices, size_type>...>, int> = 0>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
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
            return (*this)(ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...});
        }

        /// Retrieves the \ref RecordRef at the \ref ArrayIndex index constructed from the passed component
        /// indices.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) const -> decltype(auto)
        {
            return (*this)(ai);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](ArrayIndex ai) -> decltype(auto)
        {
            return (*this)(ai);
        }

#if !(defined(_MSC_VER) && defined(__NVCC__))
        template<typename V>
        auto operator[](llama::ArrayIndex<V, ArrayIndex::rank>) const
        {
            static_assert(!sizeof(V), "Passed ArrayIndex with SizeType different than Mapping::ArrayExtent");
        }
#endif

        /// Retrieves the \ref RecordRef at the 1D \ref ArrayIndex index constructed from the passed index.
        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type index) const -> decltype(auto)
        {
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator[](size_type index) -> decltype(auto)
        {
            return (*this)(index);
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() -> iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto begin() const -> const_iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.begin(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() -> iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.end(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE
        auto end() const -> const_iterator
        {
            return {ArrayIndexRange<ArrayExtents>{extents()}.end(), this};
        }

        LLAMA_FN_HOST_ACC_INLINE auto blobs() -> Array<BlobType, Mapping::blobCount>&
        {
            return m_blobs;
        }

        LLAMA_FN_HOST_ACC_INLINE auto blobs() const -> const Array<BlobType, Mapping::blobCount>&
        {
            return m_blobs;
        }

    private:
        template<typename TView, typename TBoundRecordCoord, bool OwnView>
        friend struct RecordRef;

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto access(ArrayIndex ai, RecordCoord<Coords...> rc = {}) const -> decltype(auto)
        {
            return accessor()(mapToMemory(mapping(), ai, rc, m_blobs));
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto access(ArrayIndex ai, RecordCoord<Coords...> rc = {}) -> decltype(auto)
        {
            return accessor()(mapToMemory(mapping(), ai, rc, m_blobs));
        }

        Array<BlobType, Mapping::blobCount> m_blobs;
    };

    LLAMA_EXPORT
#ifdef __cpp_lib_concepts
    template<typename View>
    inline constexpr auto isView = AnyView<View>;
#else
    template<typename View>
    inline constexpr auto isView = false;

    // this definition does neither capture SubView nor user defined view's, but the issue resolves itself with a C++20
    // upgrade.
    template<typename Mapping, typename BlobType, typename Accessor>
    inline constexpr auto isView<View<Mapping, BlobType, Accessor>> = true;
#endif

    namespace internal
    {
        // remove in C++23, from: https://en.cppreference.com/w/cpp/utility/forward_like
        // NOLINTBEGIN
        template<class T, class U>
        [[nodiscard]] LLAMA_FN_HOST_ACC_INLINE constexpr auto&& forward_like(U&& x) noexcept
        {
            constexpr bool is_adding_const = std::is_const_v<std::remove_reference_t<T>>;
            if constexpr(std::is_lvalue_reference_v<T&&>)
            {
                if constexpr(is_adding_const)
                    return std::as_const(x);
                else
                    return static_cast<U&>(x);
            }
            else
            {
                if constexpr(is_adding_const)
                    return std::move(std::as_const(x));
                else
                    return std::move(x);
            }
        }
        // NOLINTEND

        template<typename Blobs, typename TransformBlobFunc, std::size_t... Is>
        LLAMA_FN_HOST_ACC_INLINE auto makeTransformedBlobArray(
            Blobs&& blobs,
            const TransformBlobFunc& transformBlob,
            std::integer_sequence<std::size_t, Is...>)
        {
            return llama::Array{transformBlob(forward_like<Blobs>(blobs[Is]))...};
        }
    } // namespace internal

    /// Applies the given transformation to the blobs of a view and creates a new view with the transformed blobs
    /// and the same mapping and accessor as the old view.
    LLAMA_EXPORT
    template<typename ViewFwd, typename TransformBlobFunc, typename = std::enable_if_t<isView<std::decay_t<ViewFwd>>>>
    LLAMA_FN_HOST_ACC_INLINE auto transformBlobs(ViewFwd&& view, const TransformBlobFunc& transformBlob)
    {
        using View = std::decay_t<ViewFwd>;
        constexpr auto blobCount = View::Mapping::blobCount;

        auto blobs = internal::makeTransformedBlobArray(
            internal::forward_like<ViewFwd>(view.blobs()),
            transformBlob,
            std::make_index_sequence<blobCount>{});
        return llama::View<typename View::Mapping, typename decltype(blobs)::value_type, typename View::Accessor>{
            view.mapping(),
            std::move(blobs),
            view.accessor()};
    }

    /// Creates a shallow copy of a view. This copy must not outlive the view, since it references its blob array.
    /// \tparam NewBlobType The blob type of the shallow copy. Must be a non owning pointer like type.
    /// \return A new view with the same mapping as view, where each blob refers to the blob in view.
    LLAMA_EXPORT
    template<
        typename View,
        typename NewBlobType = CopyConst<std::remove_reference_t<View>, std::byte>*,
        typename = std::enable_if_t<isView<std::decay_t<View>>>>
    LLAMA_FN_HOST_ACC_INLINE auto shallowCopy(View&& view)
    {
        if constexpr(std::is_same_v<typename std::decay_t<View>::BlobType, NewBlobType>)
            return view;
        else
            return transformBlobs(
                std::forward<View>(view),
                [](auto& blob)
                {
                    LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
                    return static_cast<NewBlobType>(&blob[0]);
                    LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
                });
    }

    // Creates a new view from an existing view with the given accessor.
    // \param view A view which's mapping and blobs are forwarded into a new view with the different accessor.
    LLAMA_EXPORT
    template<typename NewAccessor, typename ViewFwd, typename = std::enable_if_t<isView<std::decay_t<ViewFwd>>>>
    LLAMA_FN_HOST_ACC_INLINE auto withAccessor(ViewFwd&& view, NewAccessor newAccessor = {})
    {
        using OldView = std::decay_t<ViewFwd>;
        return View<typename OldView::Mapping, typename OldView::BlobType, NewAccessor>{
            internal::forward_like<ViewFwd>(view.mapping()),
            internal::forward_like<ViewFwd>(view.blobs()),
            std::move(newAccessor)};
    }

    // Creates a new view from an existing view with the given mapping.
    // \param view A view which's accessor and blobs are forwarded into a new view with the different mapping.
    LLAMA_EXPORT
    template<typename NewMapping, typename ViewFwd, typename = std::enable_if_t<isView<std::decay_t<ViewFwd>>>>
    LLAMA_FN_HOST_ACC_INLINE auto withMapping(ViewFwd&& view, NewMapping newMapping = {})
    {
        using OldView = std::decay_t<ViewFwd>;
        static_assert(OldView::Mapping::blobCount == NewMapping::blobCount);
        for(std::size_t i = 0; i < NewMapping::blobCount; i++)
        {
            assert(view.mapping().blobSize(i) == newMapping.blobSize(i));
        }

        return View<NewMapping, typename OldView::BlobType, typename OldView::Accessor>{
            std::move(newMapping),
            internal::forward_like<ViewFwd>(view.blobs()),
            internal::forward_like<ViewFwd>(view.accessor())};
    }

    /// Like a \ref View, but array indices are shifted.
    /// @tparam TStoredParentView Type of the underlying view. May be cv qualified and/or a reference type.
    LLAMA_EXPORT
    template<typename TStoredParentView>
    struct SubView
    {
        using StoredParentView = TStoredParentView;
        using ParentView = std::remove_const_t<std::remove_reference_t<StoredParentView>>; ///< type of the parent view

        using Mapping = typename ParentView::Mapping;
        using ArrayExtents = typename ParentView::ArrayExtents;
        using ArrayIndex = typename ParentView::ArrayIndex;
        using BlobType = typename ParentView::BlobType;
        using RecordDim = typename ParentView::RecordDim;
        using Accessor = typename ParentView::Accessor;
        using iterator = typename ParentView::iterator;
        using const_iterator = typename ParentView::const_iterator;

    private:
        using size_type = typename ArrayExtents::value_type;

    public:
        /// Creates a SubView given an offset. The parent view is default constructed.
        LLAMA_FN_HOST_ACC_INLINE explicit SubView(ArrayIndex offset) : offset(offset)
        {
        }

        /// Creates a SubView given a parent \ref View and offset.
        template<typename StoredParentViewFwd>
        LLAMA_FN_HOST_ACC_INLINE SubView(StoredParentViewFwd&& parentView, ArrayIndex offset)
            : parentView(std::forward<StoredParentViewFwd>(parentView))
            , offset(offset)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() -> Mapping&
        {
            return parentView.mapping();
        }

        LLAMA_FN_HOST_ACC_INLINE auto mapping() const -> const Mapping&
        {
            return parentView.mapping();
        }

        LLAMA_FN_HOST_ACC_INLINE auto accessor() -> Accessor&
        {
            return parentView.accessor();
        }

        LLAMA_FN_HOST_ACC_INLINE auto accessor() const -> const Accessor&
        {
            return parentView.accessor();
        }

        LLAMA_FN_HOST_ACC_INLINE auto extents() const -> ArrayExtents
        {
            return parentView.extents();
        }

        /// Same as \ref View::operator()(ArrayIndex), but shifted by the offset of this \ref SubView.
        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) const -> decltype(auto)
        {
            return parentView(ArrayIndex{ai + offset});
        }

        LLAMA_FN_HOST_ACC_INLINE auto operator()(ArrayIndex ai) -> decltype(auto)
        {
            return parentView(ArrayIndex{ai + offset});
        }

        /// Same as corresponding operator in \ref View, but shifted by the offset of this \ref SubView.
        template<typename... Indices>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(Indices... indices) const -> decltype(auto)
        {
            static_assert(
                sizeof...(Indices) == ArrayIndex::rank,
                "Please specify as many indices as you have array dimensions");
            static_assert(
                std::conjunction_v<std::is_convertible<Indices, size_type>...>,
                "Indices must be convertible to ArrayExtents::size_type");
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
            return parentView(
                ArrayIndex{ArrayIndex{static_cast<typename ArrayIndex::value_type>(indices)...} + offset});
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coords...> rc = {}) const -> decltype(auto)
        {
            return parentView(ArrayIndex{} + offset, rc);
        }

        template<std::size_t... Coords>
        LLAMA_FN_HOST_ACC_INLINE auto operator()(RecordCoord<Coords...> rc = {}) -> decltype(auto)
        {
            return parentView(ArrayIndex{} + offset, rc);
        }

        // TODO(bgruber): implement iterators. Those would be transform iterators on top of the parent view's
        // iterators, applying the offset on access.

        LLAMA_FN_HOST_ACC_INLINE auto blobs() -> Array<BlobType, Mapping::blobCount>&
        {
            return parentView.blobs();
        }

        LLAMA_FN_HOST_ACC_INLINE auto blobs() const -> const Array<BlobType, Mapping::blobCount>&
        {
            return parentView.blobs();
        }

        StoredParentView parentView;
        const ArrayIndex offset; ///< offset by which this view's \ref ArrayIndex indices are shifted when passed
                                 ///< to the parent view.
    };

    /// SubView vview(view); will store a reference to view.
    /// SubView vview(std::move(view)); will store the view.
    LLAMA_EXPORT
    template<typename TStoredParentView>
    SubView(TStoredParentView&&, typename std::remove_reference_t<TStoredParentView>::Mapping::ArrayExtents::Index)
        -> SubView<TStoredParentView>;
} // namespace llama

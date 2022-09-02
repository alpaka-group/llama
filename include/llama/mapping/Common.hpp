// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"

#include <climits>

namespace llama::mapping
{
    template<typename TArrayExtents, typename TRecordDim>
    struct MappingBase : private TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;
        using size_type = typename ArrayExtents::value_type;

        constexpr MappingBase() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit MappingBase(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return static_cast<const ArrayExtents&>(*this);
        }
    };

    /// Functor that maps an \ref ArrayIndex into linear numbers the way C++ arrays work. The fast moving index of the
    /// ArrayIndex object should be the last one. E.g. ArrayIndex<3> a; stores 3 indices where a[2] should be
    /// incremented in the innermost loop.
    struct LinearizeArrayDimsCpp
    {
        template<typename ArrayExtents>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayExtents& extents) -> typename ArrayExtents::value_type
        {
            return product(extents);
        }

        /// \param ai Index in the array dimensions.
        /// \param extents Total size of the array dimensions.
        /// \return Linearized index.
        template<typename ArrayExtents>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
            const typename ArrayExtents::Index& ai,
            const ArrayExtents& extents) const -> typename ArrayExtents::value_type
        {
            if constexpr(ArrayExtents::rank == 0)
                return 0;
            else
            {
                auto address = ai[0];
                for(int i = 1; i < static_cast<int>(ArrayExtents::rank); i++)
                {
                    address *= extents[i];
                    address += ai[i];
                }
                return address;
            }
        }
    };

    /// Functor that maps a \ref ArrayIndex into linear numbers the way Fortran arrays work. The fast moving index of
    /// the ArrayIndex object should be the last one. E.g. ArrayIndex<3> a; stores 3 indices where a[2] should be
    /// incremented in the innermost loop.
    struct LinearizeArrayDimsFortran
    {
        template<typename ArrayExtents>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayExtents& extents) -> typename ArrayExtents::value_type
        {
            return product(extents);
        }

        /// \param ai Index in the array dimensions.
        /// \param extents Total size of the array dimensions.
        /// \return Linearized index.
        template<typename ArrayExtents>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
            const typename ArrayExtents::Index& ai,
            const ArrayExtents& extents) const -> typename ArrayExtents::value_type
        {
            if constexpr(ArrayExtents::rank == 0)
                return 0;
            else
            {
                auto address = ai[ArrayExtents::rank - 1];
                for(int i = static_cast<int>(ArrayExtents::rank) - 2; i >= 0; i--)
                {
                    address *= extents[i];
                    address += ai[i];
                }
                return address;
            }
        }
    };

    /// Functor that maps an \ref ArrayIndex into linear numbers using the Z-order space filling curve (Morton codes).
    struct LinearizeArrayDimsMorton
    {
        template<typename ArrayExtents>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayExtents& extents) const ->
            typename ArrayExtents::value_type
        {
            if constexpr(ArrayExtents::rank == 0)
                return 0;
            else
            {
                auto longest = extents[0];
                for(int i = 1; i < static_cast<int>(ArrayExtents::rank); i++)
                    longest = std::max(longest, extents[i]);
                const auto longestPO2 = bitCeil(longest);
                return intPow(longestPO2, static_cast<typename ArrayExtents::value_type>(ArrayExtents::rank));
            }
        }

        /// \param ai Coordinate in the array dimensions.
        /// \param extents Total size of the array dimensions.
        /// \return Linearized index.
        template<typename ArrayExtents>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
            const typename ArrayExtents::Index& ai,
            [[maybe_unused]] const ArrayExtents& extents) const -> typename ArrayExtents::value_type
        {
            using size_type = typename ArrayExtents::value_type;
            constexpr auto rank = static_cast<size_type>(ArrayExtents::rank);
            size_type r = 0;
            for(size_type bit = 0; bit < (static_cast<size_type>(sizeof(size_type)) * CHAR_BIT) / rank; bit++)
                for(size_type i = 0; i < rank; i++)
                    r |= (ai[i] & (size_type{1} << bit)) << ((bit + 1) * (rank - 1) - i);
            return r;
        }

    private:
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE static constexpr auto bitCeil(T n) -> T
        {
            T r = 1u;
            while(r < n)
                r <<= 1u;
            return r;
        }

        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE static constexpr auto intPow(T b, T e) -> T
        {
            e--;
            auto r = b;
            while(e != 0u)
            {
                r *= b;
                e--;
            }
            return r;
        }
    };

    /// Flattens the record dimension in the order fields are written.
    template<typename RecordDim>
    struct FlattenRecordDimInOrder
    {
        using FlatRecordDim = llama::FlatRecordDim<RecordDim>;

        template<std::size_t... RecordCoords>
        static constexpr std::size_t flatIndex = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
    };

    /// Flattens the record dimension by sorting the fields according to a given predicate on the field types.
    /// @tparam Less A binary predicate accepting two field types, which exposes a member value. Value must be true if
    /// the first field type is less than the second one, otherwise false.
    template<typename RecordDim, template<typename, typename> typename Less>
    struct FlattenRecordDimSorted
    {
    private:
        using FlatOrigRecordDim = llama::FlatRecordDim<RecordDim>;
        using FlatSortedRecordDim = boost::mp11::mp_sort<FlatOrigRecordDim, Less>;

        template<typename A, typename B>
        using LessWithIndices
            = Less<boost::mp11::mp_at<FlatOrigRecordDim, A>, boost::mp11::mp_at<FlatOrigRecordDim, B>>;

        // A permutation from new FlatSortedRecordDim index to old FlatOrigRecordDim index
        using PermutedIndices
            = boost::mp11::mp_sort<boost::mp11::mp_iota<boost::mp11::mp_size<FlatOrigRecordDim>>, LessWithIndices>;

        template<typename A, typename B>
        using LessInvertPermutation = std::bool_constant<(
            boost::mp11::mp_at<PermutedIndices, A>::value < boost::mp11::mp_at<PermutedIndices, B>::value)>;

        // A permutation from old FlatOrigRecordDim index to new FlatSortedRecordDim index
        using InversePermutedIndices = boost::mp11::
            mp_sort<boost::mp11::mp_iota<boost::mp11::mp_size<FlatOrigRecordDim>>, LessInvertPermutation>;

    public:
        using FlatRecordDim = FlatSortedRecordDim;

        template<std::size_t... RecordCoords>
        static constexpr std::size_t flatIndex = []() constexpr
        {
            constexpr auto indexBefore = flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>;
            constexpr auto indexAfter = boost::mp11::mp_at_c<InversePermutedIndices, indexBefore>::value;
            return indexAfter;
        }
        ();
    };

    namespace internal
    {
        template<typename A, typename B>
        using LessAlignment = std::bool_constant<alignof(A) < alignof(B)>;

        template<typename A, typename B>
        using MoreAlignment = std::bool_constant<(alignof(A) > alignof(B))>;
    } // namespace internal

    /// Flattens and sorts the record dimension by increasing alignment of its fields.
    template<typename RecordDim>
    using FlattenRecordDimIncreasingAlignment = FlattenRecordDimSorted<RecordDim, internal::LessAlignment>;

    /// Flattens and sorts the record dimension by decreasing alignment of its fields.
    template<typename RecordDim>
    using FlattenRecordDimDecreasingAlignment = FlattenRecordDimSorted<RecordDim, internal::MoreAlignment>;

    /// Flattens and sorts the record dimension by the alignment of its fields to minimize padding.
    template<typename RecordDim>
    using FlattenRecordDimMinimizePadding = FlattenRecordDimIncreasingAlignment<RecordDim>;
} // namespace llama::mapping

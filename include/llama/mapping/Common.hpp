// Copyright 2022 Alexander Matthes, Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "../Core.hpp"

#include <atomic>
#include <climits>
#ifndef __cpp_lib_atomic_ref
#    include <boost/atomic/atomic_ref.hpp>
#endif

namespace llama::mapping
{
    LLAMA_EXPORT
    template<typename TArrayExtents, typename TRecordDim>
    struct MappingBase : protected TArrayExtents
    {
        using ArrayExtents = TArrayExtents;
        using RecordDim = TRecordDim;

    protected:
        using ArrayIndex = typename ArrayExtents::Index;
        using size_type = typename ArrayExtents::value_type;

    public:
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

    /// Functor that maps an \ref ArrayIndex into linear numbers, where the fast moving index should be the rightmost
    /// one, which models how C++ arrays work and is analogous to mdspan's layout_right. E.g. ArrayIndex<3> a; stores 3
    /// indices where a[2] should be incremented in the innermost loop.
    LLAMA_EXPORT
    struct LinearizeArrayIndexRight
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

    LLAMA_EXPORT
    using LinearizeArrayIndexCpp = LinearizeArrayIndexRight;

    /// Functor that maps a \ref ArrayIndex into linear numbers the way Fortran arrays work. The fast moving index of
    /// the ArrayIndex object should be the last one. E.g. ArrayIndex<3> a; stores 3 indices where a[0] should be
    /// incremented in the innermost loop.
    LLAMA_EXPORT
    struct LinearizeArrayIndexLeft
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

    LLAMA_EXPORT
    using LinearizeArrayIndexFortran = LinearizeArrayIndexLeft;

    /// Functor that maps an \ref ArrayIndex into linear numbers using the Z-order space filling curve (Morton codes).
    LLAMA_EXPORT
    struct LinearizeArrayIndexMorton
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

    /// Retains the order of the record dimension's fields.
    LLAMA_EXPORT
    template<typename TFlatRecordDim>
    struct PermuteFieldsInOrder
    {
        using FlatRecordDim = TFlatRecordDim;

        template<std::size_t FlatRecordCoord>
        static constexpr std::size_t permute = FlatRecordCoord;
    };

    /// Sorts the record dimension's the fields according to a given predicate on the field types.
    /// @tparam Less A binary predicate accepting two field types, which exposes a member value. Value must be true if
    /// the first field type is less than the second one, otherwise false.
    LLAMA_EXPORT
    template<typename FlatOrigRecordDim, template<typename, typename> typename Less>
    struct PermuteFieldsSorted
    {
    private:
        using FlatSortedRecordDim = mp_sort<FlatOrigRecordDim, Less>;

        template<typename A, typename B>
        using LessWithIndices = Less<mp_at<FlatOrigRecordDim, A>, mp_at<FlatOrigRecordDim, B>>;

        // A permutation from new FlatSortedRecordDim index to old FlatOrigRecordDim index
        using PermutedIndices = mp_sort<mp_iota<mp_size<FlatOrigRecordDim>>, LessWithIndices>;

        template<typename A, typename B>
        using LessInvertPermutation
            = std::bool_constant<(mp_at<PermutedIndices, A>::value < mp_at<PermutedIndices, B>::value)>;

        // A permutation from old FlatOrigRecordDim index to new FlatSortedRecordDim index
        using InversePermutedIndices = mp_sort<mp_iota<mp_size<FlatOrigRecordDim>>, LessInvertPermutation>;

    public:
        using FlatRecordDim = FlatSortedRecordDim;

        template<std::size_t FlatRecordCoord>
        static constexpr std::size_t permute = mp_at_c<InversePermutedIndices, FlatRecordCoord>::value;
    };

    namespace internal
    {
        template<typename A, typename B>
        using LessAlignment = std::bool_constant<alignof(A) < alignof(B)>;

        template<typename A, typename B>
        using MoreAlignment = std::bool_constant<(alignof(A) > alignof(B))>;
    } // namespace internal

    /// Sorts the record dimension fields by increasing alignment of its fields.
    LLAMA_EXPORT
    template<typename FlatRecordDim>
    using PermuteFieldsIncreasingAlignment = PermuteFieldsSorted<FlatRecordDim, internal::LessAlignment>;

    /// Sorts the record dimension fields by decreasing alignment of its fields.
    LLAMA_EXPORT
    template<typename FlatRecordDim>
    using PermuteFieldsDecreasingAlignment = PermuteFieldsSorted<FlatRecordDim, internal::MoreAlignment>;

    /// Sorts the record dimension fields by the alignment of its fields to minimize padding.
    LLAMA_EXPORT
    template<typename FlatRecordDim>
    using PermuteFieldsMinimizePadding = PermuteFieldsIncreasingAlignment<FlatRecordDim>;

    namespace internal
    {
        template<auto I>
        struct S;

        template<typename CountType>
        LLAMA_FN_HOST_ACC_INLINE void atomicInc(CountType& i)
        {
#ifdef __CUDA_ARCH__
            // if you get an error here that there is no overload of atomicAdd, your CMAKE_CUDA_ARCHITECTURE might be
            // too low or you need to use a smaller CountType for the FieldAccessCount or Heatmap mapping.
            if constexpr(mp_contains<mp_list<int, unsigned int, unsigned long long int>, CountType>::value)
                atomicAdd(&i, CountType{1});
            else if constexpr(sizeof(CountType) == sizeof(unsigned int))
                atomicAdd(reinterpret_cast<unsigned int*>(&i), 1u);
            else if constexpr(sizeof(CountType) == sizeof(unsigned long long int))
                atomicAdd(reinterpret_cast<unsigned long long int*>(&i), 1ull);
            else
                static_assert(sizeof(CountType) == 0, "There is no CUDA atomicAdd for your CountType");
#elif defined(__cpp_lib_atomic_ref)
            ++std::atomic_ref<CountType>{i};
#else
            ++boost::atomic_ref<CountType>{i};
#endif
        }
    } // namespace internal

    LLAMA_EXPORT
    enum class FieldAlignment
    {
        Pack,
        Align
    };
} // namespace llama::mapping

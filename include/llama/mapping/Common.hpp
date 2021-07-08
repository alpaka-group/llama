// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"

#include <climits>

namespace llama::mapping
{
    namespace internal
    {
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto product(const ArrayDims<Dim>& size) -> std::size_t
        {
            std::size_t prod = 1;
            for(auto s : size)
                prod *= s;
            return prod;
        }
    } // namespace internal

    /// Functor that maps a \ref ArrayDims coordinate into linear numbers the way C++ arrays work.
    struct LinearizeArrayDimsCpp
    {
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDims<Dim>& size) -> std::size_t
        {
            return internal::product(size);
        }

        /// \param coord Coordinate in the array dimensions.
        /// \param size Total size of the array dimensions.
        /// \return Linearized index.
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDims<Dim>& coord, const ArrayDims<Dim>& size)
            const -> std::size_t
        {
            if constexpr(Dim == 0)
                return 0;
            else
            {
                std::size_t address = coord[0];
                for(auto i = 1; i < Dim; i++)
                {
                    address *= size[i];
                    address += coord[i];
                }
                return address;
            }
        }
    };

    /// Functor that maps a \ref ArrayDims coordinate into linear numbers the way Fortran arrays work.
    struct LinearizeArrayDimsFortran
    {
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDims<Dim>& size) -> std::size_t
        {
            return internal::product(size);
        }

        /// \param coord Coordinate in the array dimensions.
        /// \param size Total size of the array dimensions.
        /// \return Linearized index.
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(const ArrayDims<Dim>& coord, const ArrayDims<Dim>& size)
            const -> std::size_t
        {
            if constexpr(Dim == 0)
                return 0;
            else
            {
                std::size_t address = coord[Dim - 1];
                for(int i = (int) Dim - 2; i >= 0; i--)
                {
                    address *= size[i];
                    address += coord[i];
                }
                return address;
            }
        }
    };

    /// Functor that maps a \ref ArrayDims coordinate into linear numbers using the Z-order space filling curve (Morton
    /// codes).
    struct LinearizeArrayDimsMorton
    {
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto size(const ArrayDims<Dim>& size) const -> std::size_t
        {
            if constexpr(Dim == 0)
                return 0;
            else
            {
                std::size_t longest = size[0];
                for(auto i = 1; i < Dim; i++)
                    longest = std::max(longest, size[i]);
                const auto longestPO2 = bit_ceil(longest);
                return intPow(longestPO2, Dim);
            }
        }

        /// \param coord Coordinate in the array dimensions.
        /// \param size Total size of the array dimensions.
        /// \return Linearized index.
        template<std::size_t Dim>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto operator()(
            const ArrayDims<Dim>& coord,
            [[maybe_unused]] const ArrayDims<Dim>& size) const -> std::size_t
        {
            std::size_t r = 0;
            for(std::size_t bit = 0; bit < (sizeof(std::size_t) * CHAR_BIT) / Dim; bit++)
                for(std::size_t i = 0; i < Dim; i++)
                    r |= (coord[i] & (std::size_t{1} << bit)) << ((bit + 1) * (Dim - 1) - i);
            return r;
        }

    private:
        LLAMA_FN_HOST_ACC_INLINE static constexpr auto bit_ceil(std::size_t n) -> std::size_t
        {
            std::size_t r = 1;
            while(r < n)
                r <<= 1;
            return r;
        }

        LLAMA_FN_HOST_ACC_INLINE static constexpr auto intPow(std::size_t b, std::size_t e) -> std::size_t
        {
            e--;
            auto r = b;
            while(e)
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

    /// Flattens and sorts the record dimension by the alignment of its fields.
    template<typename RecordDim>
    struct FlattenRecordDimMinimizePadding
    {
    private:
        template<typename A, typename B>
        using LessAlignment = std::bool_constant<alignof(A) < alignof(B)>;

        using FlatOrigRecordDim = llama::FlatRecordDim<RecordDim>;
        using FlatSortedRecordDim = boost::mp11::mp_sort<FlatOrigRecordDim, LessAlignment>;

        template<typename A, typename B>
        using LessAlignmentForIndices = std::bool_constant<
            alignof(boost::mp11::mp_at<FlatOrigRecordDim, A>) < alignof(boost::mp11::mp_at<FlatOrigRecordDim, B>)>;

        // A permutation from new FlatSortedRecordDim index to old FlatOrigRecordDim index
        using PermutedIndices = boost::mp11::
            mp_sort<boost::mp11::mp_iota<boost::mp11::mp_size<FlatOrigRecordDim>>, LessAlignmentForIndices>;

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
} // namespace llama::mapping

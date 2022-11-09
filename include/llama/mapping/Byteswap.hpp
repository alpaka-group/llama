// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "../Core.hpp"
#include "../ProxyRefOpMixin.hpp"
#include "Common.hpp"
#include "Projection.hpp"

namespace llama::mapping
{
    namespace internal
    {
        // TODO(bgruber): replace by std::byteswap in C++23
        template<typename T>
        LLAMA_FN_HOST_ACC_INLINE auto byteswap(T t) -> T
        {
            llama::Array<std::byte, sizeof(T)> arr;
            std::memcpy(&arr, &t, sizeof(T));

            for(std::size_t i = 0; i < sizeof(T) / 2; i++)
            {
                const auto a = arr[i];
                const auto b = arr[sizeof(T) - 1 - i];
                arr[i] = b;
                arr[sizeof(T) - 1 - i] = a;
            }

            std::memcpy(&t, &arr, sizeof(T));
            return t;
        }

        template<typename T>
        struct ByteswapProjection
        {
            LLAMA_FN_HOST_ACC_INLINE static auto load(T v) -> T
            {
                return byteswap(v);
            }

            LLAMA_FN_HOST_ACC_INLINE static auto store(T v) -> T
            {
                return byteswap(v);
            }
        };

        template<typename T>
        using MakeByteswapProjectionPair = boost::mp11::mp_list<T, ByteswapProjection<T>>;

        template<typename RecordDim>
        using MakeByteswapProjectionMap
            = boost::mp11::mp_transform<MakeByteswapProjectionPair, boost::mp11::mp_unique<FlatRecordDim<RecordDim>>>;
    } // namespace internal

    /// Mapping that swaps the byte order of all values when loading/storing.
    template<typename ArrayExtents, typename RecordDim, template<typename, typename> typename InnerMapping>
    struct Byteswap : Projection<ArrayExtents, RecordDim, InnerMapping, internal::MakeByteswapProjectionMap<RecordDim>>
    {
    private:
        using Base = Projection<ArrayExtents, RecordDim, InnerMapping, internal::MakeByteswapProjectionMap<RecordDim>>;

    public:
        using Base::Base;
    };

    /// Binds parameters to a \ref ChangeType mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<template<typename, typename> typename InnerMapping>
    struct BindByteswap
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn = Byteswap<ArrayExtents, RecordDim, InnerMapping>;
    };

    template<typename Mapping>
    inline constexpr bool isByteswap = false;

    template<typename TArrayExtents, typename TRecordDim, template<typename, typename> typename InnerMapping>
    inline constexpr bool isByteswap<Byteswap<TArrayExtents, TRecordDim, InnerMapping>> = true;
} // namespace llama::mapping

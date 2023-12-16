// Copyright 2021 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include "View.hpp"
#include "mapping/AoSoA.hpp"
#include "mapping/SoA.hpp"

#include <cstring>
#include <numeric>

namespace llama
{
    namespace internal
    {
        template<typename RecordDim>
        void assertTrivialCopyable()
        {
            forEachLeafCoord<RecordDim>(
                [](auto rc)
                {
                    static_assert(
                        std::is_trivially_copyable_v<GetType<RecordDim, decltype(rc)>>,
                        "All types in the record dimension must be trivially copyable");
                });
        }

        using memcopyFunc = void* (*) (void*, const void*, std::size_t);

        inline void parallelMemcpy(
            std::byte* dst,
            const std::byte* src,
            std::size_t size,
            std::size_t threadId = 0,
            std::size_t threadCount = 1,
            memcopyFunc singleThreadMemcpy = std::memcpy)
        {
            const auto sizePerThread = size / threadCount;
            const auto sizeLastThread = sizePerThread + size % threadCount;
            const auto sizeThisThread = threadId == threadCount - 1 ? sizeLastThread : sizePerThread;
            singleThreadMemcpy(dst + threadId * sizePerThread, src + threadId * sizePerThread, sizeThisThread);
        }
    } // namespace internal

    /// Direct memcpy from source view blobs to destination view blobs. Both views need to have the same mappings with
    /// the same array dimensions.
    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
    LLAMA_EXPORT
    template<typename Mapping, typename SrcBlob, typename DstBlob>
    void blobMemcpy(
        const View<Mapping, SrcBlob>& srcView,
        View<Mapping, DstBlob>& dstView,
        std::size_t threadId = 0,
        std::size_t threadCount = 1)
    {
        internal::assertTrivialCopyable<typename Mapping::RecordDim>();

        // TODO(bgruber): we do not verify if the mappings have other runtime state than the array dimensions
        if(srcView.extents() != dstView.extents())
            throw std::runtime_error{"Array dimensions sizes are different"};

        // TODO(bgruber): this is maybe not the best parallel copying strategy
        for(std::size_t i = 0; i < Mapping::blobCount; i++)
            internal::parallelMemcpy(
                &dstView.blobs()[i][0],
                &srcView.blobs()[i][0],
                dstView.mapping().blobSize(i),
                threadId,
                threadCount);
    }

    /// Field-wise copy from source to destination view. Both views need to have the same array and record dimensions.
    /// @param threadId Optional. Thread id in case of multi-threaded copy.
    /// @param threadCount Optional. Thread count in case of multi-threaded copy.
    LLAMA_EXPORT
    template<typename SrcMapping, typename SrcBlob, typename DstMapping, typename DstBlob>
    void fieldWiseCopy(
        const View<SrcMapping, SrcBlob>& srcView,
        View<DstMapping, DstBlob>& dstView,
        std::size_t threadId = 0,
        std::size_t threadCount = 1)
    {
        // TODO(bgruber): think if we can remove this restriction
        static_assert(
            std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>,
            "The source and destination record dimensions must be the same");

        if(srcView.extents() != dstView.extents())
            throw std::runtime_error{"Array dimensions sizes are different"};

        auto copyOne = [&](auto ai) LLAMA_LAMBDA_INLINE
        {
            forEachLeafCoord<typename DstMapping::RecordDim>([&](auto rc) LLAMA_LAMBDA_INLINE
                                                             { dstView(ai)(rc) = srcView(ai)(rc); });
        };

        constexpr auto dims = SrcMapping::ArrayExtents::rank;
        const auto extents = srcView.extents().toArray();
        const auto workPerThread = (extents[0] + threadCount - 1) / threadCount;
        const auto start = threadId * workPerThread;
        const auto end = std::min((threadId + 1) * workPerThread, static_cast<std::size_t>(extents[0]));
        for(auto i = start; i < end; i++)
        {
            using SrcSizeType = typename SrcMapping::ArrayExtents::value_type;
            if constexpr(dims > 1)
                forEachArrayIndex(extents, copyOne, static_cast<SrcSizeType>(i));
            else
                copyOne(ArrayIndex<SrcSizeType, dims>{static_cast<std::size_t>(i)});
        }
    }

    namespace internal
    {
        template<typename Mapping>
        inline constexpr std::size_t aosoaLanes = 0;

        template<
            typename ArrayExtents,
            typename RecordDim,
            mapping::Blobs Blobs,
            mapping::SubArrayAlignment SubArrayAlignment,
            typename LinearizeArrayIndexFunctor,
            template<typename>
            typename PermuteSBFields>
        inline constexpr std::size_t aosoaLanes<
            mapping::
                SoA<ArrayExtents, RecordDim, Blobs, SubArrayAlignment, LinearizeArrayIndexFunctor, PermuteSBFields>>
            = std::numeric_limits<std::size_t>::max();

        template<
            typename ArrayExtents,
            typename RecordDim,
            typename ArrayExtents::value_type Lanes,
            typename LinearizeArrayIndexFunctor,
            template<typename>
            typename PermuteFields>
        inline constexpr std::size_t
            aosoaLanes<mapping::AoSoA<ArrayExtents, RecordDim, Lanes, LinearizeArrayIndexFunctor, PermuteFields>>
            = Lanes;
    } // namespace internal

    /// AoSoA copy strategy which transfers data in common blocks. SoA mappings are also allowed for at most 1
    /// argument.
    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
    LLAMA_EXPORT
    template<typename SrcMapping, typename SrcBlob, typename DstMapping, typename DstBlob>
    void aosoaCommonBlockCopy(
        const View<SrcMapping, SrcBlob>& srcView,
        View<DstMapping, DstBlob>& dstView,
        bool readOpt,
        std::size_t threadId = 0,
        std::size_t threadCount = 1)
    {
        // TODO(bgruber): think if we can remove this restriction
        static_assert(
            std::is_same_v<typename SrcMapping::RecordDim, typename DstMapping::RecordDim>,
            "The source and destination record dimensions must be the same");
        static_assert(
            std::is_same_v<
                typename SrcMapping::LinearizeArrayIndexFunctor,
                typename DstMapping::LinearizeArrayIndexFunctor>,
            "Source and destination mapping need to use the same array dimensions linearizer");
        using RecordDim = typename SrcMapping::RecordDim;
        internal::assertTrivialCopyable<RecordDim>();

        [[maybe_unused]] static constexpr bool isSrcMB = SrcMapping::blobCount > 1;
        [[maybe_unused]] static constexpr bool isDstMB = DstMapping::blobCount > 1;
        static constexpr auto lanesSrc = internal::aosoaLanes<SrcMapping>;
        static constexpr auto lanesDst = internal::aosoaLanes<DstMapping>;

        if(srcView.extents() != dstView.extents())
            throw std::runtime_error{"Array dimensions sizes are different"};

        static constexpr auto srcIsAoSoA = lanesSrc != std::numeric_limits<std::size_t>::max();
        static constexpr auto dstIsAoSoA = lanesDst != std::numeric_limits<std::size_t>::max();

        static_assert(srcIsAoSoA || dstIsAoSoA, "At least one of the mappings must be an AoSoA mapping");
        static_assert(!srcIsAoSoA || SrcMapping::blobCount == 1, "Implementation assumes AoSoA with single blob");
        static_assert(!dstIsAoSoA || DstMapping::blobCount == 1, "Implementation assumes AoSoA with single blob");

        const auto flatSize = product(dstView.extents());

        // TODO(bgruber): implement the following by adding additional copy loops for the remaining elements
        if(!srcIsAoSoA && flatSize % lanesDst != 0)
            throw std::runtime_error{"Source SoA mapping's total array elements must be evenly divisible by the "
                                     "destination AoSoA Lane count."};
        if(!dstIsAoSoA && flatSize % lanesSrc != 0)
            throw std::runtime_error{"Destination SoA mapping's total array elements must be evenly divisible by the "
                                     "source AoSoA Lane count."};

        // the same as AoSoA::blobNrAndOffset but takes a flat array index
        auto mapAoSoA = [](std::size_t flatArrayIndex, auto rc, std::size_t Lanes) LLAMA_LAMBDA_INLINE
        {
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = (sizeOf<RecordDim> * Lanes) * blockIndex + offsetOf<RecordDim, decltype(rc)> * Lanes
                + sizeof(GetType<RecordDim, decltype(rc)>) * laneIndex;
            return offset;
        };
        // the same as SoA::blobNrAndOffset but takes a flat array index
        auto mapSoA = [&](std::size_t flatArrayIndex, auto rc, bool mb) LLAMA_LAMBDA_INLINE
        {
            const auto blob = mb * flatRecordCoord<RecordDim, decltype(rc)>;
            const auto offset = !mb * offsetOf<RecordDim, decltype(rc)> * flatSize
                + sizeof(GetType<RecordDim, decltype(rc)>) * flatArrayIndex;
            return NrAndOffset{blob, offset};
        };

        auto mapSrc = [&](std::size_t flatArrayIndex, auto rc) LLAMA_LAMBDA_INLINE
        {
            if constexpr(srcIsAoSoA)
                return &srcView.blobs()[0][0] + mapAoSoA(flatArrayIndex, rc, lanesSrc);
            else
            {
                const auto [blob, off] = mapSoA(flatArrayIndex, rc, isSrcMB);
                return &srcView.blobs()[blob][off];
            }
        };
        auto mapDst = [&](std::size_t flatArrayIndex, auto rc) LLAMA_LAMBDA_INLINE
        {
            if constexpr(dstIsAoSoA)
                return &dstView.blobs()[0][0] + mapAoSoA(flatArrayIndex, rc, lanesDst);
            else
            {
                const auto [blob, off] = mapSoA(flatArrayIndex, rc, isDstMB);
                return &dstView.blobs()[blob][off];
            }
        };

        static constexpr auto l = []
        {
            if constexpr(srcIsAoSoA && dstIsAoSoA)
                return std::gcd(lanesSrc, lanesDst);
            return std::min(lanesSrc, lanesDst);
        }();
        if(readOpt)
        {
            // optimized for linear reading
            constexpr auto srcL = srcIsAoSoA ? lanesSrc : l;
            const auto elementsPerThread = flatSize / srcL / threadCount * srcL;
            {
                const auto start = threadId * elementsPerThread;
                const auto stop = threadId == threadCount - 1 ? flatSize : (threadId + 1) * elementsPerThread;

                auto copyLBlock = [&](const std::byte*& threadSrc, std::size_t dstIndex, auto rc) LLAMA_LAMBDA_INLINE
                {
                    constexpr auto bytes = l * sizeof(GetType<RecordDim, decltype(rc)>);
                    std::memcpy(mapDst(dstIndex, rc), threadSrc, bytes);
                    threadSrc += bytes;
                };
                if constexpr(srcIsAoSoA)
                {
                    auto* threadSrc = mapSrc(start, RecordCoord<>{});
                    for(std::size_t i = start; i < stop; i += lanesSrc)
                        forEachLeafCoord<RecordDim>(
                            [&](auto rc) LLAMA_LAMBDA_INLINE
                            {
                                for(std::size_t j = 0; j < lanesSrc; j += l)
                                    copyLBlock(threadSrc, i + j, rc);
                            });
                }
                else
                {
                    forEachLeafCoord<RecordDim>(
                        [&](auto rc) LLAMA_LAMBDA_INLINE
                        {
                            auto* threadSrc = mapSrc(start, rc);
                            for(std::size_t i = start; i < stop; i += l)
                                copyLBlock(threadSrc, i, rc);
                        });
                }
            }
        }
        else
        {
            // optimized for linear writing
            constexpr auto dstL = dstIsAoSoA ? lanesDst : l;
            const auto elementsPerThread = flatSize / dstL / threadCount * dstL;
            {
                const auto start = threadId * elementsPerThread;
                const auto stop = threadId == threadCount - 1 ? flatSize : (threadId + 1) * elementsPerThread;

                auto copyLBlock = [&](std::byte*& threadDst, std::size_t srcIndex, auto rc) LLAMA_LAMBDA_INLINE
                {
                    constexpr auto bytes = l * sizeof(GetType<RecordDim, decltype(rc)>);
                    std::memcpy(threadDst, mapSrc(srcIndex, rc), bytes);
                    threadDst += bytes;
                };
                if constexpr(dstIsAoSoA)
                {
                    auto* threadDst = mapDst(start, RecordCoord<>{});
                    for(std::size_t i = start; i < stop; i += lanesDst)
                        forEachLeafCoord<RecordDim>(
                            [&](auto rc) LLAMA_LAMBDA_INLINE
                            {
                                for(std::size_t j = 0; j < lanesDst; j += l)
                                    copyLBlock(threadDst, i + j, rc);
                            });
                }
                else
                {
                    forEachLeafCoord<RecordDim>(
                        [&](auto rc) LLAMA_LAMBDA_INLINE
                        {
                            auto* threadDst = mapDst(start, rc);
                            for(std::size_t i = start; i < stop; i += l)
                                copyLBlock(threadDst, i, rc);
                        });
                }
            }
        }
    }

    /// @brief Generic implementation of \ref copy defaulting to \ref fieldWiseCopy. LLAMA provides several
    /// specializations of this construct for specific mappings. Users are encouraged to also specialize this template
    /// with better copy algorithms for further combinations of mappings, if they can and want to provide a better
    /// implementation.
    LLAMA_EXPORT
    template<typename SrcMapping, typename DstMapping, typename SFINAE = void>
    struct Copy
    {
        template<typename SrcView, typename DstView>
        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
        {
            fieldWiseCopy(srcView, dstView, threadId, threadCount);
        }
    };

    LLAMA_EXPORT
    template<typename Mapping>
    struct Copy<Mapping, Mapping>
    {
        template<typename SrcView, typename DstView>
        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
        {
            blobMemcpy(srcView, dstView, threadId, threadCount);
        }
    };

    LLAMA_EXPORT
    template<
        typename ArrayExtents,
        typename RecordDim,
        typename LinearizeArrayIndex,
        std::size_t LanesSrc,
        std::size_t LanesDst,
        template<typename>
        typename PermuteFields>
    struct Copy<
        mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>,
        mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>,
        std::enable_if_t<LanesSrc != LanesDst>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>, SrcBlob>&
                srcView,
            View<mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>, DstBlob>&
                dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    LLAMA_EXPORT
    template<
        typename ArrayExtents,
        typename RecordDim,
        typename LinearizeArrayIndex,
        template<typename>
        typename PermuteFields,
        std::size_t LanesSrc,
        mapping::Blobs DstBlobs,
        mapping::SubArrayAlignment DstSubArrayAlignment>
    struct Copy<
        mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>,
        mapping::SoA<ArrayExtents, RecordDim, DstBlobs, DstSubArrayAlignment, LinearizeArrayIndex, PermuteFields>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::AoSoA<ArrayExtents, RecordDim, LanesSrc, LinearizeArrayIndex, PermuteFields>, SrcBlob>&
                srcView,
            View<
                mapping::
                    SoA<ArrayExtents, RecordDim, DstBlobs, DstSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
                DstBlob>& dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    LLAMA_EXPORT
    template<
        typename ArrayExtents,
        typename RecordDim,
        typename LinearizeArrayIndex,
        template<typename>
        typename PermuteFields,
        std::size_t LanesDst,
        mapping::Blobs SrcBlobs,
        mapping::SubArrayAlignment SrcSubArrayAlignment>
    struct Copy<
        mapping::SoA<ArrayExtents, RecordDim, SrcBlobs, SrcSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
        mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<
                mapping::
                    SoA<ArrayExtents, RecordDim, SrcBlobs, SrcSubArrayAlignment, LinearizeArrayIndex, PermuteFields>,
                SrcBlob>& srcView,
            View<mapping::AoSoA<ArrayExtents, RecordDim, LanesDst, LinearizeArrayIndex, PermuteFields>, DstBlob>&
                dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    /// Copy data from source view to destination view. Both views need to have the same array and record
    /// dimensions. Delegates to \ref Copy to choose an implementation.
    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
    LLAMA_EXPORT
    template<typename SrcMapping, typename SrcBlob, typename DstMapping, typename DstBlob>
    void copy(
        const View<SrcMapping, SrcBlob>& srcView,
        View<DstMapping, DstBlob>& dstView,
        std::size_t threadId = 0,
        std::size_t threadCount = 1)
    {
        Copy<SrcMapping, DstMapping>{}(srcView, dstView, threadId, threadCount);
    }
} // namespace llama

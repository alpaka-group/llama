// SPDX-License-Identifier: GPL-3.0-or-later

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
            forEachLeaf<RecordDim>(
                [](auto coord)
                {
                    static_assert(
                        std::is_trivially_copyable_v<GetType<RecordDim, decltype(coord)>>,
                        "All types in the record dimension must be trivially copyable");
                });
        }

        using memcopyFunc = void* (*) (void*, const void*, std::size_t);

        inline void parallel_memcpy(
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
    template<typename Mapping, typename SrcBlob, typename DstBlob>
    void blobMemcpy(
        const View<Mapping, SrcBlob>& srcView,
        View<Mapping, DstBlob>& dstView,
        std::size_t threadId = 0,
        std::size_t threadCount = 1)
    {
        internal::assertTrivialCopyable<typename Mapping::RecordDim>();

        // TODO(bgruber): we do not verify if the mappings have other runtime state than the array dimensions
        if(srcView.mapping().arrayDims() != dstView.mapping().arrayDims())
            throw std::runtime_error{"Array dimensions sizes are different"};

        // TODO(bgruber): this is maybe not the best parallel copying strategy
        for(std::size_t i = 0; i < Mapping::blobCount; i++)
            internal::parallel_memcpy(
                &dstView.storageBlobs[i][0],
                &srcView.storageBlobs[i][0],
                dstView.mapping().blobSize(i),
                threadId,
                threadCount);
    }

    /// Field-wise copy from source to destination view. Both views need to have the same array and record dimensions.
    /// @param threadId Optional. Thread id in case of multi-threaded copy.
    /// @param threadCount Optional. Thread count in case of multi-threaded copy.
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

        if(srcView.mapping().arrayDims() != dstView.mapping().arrayDims())
            throw std::runtime_error{"Array dimensions sizes are different"};

        auto copyOne = [&](auto ad) LLAMA_LAMBDA_INLINE
        {
            forEachLeaf<typename DstMapping::RecordDim>([&](auto coord) LLAMA_LAMBDA_INLINE
                                                        { dstView(ad)(coord) = srcView(ad)(coord); });
        };

        constexpr auto dims = SrcMapping::ArrayDims::rank;
        const auto& adSize = srcView.mapping().arrayDims();
        const auto workPerThread = (adSize[0] + threadCount - 1) / threadCount;
        const auto start = threadId * workPerThread;
        const auto end = std::min((threadId + 1) * workPerThread, adSize[0]);
        for(auto i = threadId * workPerThread; i < end; i++)
        {
            if constexpr(dims > 1)
                forEachADCoord(ArrayDims<dims - 1>{pop_front(adSize)}, copyOne, static_cast<std::size_t>(i));
            else
                copyOne(ArrayDims<dims>{static_cast<std::size_t>(i)});
        }
    }

    namespace internal
    {
        template<typename Mapping>
        inline constexpr std::size_t aosoaLanes = 0;

        template<typename ArrayDims, typename RecordDim, bool SeparateBuffers, typename LinearizeArrayDimsFunctor>
        inline constexpr std::size_t aosoaLanes<
            mapping::SoA<ArrayDims, RecordDim, SeparateBuffers, LinearizeArrayDimsFunctor>> = std::
            numeric_limits<std::size_t>::max();

        template<typename ArrayDims, typename RecordDim, std::size_t Lanes, typename LinearizeArrayDimsFunctor>
        inline constexpr std::size_t
            aosoaLanes<mapping::AoSoA<ArrayDims, RecordDim, Lanes, LinearizeArrayDimsFunctor>> = Lanes;
    } // namespace internal

    /// AoSoA copy strategy which transfers data in common blocks. SoA mappings are also allowed for at most 1
    /// argument.
    /// @param threadId Optional. Zero-based id of calling thread for multi-threaded invocations.
    /// @param threadCount Optional. Thread count in case of multi-threaded invocation.
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
                typename SrcMapping::LinearizeArrayDimsFunctor,
                typename DstMapping::LinearizeArrayDimsFunctor>,
            "Source and destination mapping need to use the same array dimensions linearizer");
        using RecordDim = typename SrcMapping::RecordDim;
        internal::assertTrivialCopyable<RecordDim>();

        static constexpr bool MBSrc = SrcMapping::blobCount > 1;
        static constexpr bool MBDst = DstMapping::blobCount > 1;
        static constexpr auto LanesSrc = internal::aosoaLanes<SrcMapping>;
        static constexpr auto LanesDst = internal::aosoaLanes<DstMapping>;

        if(srcView.mapping().arrayDims() != dstView.mapping().arrayDims())
            throw std::runtime_error{"Array dimensions sizes are different"};

        static constexpr auto srcIsAoSoA = LanesSrc != std::numeric_limits<std::size_t>::max();
        static constexpr auto dstIsAoSoA = LanesDst != std::numeric_limits<std::size_t>::max();

        static_assert(srcIsAoSoA || dstIsAoSoA, "At least one of the mappings must be an AoSoA mapping");
        static_assert(
            !srcIsAoSoA || decltype(srcView.storageBlobs)::rank == 1,
            "Implementation assumes AoSoA with single blob");
        static_assert(
            !dstIsAoSoA || decltype(dstView.storageBlobs)::rank == 1,
            "Implementation assumes AoSoA with single blob");

        const auto arrayDims = dstView.mapping().arrayDims();
        const auto flatSize
            = std::reduce(std::begin(arrayDims), std::end(arrayDims), std::size_t{1}, std::multiplies<>{});

        // TODO(bgruber): implement the following by adding additional copy loops for the remaining elements
        if(!srcIsAoSoA && flatSize % LanesDst != 0)
            throw std::runtime_error{"Source SoA mapping's total array elements must be evenly divisible by the "
                                     "destination AoSoA Lane count."};
        if(!dstIsAoSoA && flatSize % LanesSrc != 0)
            throw std::runtime_error{"Destination SoA mapping's total array elements must be evenly divisible by the "
                                     "source AoSoA Lane count."};

        // the same as AoSoA::blobNrAndOffset but takes a flat array index
        auto mapAoSoA = [](std::size_t flatArrayIndex, auto coord, std::size_t Lanes) LLAMA_LAMBDA_INLINE
        {
            const auto blockIndex = flatArrayIndex / Lanes;
            const auto laneIndex = flatArrayIndex % Lanes;
            const auto offset = (sizeOf<RecordDim> * Lanes) * blockIndex + offsetOf<RecordDim, decltype(coord)> * Lanes
                + sizeof(GetType<RecordDim, decltype(coord)>) * laneIndex;
            return offset;
        };
        // the same as SoA::blobNrAndOffset but takes a flat array index
        auto mapSoA = [&](std::size_t flatArrayIndex, auto coord, bool mb) LLAMA_LAMBDA_INLINE
        {
            const auto blob = mb * flatRecordCoord<RecordDim, decltype(coord)>;
            const auto offset = !mb * offsetOf<RecordDim, decltype(coord)> * flatSize
                + sizeof(GetType<RecordDim, decltype(coord)>) * flatArrayIndex;
            return NrAndOffset{blob, offset};
        };

        auto mapSrc = [&srcView, &mapAoSoA, &mapSoA](std::size_t flatArrayIndex, auto coord) LLAMA_LAMBDA_INLINE
        {
            if constexpr(srcIsAoSoA)
                return &srcView.storageBlobs[0][0] + mapAoSoA(flatArrayIndex, coord, LanesSrc);
            else
            {
                const auto [blob, off] = mapSoA(flatArrayIndex, coord, MBSrc);
                return &srcView.storageBlobs[blob][off];
            }
        };
        auto mapDst = [&dstView, &mapAoSoA, &mapSoA](std::size_t flatArrayIndex, auto coord) LLAMA_LAMBDA_INLINE
        {
            if constexpr(dstIsAoSoA)
                return &dstView.storageBlobs[0][0] + mapAoSoA(flatArrayIndex, coord, LanesDst);
            else
            {
                const auto [blob, off] = mapSoA(flatArrayIndex, coord, MBDst);
                return &dstView.storageBlobs[blob][off];
            }
        };

        static constexpr auto L = []
        {
            if constexpr(srcIsAoSoA && dstIsAoSoA)
                return std::gcd(LanesSrc, LanesDst);
            return std::min(LanesSrc, LanesDst);
        }();
        if(readOpt)
        {
            // optimized for linear reading
            constexpr auto srcL = srcIsAoSoA ? LanesSrc : L;
            const auto elementsPerThread = flatSize / srcL / threadCount * srcL;
            {
                const auto start = threadId * elementsPerThread;
                const auto stop = threadId == threadCount - 1 ? flatSize : (threadId + 1) * elementsPerThread;

                auto copyLBlock
                    = [&](const std::byte*& threadSrc, std::size_t dstIndex, auto coord) LLAMA_LAMBDA_INLINE
                {
                    constexpr auto bytes = L * sizeof(GetType<RecordDim, decltype(coord)>);
                    std::memcpy(mapDst(dstIndex, coord), threadSrc, bytes);
                    threadSrc += bytes;
                };
                if constexpr(srcIsAoSoA)
                {
                    auto* threadSrc = mapSrc(start, RecordCoord<>{});
                    for(std::size_t i = start; i < stop; i += LanesSrc)
                        forEachLeaf<RecordDim>(
                            [&](auto coord) LLAMA_LAMBDA_INLINE
                            {
                                for(std::size_t j = 0; j < LanesSrc; j += L)
                                    copyLBlock(threadSrc, i + j, coord);
                            });
                }
                else
                {
                    forEachLeaf<RecordDim>(
                        [&](auto coord) LLAMA_LAMBDA_INLINE
                        {
                            auto* threadSrc = mapSrc(start, coord);
                            for(std::size_t i = start; i < stop; i += L)
                                copyLBlock(threadSrc, i, coord);
                        });
                }
            }
        }
        else
        {
            // optimized for linear writing
            constexpr auto dstL = dstIsAoSoA ? LanesDst : L;
            const auto elementsPerThread = flatSize / dstL / threadCount * dstL;
            {
                const auto start = threadId * elementsPerThread;
                const auto stop = threadId == threadCount - 1 ? flatSize : (threadId + 1) * elementsPerThread;

                auto copyLBlock = [&](std::byte*& threadDst, std::size_t srcIndex, auto coord) LLAMA_LAMBDA_INLINE
                {
                    constexpr auto bytes = L * sizeof(GetType<RecordDim, decltype(coord)>);
                    std::memcpy(threadDst, mapSrc(srcIndex, coord), bytes);
                    threadDst += bytes;
                };
                if constexpr(dstIsAoSoA)
                {
                    auto* threadDst = mapDst(start, RecordCoord<>{});
                    for(std::size_t i = start; i < stop; i += LanesDst)
                        forEachLeaf<RecordDim>(
                            [&](auto coord) LLAMA_LAMBDA_INLINE
                            {
                                for(std::size_t j = 0; j < LanesDst; j += L)
                                    copyLBlock(threadDst, i + j, coord);
                            });
                }
                else
                {
                    forEachLeaf<RecordDim>(
                        [&](auto coord) LLAMA_LAMBDA_INLINE
                        {
                            auto* threadDst = mapDst(start, coord);
                            for(std::size_t i = start; i < stop; i += L)
                                copyLBlock(threadDst, i, coord);
                        });
                }
            }
        }
    }

    /// @brief Generic implementation of \ref copy defaulting to \ref fieldWiseCopy. LLAMA provides several
    /// specializations of this construct for specific mappings. Users are encouraged to also specialize this template
    /// with better copy algorithms for further combinations of mappings, if they can and want to provide a better
    /// implementation.
    template<typename SrcMapping, typename DstMapping, typename SFINAE = void>
    struct Copy
    {
        template<typename SrcView, typename DstView>
        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
        {
            fieldWiseCopy(srcView, dstView, threadId, threadCount);
        }
    };

    template<typename Mapping>
    struct Copy<Mapping, Mapping>
    {
        template<typename SrcView, typename DstView>
        void operator()(const SrcView& srcView, DstView& dstView, std::size_t threadId, std::size_t threadCount) const
        {
            blobMemcpy(srcView, dstView, threadId, threadCount);
        }
    };

    template<
        typename ArrayDims,
        typename RecordDim,
        typename LinearizeArrayDims,
        std::size_t LanesSrc,
        std::size_t LanesDst>
    struct Copy<
        mapping::AoSoA<ArrayDims, RecordDim, LanesSrc, LinearizeArrayDims>,
        mapping::AoSoA<ArrayDims, RecordDim, LanesDst, LinearizeArrayDims>,
        std::enable_if_t<LanesSrc != LanesDst>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::AoSoA<ArrayDims, RecordDim, LanesSrc, LinearizeArrayDims>, SrcBlob>& srcView,
            View<mapping::AoSoA<ArrayDims, RecordDim, LanesDst, LinearizeArrayDims>, DstBlob>& dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    template<
        typename ArrayDims,
        typename RecordDim,
        typename LinearizeArrayDims,
        std::size_t LanesSrc,
        bool DstSeparateBuffers>
    struct Copy<
        mapping::AoSoA<ArrayDims, RecordDim, LanesSrc, LinearizeArrayDims>,
        mapping::SoA<ArrayDims, RecordDim, DstSeparateBuffers, LinearizeArrayDims>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::AoSoA<ArrayDims, RecordDim, LanesSrc, LinearizeArrayDims>, SrcBlob>& srcView,
            View<mapping::SoA<ArrayDims, RecordDim, DstSeparateBuffers, LinearizeArrayDims>, DstBlob>& dstView,
            std::size_t threadId,
            std::size_t threadCount)
        {
            constexpr auto readOpt = true; // TODO(bgruber): how to choose?
            aosoaCommonBlockCopy(srcView, dstView, readOpt, threadId, threadCount);
        }
    };

    template<
        typename ArrayDims,
        typename RecordDim,
        typename LinearizeArrayDims,
        std::size_t LanesDst,
        bool SrcSeparateBuffers>
    struct Copy<
        mapping::SoA<ArrayDims, RecordDim, SrcSeparateBuffers, LinearizeArrayDims>,
        mapping::AoSoA<ArrayDims, RecordDim, LanesDst, LinearizeArrayDims>>
    {
        template<typename SrcBlob, typename DstBlob>
        void operator()(
            const View<mapping::SoA<ArrayDims, RecordDim, SrcSeparateBuffers, LinearizeArrayDims>, SrcBlob>& srcView,
            View<mapping::AoSoA<ArrayDims, RecordDim, LanesDst, LinearizeArrayDims>, DstBlob>& dstView,
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

// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include "../StructName.hpp"
#include "Common.hpp"

#include <cstdio>
#include <iomanip>
#include <iostream>

namespace llama::mapping
{
    template<typename CountType>
    struct AccessCounts
    {
        union
        {
            CountType memLocsComputed;
            CountType reads;
        };
        CountType writes;
    };

    namespace internal
    {
        template<typename Value, typename Ref, typename Count>
        struct FieldAccessCountReference : ProxyRefOpMixin<FieldAccessCountReference<Value, Ref, Count>, Value>
        {
            using value_type = Value;

            template<typename RefFwd>
            LLAMA_FN_HOST_ACC_INLINE constexpr FieldAccessCountReference(RefFwd&& r, AccessCounts<Count>* hits)
                : r(std::forward<RefFwd>(r))
                , hits(hits)
            {
                static_assert(std::is_same_v<std::remove_reference_t<Ref>, std::remove_reference_t<RefFwd>>);
            }

            FieldAccessCountReference(const FieldAccessCountReference&) = default;
            FieldAccessCountReference(FieldAccessCountReference&&) noexcept = default;
            auto operator=(FieldAccessCountReference&& ref) noexcept -> FieldAccessCountReference& = default;
            ~FieldAccessCountReference() = default;

            LLAMA_FN_HOST_ACC_INLINE auto operator=(const FieldAccessCountReference& ref) -> FieldAccessCountReference&
            {
                if(&ref != this)
                {
                    internal::atomicInc(hits->writes);
                    r = static_cast<value_type>(ref);
                }
                return *this;
            }

            LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type value) -> FieldAccessCountReference&
            {
                internal::atomicInc(hits->writes);
                r = value;
                return *this;
            }

            // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
            LLAMA_FN_HOST_ACC_INLINE operator value_type() const
            {
                internal::atomicInc(hits->reads);
                return static_cast<value_type>(r);
            }

        private:
            Ref r;
            AccessCounts<Count>* hits;
        };
    } // namespace internal

    /// Forwards all calls to the inner mapping. Counts all accesses made through this mapping and allows printing a
    /// summary.
    /// @tparam Mapping The type of the inner mapping.
    /// @tparam TCountType The type used for counting the number of accesses.
    /// @tparam MyCodeHandlesProxyReferences If false, FieldAccessCount will avoid proxy references but can then only
    /// count the number of address computations
    template<typename Mapping, typename TCountType = std::size_t, bool MyCodeHandlesProxyReferences = true>
    struct FieldAccessCount : Mapping
    {
    private:
        using size_type = typename Mapping::ArrayExtents::value_type;

    public:
        using RecordDim = typename Mapping::RecordDim;
        using CountType = TCountType;
        inline static constexpr bool myCodeHandlesProxyReferences = MyCodeHandlesProxyReferences;

        struct FieldHitsArray : Array<AccessCounts<CountType>, flatFieldCount<RecordDim>>
        {
            LLAMA_FN_HOST_ACC_INLINE auto total() const -> AccessCounts<CountType>
            {
                AccessCounts<CountType> total{};
                for(const auto& ac : *this)
                {
                    if constexpr(MyCodeHandlesProxyReferences)
                    {
                        total.reads += ac.reads;
                        total.writes += ac.writes;
                    }
                    else
                        total.memLocsComputed += ac.memLocsComputed;
                }
                return total;
            }

            struct TotalBytes
            {
                CountType totalRead;
                CountType totalWritten;
            };

            /// When MyCodeHandlesProxyReferences is true, return a pair of the total read and written bytes. If false,
            /// returns the total bytes of accessed data as a single value.
            LLAMA_FN_HOST_ACC_INLINE auto totalBytes() const
            {
                CountType r = 0;
                CountType w = 0; // NOLINT(misc-const-correctness)
                forEachLeafCoord<RecordDim>(
                    [&](auto rc)
                    {
                        const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
                        const auto fieldSize = sizeof(GetType<RecordDim, decltype(rc)>);
                        if constexpr(MyCodeHandlesProxyReferences)
                        {
                            r += (*this)[i].reads * fieldSize;
                            w += (*this)[i].writes * fieldSize;
                        }
                        else
                            r += (*this)[i].memLocsComputed * fieldSize;
                    });
                if constexpr(MyCodeHandlesProxyReferences)
                    return TotalBytes{r, w};
                else
                    return r;
            }
        };

        inline static constexpr auto blobCount = Mapping::blobCount + 1;

        constexpr FieldAccessCount() = default;

        LLAMA_FN_HOST_ACC_INLINE
        explicit FieldAccessCount(Mapping mapping) : Mapping(std::move(mapping))
        {
        }

        template<typename... Args>
        LLAMA_FN_HOST_ACC_INLINE explicit FieldAccessCount(Args&&... innerArgs)
            : Mapping(std::forward<Args>(innerArgs)...)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr auto blobSize(size_type blobIndex) const -> size_type
        {
            if(blobIndex < size_type{Mapping::blobCount})
                return inner().blobSize(blobIndex);
            return sizeof(FieldHitsArray);
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>)
        {
            return true;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto compute(
            typename Mapping::ArrayIndex ai,
            RecordCoord<RecordCoords...> rc,
            Blobs& blobs) const -> decltype(auto)
        {
            static_assert(
                !std::is_const_v<Blobs>,
                "Cannot access (even just reading) data through FieldAccessCount from const blobs/view, since we need "
                "to write "
                "the access counts");

            auto& hits = fieldHits(blobs)[+flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>];
            decltype(auto) ref = mapToMemory(inner(), ai, rc, blobs); // T& or proxy reference (value)
            if constexpr(MyCodeHandlesProxyReferences)
            {
                using Value = GetType<RecordDim, decltype(rc)>;
                using Ref = decltype(ref);
                return internal::FieldAccessCountReference<Value, Ref, CountType>{std::forward<Ref>(ref), &hits};
            }
            else
            {
                internal::atomicInc(hits.memLocsComputed);
                return ref;
            }
        }

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto fieldHits(const Blobs& blobs) const -> const FieldHitsArray&
        {
            return reinterpret_cast<const FieldHitsArray&>(*&blobs[blobCount - 1][0]);
        }

        LLAMA_SUPPRESS_HOST_DEVICE_WARNING
        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto fieldHits(Blobs& blobs) const -> FieldHitsArray&
        {
            return reinterpret_cast<FieldHitsArray&>(*&blobs[blobCount - 1][0]);
        }

        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE void printFieldHits(const Blobs& blobs) const
        {
            printFieldHits(fieldHits(blobs));
        }

        LLAMA_FN_HOST_ACC_INLINE void printFieldHits(const FieldHitsArray& hits) const
        {
#ifdef __CUDA_ARCH__
            printFieldHitsDevice(hits);
#else
            printFieldHitsHost(hits);
#endif
        }

    private:
        static constexpr auto columnWidth = 10;
        static constexpr auto sizeColumnWidth = 5;

        void printFieldHitsHost(const FieldHitsArray& hits) const
        {
            if constexpr(MyCodeHandlesProxyReferences)
                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
                          << std::setw(sizeColumnWidth) << "Size" << std::right << std::setw(columnWidth) << "Reads"
                          << ' ' << std::right << std::setw(columnWidth) << "Writes" << '\n';
            else
                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
                          << std::setw(sizeColumnWidth) << "Size" << std::right << std::setw(columnWidth)
                          << "Mlocs cmp" << '\n';
            forEachLeafCoord<RecordDim>(
                [&](auto rc)
                {
                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
                    const auto fieldSize = sizeof(GetType<RecordDim, decltype(rc)>);
                    if constexpr(MyCodeHandlesProxyReferences)
                        std::cout << std::left << std::setw(columnWidth) << prettyRecordCoord<RecordDim>(rc) << ' '
                                  << std::right << std::setw(sizeColumnWidth) << fieldSize << std::right
                                  << std::setw(columnWidth) << hits[i].reads << ' ' << std::right
                                  << std::setw(columnWidth) << hits[i].writes << '\n';
                    else
                        std::cout << std::left << std::setw(columnWidth) << prettyRecordCoord<RecordDim>(rc) << ' '
                                  << std::right << std::setw(sizeColumnWidth) << fieldSize << std::right
                                  << std::setw(columnWidth) << hits[i].memLocsComputed << '\n';
                });
            const auto total = hits.totalBytes();
            if constexpr(MyCodeHandlesProxyReferences)
            {
                const auto [rsize, runit] = prettySize(total.totalRead);
                const auto [wsize, wunit] = prettySize(total.totalWritten);
                std::cout << std::left << std::setw(columnWidth) << "Total" << ' ' << std::right
                          << std::setw(sizeColumnWidth) << ' ' << std::right << std::setw(columnWidth) << rsize
                          << runit << ' ' << std::right << std::setw(columnWidth - 2) << wsize << wunit << '\n';
            }
            else
            {
                const auto [size, unit] = prettySize(total);
                std::cout << std::left << std::setw(columnWidth) << "Total" << ' ' << std::right
                          << std::setw(sizeColumnWidth) << ' ' << std::right << std::setw(columnWidth) << size << unit
                          << '\n';
            }
            std::cout << std::internal;
        }

        LLAMA_ACC void printFieldHitsDevice(const FieldHitsArray& hits) const
        {
            if constexpr(MyCodeHandlesProxyReferences)
            {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                printf(
                    "%*s %*s %*s %*s\n",
                    columnWidth,
                    "Field",
                    sizeColumnWidth,
                    "Size",
                    columnWidth,
                    "Reads",
                    columnWidth,
                    "Writes");
            }
            else
            {
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                printf("%*s %*s %*s\n", columnWidth, "Field", sizeColumnWidth, "Size", columnWidth, "Mlocs cmp");
            }
            forEachLeafCoord<RecordDim>(
                [&](auto rc)
                {
                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
                    const auto fieldSize = sizeof(GetType<RecordDim, decltype(rc)>);
                    constexpr auto fieldName = prettyRecordCoord<RecordDim>(rc);
                    char fieldNameZT[fieldName.size() + 1]{}; // nvcc does not handle the %*.*s parameter correctly
                    llama::internal::constexprCopy(fieldName.begin(), fieldName.end(), fieldNameZT);
                    if constexpr(MyCodeHandlesProxyReferences)
                    {
                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                        printf(
                            "%*.s %*lu %*lu %*lu\n",
                            columnWidth,
                            fieldNameZT,
                            sizeColumnWidth,
                            fieldSize,
                            columnWidth,
                            static_cast<unsigned long>(hits[i].reads),
                            columnWidth,
                            static_cast<unsigned long>(hits[i].writes));
                    }
                    else
                    {
                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                        printf(
                            "%*.s %*lu %*lu\n",
                            columnWidth,
                            fieldNameZT,
                            sizeColumnWidth,
                            fieldSize,
                            columnWidth,
                            static_cast<unsigned long>(hits[i].memLocsComputed));
                    }
                });

            const auto total = hits.totalBytes();
            if constexpr(MyCodeHandlesProxyReferences)
            {
                const auto [rsize, runit] = prettySize(total.totalRead);
                const auto [wsize, wunit] = prettySize(total.totalWritten);
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                printf(
                    "%*s %*s %*f%s %*f%s\n",
                    columnWidth,
                    "Total",
                    sizeColumnWidth,
                    "",
                    columnWidth,
                    rsize,
                    runit,
                    columnWidth - 2,
                    wsize,
                    wunit);
            }
            else
            {
                const auto [size, unit] = prettySize(total);
                // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg,hicpp-vararg)
                printf("%*s %*s %*f%s\n", columnWidth, "Total", sizeColumnWidth, "", columnWidth, size, unit);
            }
        }

        LLAMA_FN_HOST_ACC_INLINE auto inner() const -> const Mapping&
        {
            return static_cast<const Mapping&>(*this);
        }
    };

    template<typename Mapping>
    inline constexpr bool isFieldAccessCount = false;

    template<typename Mapping, typename CountType, bool MyCodeHandlesProxyReferences>
    inline constexpr bool isFieldAccessCount<FieldAccessCount<Mapping, CountType, MyCodeHandlesProxyReferences>>
        = true;
} // namespace llama::mapping

#pragma once

#include "../StructName.hpp"
#include "Common.hpp"

#include <atomic>
#include <cstdio>
#include <iomanip>
#include <iostream>

#ifndef __cpp_lib_atomic_ref
#    include <boost/atomic/atomic_ref.hpp>
#endif

namespace llama::mapping
{
    namespace internal
    {
        template<typename CountType>
        LLAMA_FN_HOST_ACC_INLINE void atomicInc(CountType& i)
        {
#ifdef __CUDA_ARCH__
            // if you get an error here that there is no overload of atomicAdd, your CMAKE_CUDA_ARCHITECTURE might be
            // too low or you need to use a smaller CountType for the Trace mapping.
            atomicAdd(&i, CountType{1});
#elif defined(__cpp_lib_atomic_ref)
            ++std::atomic_ref<CountType>{i};
#else
            ++boost::atomic_ref<CountType>{i};
#endif
        }
    } // namespace internal

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
        struct TraceReference : ProxyRefOpMixin<TraceReference<Value, Ref, Count>, Value>
        {
            using value_type = Value;

            Ref r;
            AccessCounts<Count>* hits;

            LLAMA_FN_HOST_ACC_INLINE auto operator=(const TraceReference& ref) -> TraceReference&
            {
                internal::atomicInc(hits->writes);
                r = static_cast<value_type>(ref);
                return *this;
            }

            LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type value) -> TraceReference&
            {
                internal::atomicInc(hits->writes);
                r = value;
                return *this;
            }

            LLAMA_FN_HOST_ACC_INLINE operator value_type() const
            {
                internal::atomicInc(hits->reads);
                return static_cast<value_type>(r);
            }
        };
    } // namespace internal

    /// Forwards all calls to the inner mapping. Traces all accesses made through this mapping and allows printing a
    /// summary.
    /// /tparam Mapping The type of the inner mapping.
    /// /tparam CountType The type used for counting the number of accesses.
    /// /tparam MyCodeHandlesProxyReferences If false, Trace will avoid proxy references but can then only count
    /// the number of address computations
    template<typename Mapping, typename CountType = std::size_t, bool MyCodeHandlesProxyReferences = true>
    struct Trace : Mapping
    {
    private:
        using size_type = typename Mapping::ArrayExtents::value_type;

    public:
        using RecordDim = typename Mapping::RecordDim;
        using FieldHitsArray = Array<AccessCounts<CountType>, flatFieldCount<RecordDim>>;

        inline static constexpr auto blobCount = Mapping::blobCount + 1;

        constexpr Trace() = default;

        LLAMA_FN_HOST_ACC_INLINE
        explicit Trace(Mapping mapping) : Mapping(std::move(mapping))
        {
        }

        template<typename... Args>
        LLAMA_FN_HOST_ACC_INLINE explicit Trace(Args&&... innerArgs) : Mapping(std::forward<Args>(innerArgs)...)
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
                "Cannot access (even just reading) data through Trace from const blobs/view, since we need to write "
                "the access counts");

            auto& hits = fieldHits(blobs)[+flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>];
            decltype(auto) ref = mapToMemory(inner(), ai, rc, blobs); // T& or proxy reference (value)
            if constexpr(MyCodeHandlesProxyReferences)
            {
                using Value = GetType<RecordDim, decltype(rc)>;
                using Ref = decltype(ref);
                return internal::TraceReference<Value, Ref, CountType>{{}, std::forward<Ref>(ref), &hits};
            }
            else
            {
                internal::atomicInc(hits.memLocsComputed);
                return ref;
            }
        }

        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto fieldHits(const Blobs& blobs) const -> const FieldHitsArray&
        {
            return reinterpret_cast<const FieldHitsArray&>(*&blobs[blobCount - 1][0]);
        }

        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE auto fieldHits(Blobs& blobs) const -> FieldHitsArray&
        {
            return const_cast<FieldHitsArray&>(fieldHits(std::as_const(blobs)));
        }

        template<typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE void printFieldHits(const Blobs& blobs) const
        {
            printFieldHits(fieldHits(blobs));
        }

        LLAMA_FN_HOST_ACC_INLINE void printFieldHits(const FieldHitsArray& hits) const
        {
            constexpr auto columnWidth = 10;
#ifdef __CUDA_ARCH__
            if constexpr(MyCodeHandlesProxyReferences)
                printf("%*s %*s %*s\n", columnWidth, "Field", columnWidth, "Reads", columnWidth, "Writes");
            else
                printf("%*s %*s\n", columnWidth, "Field", columnWidth, "Mlocs comp");
            forEachLeafCoord<RecordDim>(
                [&](auto rc)
                {
                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
                    constexpr auto fieldName = recordCoordTags<RecordDim>(rc);
                    char fieldNameZT[fieldName.size() + 1]{}; // nvcc does not handle the %*.*s parameter correctly
                    llama::internal::constexprCopy(fieldName.begin(), fieldName.end(), fieldNameZT);
                    if constexpr(MyCodeHandlesProxyReferences)
                        printf(
                            "%*.s %*lu %*lu\n",
                            columnWidth,
                            fieldNameZT,
                            columnWidth,
                            static_cast<unsigned long>(hits[i].reads),
                            columnWidth,
                            static_cast<unsigned long>(hits[i].writes));
                    else
                        printf(
                            "%*.s %*lu %*lu\n",
                            columnWidth,
                            fieldNameZT,
                            columnWidth,
                            static_cast<unsigned long>(hits[i].memLocsComputed));
                });
#else
            if constexpr(MyCodeHandlesProxyReferences)
                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
                          << std::setw(columnWidth) << "Reads" << ' ' << std::right << std::setw(columnWidth)
                          << "Writes" << '\n';
            else
                std::cout << std::left << std::setw(columnWidth) << "Field" << ' ' << std::right
                          << std::setw(columnWidth) << "Mlocs comp" << '\n';
            forEachLeafCoord<RecordDim>(
                [&](auto rc)
                {
                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
                    if constexpr(MyCodeHandlesProxyReferences)
                        std::cout << std::left << std::setw(columnWidth) << recordCoordTags<RecordDim>(rc) << ' '
                                  << std::right << std::setw(columnWidth) << hits[i].reads << ' ' << std::right
                                  << std::setw(columnWidth) << hits[i].writes << '\n';
                    else
                        std::cout << std::left << std::setw(columnWidth) << recordCoordTags<RecordDim>(rc) << ' '
                                  << std::right << std::setw(columnWidth) << hits[i].memLocsComputed << '\n';
                });
            std::cout << std::internal;
#endif
        }

    private:
        LLAMA_FN_HOST_ACC_INLINE auto inner() const -> const Mapping&
        {
            return static_cast<const Mapping&>(*this);
        }
    };

    template<typename Mapping>
    inline constexpr bool isTrace = false;

    template<typename Mapping, typename CountType, bool MyCodeHandlesProxyReferences>
    inline constexpr bool isTrace<Trace<Mapping, CountType, MyCodeHandlesProxyReferences>> = true;
} // namespace llama::mapping

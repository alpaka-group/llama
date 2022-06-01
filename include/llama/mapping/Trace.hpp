#pragma once

#include "Common.hpp"

#include <atomic>
#include <cstdio>
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
            auto& hits = fieldHits(blobs)[+flatRecordCoord<RecordDim, RecordCoord<RecordCoords...>>];
            auto&& ref = mapToMemory(inner(), ai, rc, blobs);
            if constexpr(MyCodeHandlesProxyReferences)
            {
                using Ref = decltype(mapToMemory(inner(), ai, rc,
                                                 blobs)); // T& or proxy reference
                using VT = GetType<RecordDim, decltype(rc)>;
                struct Reference : ProxyRefOpMixin<Reference, VT>
                {
                    using value_type = VT;

                    Ref r;
                    AccessCounts<CountType>* hits;

                    LLAMA_FN_HOST_ACC_INLINE auto operator=(value_type t) -> Reference&
                    {
                        internal::atomicInc(hits->writes);
                        r = t;
                        return *this;
                    }

                    LLAMA_FN_HOST_ACC_INLINE operator value_type() const
                    {
                        internal::atomicInc(hits->reads);
                        return static_cast<value_type>(r);
                    }
                };
                return Reference{{}, std::forward<Ref>(ref), &hits};
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
#ifdef __CUDA_ARCH__
            if constexpr(MyCodeHandlesProxyReferences)
                printf("Trace mapping, number of accesses:\n");
            else
                printf("Trace mapping, number of memory locations computed:\n");

            for(int i = 0; i < hits.size(); i++)
                if constexpr(MyCodeHandlesProxyReferences)
                    printf(
                        "\t%i:\tR: %lu\tW: %lu\n",
                        i,
                        static_cast<unsigned long>(hits[i].reads),
                        static_cast<unsigned long>(hits[i].writes));
                else
                    printf("\t%i:\t%lu\n", i, static_cast<unsigned long>(hits[i].memLocsComputed));
#else
            if constexpr(MyCodeHandlesProxyReferences)
                std::cout << "Trace mapping, number of accesses:\n";
            else
                std::cout << "Trace mapping, number of memory locations computed:\n";
            forEachLeafCoord<RecordDim>(
                [&](auto rc)
                {
                    const size_type i = flatRecordCoord<RecordDim, decltype(rc)>;
                    if constexpr(MyCodeHandlesProxyReferences)
                        std::cout << '\t' << recordCoordTags<RecordDim>(rc) << ":\tR: " << hits[i].reads
                                  << "\tW: " << hits[i].writes << '\n';
                    else
                        std::cout << '\t' << recordCoordTags<RecordDim>(rc) << ":\t " << hits[i].memLocsComputed
                                  << '\n';
                });
#endif
        }

    private:
        LLAMA_FN_HOST_ACC_INLINE auto inner() const -> const Mapping&
        {
            return static_cast<const Mapping&>(*this);
        }
    };
} // namespace llama::mapping

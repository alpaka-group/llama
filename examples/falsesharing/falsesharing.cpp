// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

// This example shows how a LLAMA mapping can be used to detect false sharing. The initial idea came from discussion
// with Andreas Knüpfer after a presentation of LLAMA at the ZIH PhD student retreat.

#include <iostream>
#include <llama/llama.hpp>
#include <mutex>
#include <new>
#include <thread>

#if defined(__cpp_lib_jthread) && defined(__cpp_lib_hardware_interference_size)
template<typename Mapping, typename THID = std::thread::id>
struct DetectFalseSharing : private Mapping
{
    static_assert(
        !llama::hasAnyComputedField<Mapping>,
        "Detecting flase sharing for computed mappings is not implemented.");

private:
    using size_type = typename Mapping::ArrayExtents::value_type;
    using ArrayIndex = typename Mapping::ArrayExtents::Index;

public:
    using Inner = Mapping;
    using ArrayExtents = typename Mapping::ArrayExtents;
    using RecordDim = typename Mapping::RecordDim;

    inline static constexpr size_type clSize = std::hardware_destructive_interference_size;

    // We duplicate every blob of the inner mapping with a shadow blob, where we count the accesses
    inline static constexpr std::size_t blobCount = Mapping::blobCount * 2;

    constexpr DetectFalseSharing() = default;

    LLAMA_FN_HOST_ACC_INLINE
    explicit DetectFalseSharing(Mapping mapping) : Mapping(std::move(mapping))
    {
    }

    template<typename... Args>
    LLAMA_FN_HOST_ACC_INLINE explicit DetectFalseSharing(Args&&... innerArgs)
        : Mapping(std::forward<Args>(innerArgs)...)
    {
    }

    using Mapping::extents;

    LLAMA_FN_HOST_ACC_INLINE
    constexpr auto blobSize(size_type blobIndex) const -> size_type
    {
        if(blobIndex < size_type{Mapping::blobCount})
            return Mapping::blobSize(blobIndex);
        return thidBlobSize(blobIndex - size_type{Mapping::blobCount}) * sizeof(THID);
    }

    template<std::size_t... RecordCoords>
    static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
    {
        return true;
    }

    template<std::size_t... RecordCoords, typename Blobs>
    LLAMA_FN_HOST_ACC_INLINE auto compute(ArrayIndex ai, llama::RecordCoord<RecordCoords...> rc, Blobs& blobs) const
        -> decltype(auto)
    {
        static_assert(
            !std::is_const_v<Blobs>,
            "Cannot access (even just reading) data through DetectFalseSharing from const blobs/view, since we need "
            "to write "
            "the thread ids");

        const auto [nr, offset] = Mapping::blobNrAndOffset(ai, rc);
        using Type = llama::GetType<typename Mapping::RecordDim, llama::RecordCoord<RecordCoords...>>;

        auto* thids = thidBlobPtr(nr, blobs);
        for(size_type i = 0; i < llama::divCeil(size_type{sizeof(Type)}, clSize); i++)
        {
            static std::mutex m; // TODO(bgruber): make this more efficient
            const std::lock_guard _{m};
            auto& prevThid = thids[offset / clSize + i];
            const auto thisThid = std::this_thread::get_id();
            if(prevThid == std::thread::id{})
                prevThid = thisThid;
            else if(prevThid != thisThid)
                std::cout << "False sharing: CL @ blob " << nr << " offset " << offset << " owned by " << prevThid
                          << " accessed by " << thisThid << '\n';
        }

        LLAMA_BEGIN_SUPPRESS_HOST_DEVICE_WARNING
        return reinterpret_cast<llama::CopyConst<std::remove_reference_t<decltype(blobs[nr][offset])>, Type>&>(
            blobs[nr][offset]);
        LLAMA_END_SUPPRESS_HOST_DEVICE_WARNING
    }

    // Returns the size of the block hits buffer for blob forBlobI in block counts.
    LLAMA_FN_HOST_ACC_INLINE auto thidBlobSize(size_type forBlobI) const -> size_type
    {
        assert(forBlobI < Mapping::blobCount);
        return llama::divCeil(Mapping::blobSize(forBlobI), clSize);
    }

    LLAMA_SUPPRESS_HOST_DEVICE_WARNING
    template<typename Blobs>
    LLAMA_FN_HOST_ACC_INLINE auto thidBlobPtr(size_type forBlobI, Blobs& blobs) const -> llama::CopyConst<Blobs, THID>*
    {
        return reinterpret_cast<llama::CopyConst<Blobs, THID>*>(&blobs[size_type{Mapping::blobCount} + forBlobI][0]);
    }

    template<typename Blobs>
    auto thidBlob(size_type forBlobI, Blobs& blobs) const -> std::span<llama::CopyConst<Blobs, THID>>
    {
        return {thidBlobPtr(forBlobI, blobs), static_cast<std::size_t>(thidBlobSize(forBlobI))};
    }

    template<typename Blobs>
    void resetThids(Blobs& blobs) const
    {
        for(size_type b = 0; b < Mapping::blobCount; b++)
            for(auto& thid : thidBlob(b, blobs))
                thid = std::thread::id{};
    }
};

void resetThids(auto& view)
{
    view.mapping().resetThids(view.blobs());
}

auto main() -> int
{
    const auto n = 1000;
    const auto mapping = DetectFalseSharing<llama::mapping::AoS<llama::ArrayExtents<int, llama::dyn>, double>>{
        llama::ArrayExtents{n}};

    std::cout << "Current thread id: " << std::this_thread::get_id() << "\n";

    auto view = llama::allocViewUninitialized(mapping);

    resetThids(view);
    std::cout << "Access by one thread:\n";
    for(int i = 0; i < n; i++)
        view[i] = 2;

    resetThids(view);
    std::cout << "Access by two threads half/half:\n";
    {
        auto t1 = std::jthread{[&]
                               {
                                   for(int i = 0; i < n / 2; i++)
                                       view[i] = 2;
                               }};
        auto t2 = std::jthread{[&]
                               {
                                   for(int i = n / 2; i < n; i++)
                                       view[i] = 2;
                               }};
    }

    resetThids(view);
    std::cout << "Access by two threads alternating cachelines:\n";
    constexpr auto clItems = std::hardware_destructive_interference_size / sizeof(double);
    {
        auto t1 = std::jthread{[&]
                               {
                                   for(int i = 0; i < n; i += clItems * 2)
                                       view[i] = 2;
                               }};
        auto t2 = std::jthread{[&]
                               {
                                   for(int i = clItems; i < n; i += clItems * 2)
                                       view[i] = 2;
                               }};
    }

    resetThids(view);
    std::cout << "Access by two threads alternating elements:\n";
    {
        auto t1 = std::jthread{[&]
                               {
                                   for(int i = 0; i < 10; i += 2)
                                       view[i] = 2;
                               }};
        auto t2 = std::jthread{[&]
                               {
                                   for(int i = 1; i < 10; i += 2)
                                       view[i] = 2;
                               }};
    }

    return 0;
}

#else
auto main() -> int
{
    std::cout << "Required C++ features not supported\n";
}
#endif

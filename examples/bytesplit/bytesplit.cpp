#include <cstdint>
#include <fmt/core.h>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct A{};
    struct B{};
    struct C{};
    struct D{};
    struct E{};
    struct F{};
} // namespace tag

using Data = llama::Record<
    llama::Field<tag::A, std::uint16_t>,
    llama::Field<tag::B, std::int32_t>,
    llama::Field<tag::C, std::uint64_t>,
    llama::Field<tag::D, float>,
    llama::Field<tag::E, double>,
    llama::Field<tag::F, unsigned char>
>;
// clang-format on

template<typename T>
using ReplaceByByteArray = std::byte[sizeof(T)];

template<typename RecordDim>
using SplitBytes = llama::TransformLeaves<RecordDim, ReplaceByByteArray>;

template<typename TArrayExtents, typename TRecordDim>
struct BytesplitSoA : private llama::mapping::SoA<TArrayExtents, SplitBytes<TRecordDim>, false>
{
    using Base = llama::mapping::SoA<TArrayExtents, SplitBytes<TRecordDim>, false>;

    using ArrayExtents = typename Base::ArrayExtents;
    using ArrayIndex = typename Base::ArrayIndex;
    using RecordDim = TRecordDim; // hide Base::RecordDim
    using Base::blobCount;

    using Base::Base;
    using Base::blobSize;
    using Base::extents;

    LLAMA_FN_HOST_ACC_INLINE
    constexpr explicit BytesplitSoA(TArrayExtents extents, TRecordDim = {}) : Base(extents)
    {
    }

    template<std::size_t... RecordCoords>
    static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
    {
        return true;
    }

    template<typename QualifiedBase, typename RC, typename BlobArray>
    struct Reference
    {
        QualifiedBase& innerMapping;
        ArrayIndex ai;
        BlobArray& blobs;

        using DstType = llama::GetType<TRecordDim, RC>;

        // NOLINTNEXTLINE(google-explicit-constructor,hicpp-explicit-conversions)
        operator DstType() const
        {
            DstType v;
            auto* p = reinterpret_cast<std::byte*>(&v);
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(DstType)>>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    const auto [nr, off] = innerMapping.blobNrAndOffset(ai, llama::Cat<RC, llama::RecordCoord<i>>{});
                    p[i] = blobs[nr][off];
                });
            return v;
        }

        auto operator=(DstType v) -> Reference&
        {
            auto* p = reinterpret_cast<std::byte*>(&v);
            boost::mp11::mp_for_each<boost::mp11::mp_iota_c<sizeof(DstType)>>(
                [&](auto ic)
                {
                    constexpr auto i = decltype(ic)::value;
                    const auto [nr, off] = innerMapping.blobNrAndOffset(ai, llama::Cat<RC, llama::RecordCoord<i>>{});
                    blobs[nr][off] = p[i];
                });
            return *this;
        }
    };

    template<std::size_t... RecordCoords, typename BlobArray>
    LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(
        typename Base::ArrayIndex ai,
        llama::RecordCoord<RecordCoords...>,
        BlobArray& blobs) const
    {
        return Reference<decltype(*this), llama::RecordCoord<RecordCoords...>, BlobArray>{*this, ai, blobs};
    }
};

auto main() -> int
{
    constexpr auto N = 128;
    using ArrayExtents = llama::ArrayExtentsDynamic<1>;
    const auto mapping = BytesplitSoA<ArrayExtents, Data>{{N}};

    auto view = llama::allocView(mapping);

    int value = 0;
    for(std::size_t i = 0; i < N; i++)
        llama::forEachLeafCoord<Data>([&](auto rc) { view(i)(rc) = ++value; });

    value = 0;
    for(std::size_t i = 0; i < N; i++)
        llama::forEachLeafCoord<Data>(
            [&](auto rc)
            {
                using T = llama::GetType<Data, decltype(rc)>;
                ++value;
                if(view(i)(rc) != static_cast<T>(value))
                    fmt::print("Error: value after store is corrupt. {} != {}\n", view(i)(rc), value);
            });

    // extract into a view of unsplit fields
    auto viewExtracted = llama::allocViewUninitialized(llama::mapping::AoS<ArrayExtents, Data>{{N}});
    llama::copy(view, viewExtracted);
    if(!std::equal(view.begin(), view.end(), viewExtracted.begin(), viewExtracted.end()))
        fmt::print("ERROR: unsplit view is different\n");

    // compute something on the extracted view
    for(std::size_t i = 0; i < N; i++)
        viewExtracted(i) *= 2;

    // rearrange back into split view
    llama::copy(viewExtracted, view);

    value = 0;
    for(std::size_t i = 0; i < N; i++)
        llama::forEachLeafCoord<Data>(
            [&](auto rc)
            {
                using T = llama::GetType<Data, decltype(rc)>;
                ++value;
                if(view(i)(rc) != static_cast<T>(static_cast<T>(value) * 2))
                    fmt::print("Error: value after resplit is corrupt. {} != {}\n", view(i)(rc), value);
            });

    // compute something on the split view
    for(std::size_t i = 0; i < N; i++)
        view(i) = view(i) * 2; // cannot do view(i) *= 2; with proxy references

    value = 0;
    for(std::size_t i = 0; i < N; i++)
        llama::forEachLeafCoord<Data>(
            [&](auto rc)
            {
                using T = llama::GetType<Data, decltype(rc)>;
                ++value;
                if(view(i)(rc) != static_cast<T>(static_cast<T>(value) * 4))
                    fmt::print(
                        "Error: value after computation on split data is corrupt. {} != {}\n",
                        view(i)(rc),
                        value);
            });

    fmt::print("Done\n");
}

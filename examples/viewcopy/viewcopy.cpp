#include <boost/functional/hash.hpp>
#include <boost/mp11.hpp>
#include <chrono>
#include <llama/llama.hpp>
#include <numeric>
#include <string_view>

// clang-format off
namespace tag
{
    struct Pos{};
    struct Vel{};
    struct X{};
    struct Y{};
    struct Z{};
    struct Mass{};
}

using Particle = llama::DS<
    llama::DE<tag::Pos, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>,
        llama::DE<tag::Z, float>
    >>,
    llama::DE<tag::Vel, llama::DS<
        llama::DE<tag::X, float>,
        llama::DE<tag::Y, float>,
        llama::DE<tag::Z, float>
    >>,
    llama::DE<tag::Mass, float>
>;
// clang-format on

template <typename Mapping1, typename BlobType1, typename Mapping2, typename BlobType2>
void naive_copy(const llama::View<Mapping1, BlobType1>& srcView, llama::View<Mapping2, BlobType2>& dstView)
{
    static_assert(std::is_same_v<typename Mapping1::DatumDomain, typename Mapping2::DatumDomain>);

    if (srcView.mapping.arrayDomainSize != dstView.mapping.arrayDomainSize)
        throw std::runtime_error{"UserDomain sizes are different"};

    for (auto ad : llama::ArrayDomainIndexRange{srcView.mapping.arrayDomainSize})
        llama::forEach<typename Mapping1::DatumDomain>([&](auto coord) {
            dstView(ad)(coord) = srcView(ad)(coord);
            // std::memcpy(
            //    &dstView(ad)(coord),
            //    &srcView(ad)(coord),
            //    sizeof(llama::GetType<typename Mapping1::DatumDomain, decltype(coord)>));
        });
}

template <
    bool ReadOpt,
    typename ArrayDomain,
    typename DatumDomain,
    std::size_t LanesSrc,
    typename BlobType1,
    std::size_t LanesDst,
    typename BlobType2>
void aosoa_copy(
    const llama::View<
        llama::mapping::AoSoA<ArrayDomain, DatumDomain, LanesSrc, llama::mapping::LinearizeArrayDomainCpp>,
        BlobType1>& srcView,
    llama::View<
        llama::mapping::AoSoA<ArrayDomain, DatumDomain, LanesDst, llama::mapping::LinearizeArrayDomainCpp>,
        BlobType2>& dstView)
{
    static_assert(srcView.storageBlobs.rank == 1);
    static_assert(dstView.storageBlobs.rank == 1);

    if (srcView.mapping.arrayDomainSize != dstView.mapping.arrayDomainSize)
        throw std::runtime_error{"UserDomain sizes are different"};

    const auto flatSize = std::reduce(
        std::begin(dstView.mapping.arrayDomainSize),
        std::end(dstView.mapping.arrayDomainSize),
        std::size_t{1},
        std::multiplies<>{});

    const std::byte* src = srcView.storageBlobs[0].data();
    std::byte* dst = dstView.storageBlobs[0].data();

    // the same as AoSoA::getBlobNrAndOffset but takes a flat array index
    auto map = [](std::size_t flatArrayIndex, auto coord, std::size_t Lanes) {
        const auto blockIndex = flatArrayIndex / Lanes;
        const auto laneIndex = flatArrayIndex % Lanes;
        const auto offset = (llama::sizeOf<DatumDomain> * Lanes) * blockIndex
            + llama::offsetOf<DatumDomain, decltype(coord)> * Lanes
            + sizeof(llama::GetType<DatumDomain, decltype(coord)>) * laneIndex;
        return offset;
    };

    if constexpr (ReadOpt)
    {
        // optimized for linear reading
        for (std::size_t i = 0; i < flatSize; i += LanesSrc)
        {
            llama::forEach<DatumDomain>([&](auto coord) {
                constexpr auto L = std::min(LanesSrc, LanesDst);
                for (std::size_t j = 0; j < LanesSrc; j += L)
                {
                    constexpr auto bytes = L * sizeof(llama::GetType<DatumDomain, decltype(coord)>);
                    std::memcpy(&dst[map(i + j, coord, LanesDst)], src, bytes);
                    src += bytes;
                }
            });
        }
    }
    else
    {
        // optimized for linear writing
        for (std::size_t i = 0; i < flatSize; i += LanesDst)
        {
            llama::forEach<DatumDomain>([&](auto coord) {
                constexpr auto L = std::min(LanesSrc, LanesDst);
                for (std::size_t j = 0; j < LanesDst; j += L)
                {
                    constexpr auto bytes = L * sizeof(llama::GetType<DatumDomain, decltype(coord)>);
                    std::memcpy(dst, &src[map(i + j, coord, LanesSrc)], bytes);
                    dst += bytes;
                }
            });
        }
    }
}

template <typename Mapping, typename BlobType>
auto hash(const llama::View<Mapping, BlobType>& view)
{
    std::size_t acc = 0;
    for (auto ad : llama::ArrayDomainIndexRange{view.mapping.arrayDomainSize})
        llama::forEach<Particle>([&](auto coord) { boost::hash_combine(acc, view(ad)(coord)); });
    return acc;
}
template <typename Mapping>
auto prepareViewAndHash(Mapping mapping)
{
    auto view = llama::allocView(mapping);

    auto value = 0.0f;
    for (auto ad : llama::ArrayDomainIndexRange{mapping.arrayDomainSize})
    {
        auto p = view(ad);
        p(tag::Pos{}, tag::X{}) = value++;
        p(tag::Pos{}, tag::Y{}) = value++;
        p(tag::Pos{}, tag::Z{}) = value++;
        p(tag::Vel{}, tag::X{}) = value++;
        p(tag::Vel{}, tag::Y{}) = value++;
        p(tag::Vel{}, tag::Z{}) = value++;
        p(tag::Mass{}) = value++;
    }

    const auto checkSum = hash(view);
    return std::tuple{view, checkSum};
}

template <typename SrcView, typename DstMapping, typename F>
void benchmarkCopy(std::string_view name, const SrcView& srcView, std::size_t srcHash, DstMapping dstMapping, F copy)
{
    auto dstView = llama::allocView(dstMapping);
    const auto start = std::chrono::high_resolution_clock::now();
    copy(srcView, dstView);
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto dstHash = hash(dstView);

    std::cout << name << "\ttook " << std::chrono::duration<double>(stop - start).count() << "s\thash "
              << (srcHash == dstHash ? "good" : "BAD ") << "\n";
}

int main(int argc, char** argv)
{
    const auto userDomain = llama::ArrayDomain{1024, 1024, 16};

    {
        std::cout << "AoS -> SoA\n";
        const auto srcMapping = llama::mapping::AoS{userDomain, Particle{}};
        const auto dstMapping = llama::mapping::SoA{userDomain, Particle{}};

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);
        benchmarkCopy("naive_copy", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            naive_copy(srcView, dstView);
        });
        benchmarkCopy("memcpy    ", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            static_assert(srcView.storageBlobs.rank == 1);
            static_assert(dstView.storageBlobs.rank == 1);
            std::memcpy(dstView.storageBlobs[0].data(), srcView.storageBlobs[0].data(), dstView.storageBlobs[0].size());
        });
    }

    {
        std::cout << "SoA -> AoS\n";
        const auto srcMapping = llama::mapping::SoA{userDomain, Particle{}};
        const auto dstMapping = llama::mapping::AoS{userDomain, Particle{}};

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);
        benchmarkCopy("naive_copy", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            naive_copy(srcView, dstView);
        });
        benchmarkCopy("memcpy    ", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            static_assert(srcView.storageBlobs.rank == 1);
            static_assert(dstView.storageBlobs.rank == 1);
            std::memcpy(dstView.storageBlobs[0].data(), srcView.storageBlobs[0].data(), dstView.storageBlobs[0].size());
        });
    }

    using namespace boost::mp11;
    mp_for_each<mp_list<
        mp_list_c<std::size_t, 8, 32>,
        mp_list_c<std::size_t, 8, 64>,
        mp_list_c<std::size_t, 32, 8>,
        mp_list_c<std::size_t, 64, 8>>>([&](auto pair) {
        constexpr auto LanesSrc = mp_first<decltype(pair)>::value;
        constexpr auto LanesDst = mp_second<decltype(pair)>::value;

        std::cout << "AoSoA" << LanesSrc << " -> AoSoA" << LanesDst << "\n"; // e.g. AVX2 -> CUDA
        const auto srcMapping = llama::mapping::AoSoA<decltype(userDomain), Particle, LanesSrc>{userDomain};
        const auto dstMapping = llama::mapping::AoSoA<decltype(userDomain), Particle, LanesDst>{userDomain};

        auto [srcView, srcHash] = prepareViewAndHash(srcMapping);
        benchmarkCopy("naive_copy   ", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            naive_copy(srcView, dstView);
        });
        benchmarkCopy("memcpy       ", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            static_assert(srcView.storageBlobs.rank == 1);
            static_assert(dstView.storageBlobs.rank == 1);
            std::memcpy(dstView.storageBlobs[0].data(), srcView.storageBlobs[0].data(), dstView.storageBlobs[0].size());
        });
        benchmarkCopy("aosoa_copy(r)", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            aosoa_copy<true>(srcView, dstView);
        });
        benchmarkCopy("aosoa_copy(w)", srcView, srcHash, dstMapping, [](const auto& srcView, auto& dstView) {
            aosoa_copy<false>(srcView, dstView);
        });
    });
}

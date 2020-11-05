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
        llama::forEach<Mapping1::DatumDomain>([&](auto coord) {
            dstView(ad)(coord) = srcView(ad)(coord);
            // std::memcpy(
            //    &dstView(ad)(coord),
            //    &srcView(ad)(coord),
            //    sizeof(llama::GetType<typename Mapping1::DatumDomain, decltype(coord)>));
        });
}

template <typename Mapping, typename BlobType>
auto sumAllValues(const llama::View<Mapping, BlobType>& view)
{
    auto sum = 0.0;
    for (auto ad : llama::ArrayDomainIndexRange{view.mapping.arrayDomainSize})
        llama::forEach<Particle>([&](auto coord) { sum += view(ad)(coord); });
    return sum;
}
template <typename Mapping>
auto prepareViewAndChecksum(Mapping mapping)
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

    const auto checkSum = sumAllValues(view);
    return std::tuple{view, checkSum};
}

template <typename SrcView, typename DstMapping, typename F>
void benchmarkCopy(std::string_view name, const SrcView& srcView, double srcCheckSum, DstMapping dstMapping, F copy)
{
    auto dstView = llama::allocView(dstMapping);
    const auto start = std::chrono::high_resolution_clock::now();
    copy(srcView, dstView);
    const auto stop = std::chrono::high_resolution_clock::now();
    const auto dstCheckSum = sumAllValues(dstView);

    std::cout << name << "\ttook " << std::chrono::duration<double>(stop - start).count() << "s\tchecksum "
              << (srcCheckSum == dstCheckSum ? "good" : "BAD ") << "\n";
}

int main(int argc, char** argv)
{
    const auto userDomain = llama::ArrayDomain{1024, 1024, 16};

    {
        std::cout << "AoS -> SoA\n";
        const auto srcMapping = llama::mapping::AoS{userDomain, Particle{}};
        const auto dstMapping = llama::mapping::SoA{userDomain, Particle{}};

        auto [srcView, srcCheckSum] = prepareViewAndChecksum(srcMapping);
        benchmarkCopy("naive_copy", srcView, srcCheckSum, dstMapping, [](const auto& view1, auto& view2) {
            naive_copy(view1, view2);
        });
        benchmarkCopy("memcpy    ", srcView, srcCheckSum, dstMapping, [](const auto& view1, auto& view2) {
            static_assert(view1.storageBlobs.rank == 1);
            static_assert(view2.storageBlobs.rank == 1);
            std::memcpy(view2.storageBlobs[0].data(), view1.storageBlobs[0].data(), view2.storageBlobs[0].size());
        });
    }
}

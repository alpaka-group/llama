#include <chrono>
#include <llama/llama.hpp>
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

    if (srcView.mapping.userDomainSize != dstView.mapping.userDomainSize)
        throw std::runtime_error{"UserDomain sizes are different"};

    for (auto ud : llama::UserDomainCoordRange{srcView.mapping.userDomainSize})
        llama::forEach<Mapping1::DatumDomain>([&](auto, auto inner) { dstView(ud)(inner) = srcView(ud)(inner); });
}

template <typename Mapping, typename BlobType>
auto sumAllValues(const llama::View<Mapping, BlobType>& view)
{
    auto sum = 0.0;
    for (auto ud : llama::UserDomainCoordRange{view.mapping.userDomainSize})
        llama::forEach<Particle>([&](auto, auto inner) { sum += view(ud)(inner); });
    return sum;
}

template <typename Mapping1, typename Mapping2, typename F>
void benchmarkCopy(std::string_view name, Mapping1 mapping1, Mapping2 mapping2, F copy)
{
    auto view1 = llama::allocView(mapping1);
    auto view2 = llama::allocView(mapping2);

    auto value = 0.0f;
    for (auto ud : llama::UserDomainCoordRange{mapping1.userDomainSize})
    {
        auto p = view1(ud);
        p(tag::Pos{}, tag::X{}) = value++;
        p(tag::Pos{}, tag::Y{}) = value++;
        p(tag::Pos{}, tag::Z{}) = value++;
        p(tag::Vel{}, tag::X{}) = value++;
        p(tag::Vel{}, tag::Y{}) = value++;
        p(tag::Vel{}, tag::Z{}) = value++;
        p(tag::Mass{}) = value++;
    }

    const auto checkSum1 = sumAllValues(view1);

    const auto start = std::chrono::high_resolution_clock::now();
    copy(view1, view2);
    const auto stop = std::chrono::high_resolution_clock::now();

    const auto checkSum2 = sumAllValues(view2);

    std::cout << name << "\tchecksum " << (checkSum1 == checkSum2 ? "good" : "BAD ") << "\ttook "
              << std::chrono::duration<double>(stop - start).count() << "s\n";
}

int main(int argc, char** argv)
{
    const auto userDomain = llama::UserDomain{1024, 4096};
    const auto mapping1 = llama::mapping::AoS{userDomain, Particle{}};
    const auto mapping2 = llama::mapping::SoA{userDomain, Particle{}};

    benchmarkCopy("naive_copy", mapping1, mapping2, [](const auto& view1, auto& view2) { naive_copy(view1, view2); });
}

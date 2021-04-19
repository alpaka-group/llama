#include <algorithm>
#include <array>
#include <cstring>
#include <fmt/core.h>
#include <fstream>
#include <llama/DumpMapping.hpp>
#include <llama/llama.hpp>

// clang-format off
namespace tag
{
    struct X{};
    struct Y{};
    struct Z{};
} // namespace tag

using Vector = llama::DS<
    llama::DE<tag::X, int>,
    llama::DE<tag::Y, int>,
    llama::DE<tag::Z, int>
>;
// clang-format on

template <template <typename, typename> typename InnerMapping, typename T_ArrayDomain, typename T_DatumDomain>
struct GuardMapping2D
{
    static_assert(std::is_same_v<T_ArrayDomain, llama::ArrayDomain<2>>, "Only 2D arrays are implemented");

    using ArrayDomain = T_ArrayDomain;
    using DatumDomain = T_DatumDomain;

    constexpr GuardMapping2D() = default;

    constexpr explicit GuardMapping2D(ArrayDomain size, DatumDomain = {})
        : arrayDomainSize(size)
        , left({size[0] - 2})
        , right({size[0] - 2})
        , top({size[1] - 2})
        , bot({size[1] - 2})
        , center({size[0] - 2, size[1] - 2})
    {
    }

    constexpr auto blobSize(std::size_t i) const -> std::size_t
    {
        if (i >= centerOff)
            return center.blobSize(i - centerOff);
        if (i >= botOff)
            return bot.blobSize(i - botOff);
        if (i >= topOff)
            return top.blobSize(i - topOff);
        if (i >= rightOff)
            return right.blobSize(i - rightOff);
        if (i >= leftOff)
            return left.blobSize(i - leftOff);
        if (i >= rightBotOff)
            return rightBot.blobSize(i - rightBotOff);
        if (i >= rightTopOff)
            return rightTop.blobSize(i - rightTopOff);
        if (i >= leftBotOff)
            return leftBot.blobSize(i - leftBotOff);
        if (i >= leftTopOff)
            return leftTop.blobSize(i - leftTopOff);
        std::abort();
    }

    template <std::size_t... DatumDomainCoord>
    constexpr auto blobNrAndOffset(ArrayDomain coord) const -> llama::NrAndOffset
    {
        // [0][0] is at left top
        const auto [row, col] = coord;
        const auto [rowMax, colMax] = arrayDomainSize;

        if (col == 0)
        {
            if (row == 0)
                return offsetBlobNr(leftTop.template blobNrAndOffset<DatumDomainCoord...>({}), leftTopOff);
            if (row == rowMax - 1)
                return offsetBlobNr(leftBot.template blobNrAndOffset<DatumDomainCoord...>({}), leftBotOff);
            return offsetBlobNr(left.template blobNrAndOffset<DatumDomainCoord...>({row - 1}), leftOff);
        }
        if (col == colMax - 1)
        {
            if (row == 0)
                return offsetBlobNr(rightTop.template blobNrAndOffset<DatumDomainCoord...>({}), rightTopOff);
            if (row == rowMax - 1)
                return offsetBlobNr(rightBot.template blobNrAndOffset<DatumDomainCoord...>({}), rightBotOff);
            return offsetBlobNr(right.template blobNrAndOffset<DatumDomainCoord...>({row - 1}), rightOff);
        }
        if (row == 0)
            return offsetBlobNr(top.template blobNrAndOffset<DatumDomainCoord...>({col - 1}), topOff);
        if (row == rowMax - 1)
            return offsetBlobNr(bot.template blobNrAndOffset<DatumDomainCoord...>({col - 1}), botOff);
        return offsetBlobNr(center.template blobNrAndOffset<DatumDomainCoord...>({row - 1, col - 1}), centerOff);
    }

    constexpr auto centerBlobs() const
    {
        return blobIndices(center, centerOff);
    }

    constexpr auto leftTopBlobs() const
    {
        return blobIndices(leftTop, leftTopOff);
    }

    constexpr auto leftBotBlobs() const
    {
        return blobIndices(leftBot, leftBotOff);
    }

    constexpr auto leftBlobs() const
    {
        return blobIndices(left, leftOff);
    }

    constexpr auto rightTopBlobs() const
    {
        return blobIndices(rightTop, rightTopOff);
    }

    constexpr auto rightBotBlobs() const
    {
        return blobIndices(rightBot, rightBotOff);
    }

    constexpr auto rightBlobs() const
    {
        return blobIndices(right, rightOff);
    }

    constexpr auto topBlobs() const
    {
        return blobIndices(top, topOff);
    }

    constexpr auto botBlobs() const
    {
        return blobIndices(bot, botOff);
    }

private:
    constexpr auto offsetBlobNr(llama::NrAndOffset nao, std::size_t blobNrOffset) const -> llama::NrAndOffset
    {
        nao.nr += blobNrOffset;
        return nao;
    }

    template <typename Mapping>
    constexpr auto blobIndices(const Mapping&, std::size_t offset) const
    {
        std::array<std::size_t, Mapping::blobCount> a{};
        std::generate(begin(a), end(a), [i = offset]() mutable { return i++; });
        return a;
    }

    llama::mapping::One<ArrayDomain, DatumDomain> leftTop;
    llama::mapping::One<ArrayDomain, DatumDomain> leftBot;
    llama::mapping::One<ArrayDomain, DatumDomain> rightTop;
    llama::mapping::One<ArrayDomain, DatumDomain> rightBot;
    InnerMapping<llama::ArrayDomain<1>, DatumDomain> left;
    InnerMapping<llama::ArrayDomain<1>, DatumDomain> right;
    InnerMapping<llama::ArrayDomain<1>, DatumDomain> top;
    InnerMapping<llama::ArrayDomain<1>, DatumDomain> bot;
    InnerMapping<llama::ArrayDomain<2>, DatumDomain> center;

    static constexpr auto leftTopOff = std::size_t{0};
    static constexpr auto leftBotOff = leftTopOff + decltype(leftTop)::blobCount;
    static constexpr auto rightTopOff = leftBotOff + decltype(leftBot)::blobCount;
    static constexpr auto rightBotOff = rightTopOff + decltype(rightTop)::blobCount;
    static constexpr auto leftOff = rightBotOff + decltype(rightBot)::blobCount;
    static constexpr auto rightOff = leftOff + decltype(left)::blobCount;
    static constexpr auto topOff = rightOff + decltype(right)::blobCount;
    static constexpr auto botOff = topOff + decltype(top)::blobCount;
    static constexpr auto centerOff = botOff + decltype(bot)::blobCount;

public:
    static constexpr auto blobCount = centerOff + decltype(center)::blobCount;

    ArrayDomain arrayDomainSize;
};

template <typename View>
void printView(const View& view, std::size_t rows, std::size_t cols)
{
    for (std::size_t row = 0; row < rows; row++)
    {
        for (std::size_t col = 0; col < cols; col++)
            std::cout << fmt::format(
                "[{:3},{:3},{:3}]",
                view(row, col)(tag::X{}),
                view(row, col)(tag::Y{}),
                view(row, col)(tag::Z{}));
        std::cout << '\n';
    }
}

template <template <typename, typename> typename Mapping>
void run(const std::string& mappingName)
{
    std::cout << "\n===== Mapping " << mappingName << " =====\n\n";

    constexpr auto rows = 7;
    constexpr auto cols = 5;
    const auto arrayDomain = llama::ArrayDomain{rows, cols};
    const auto mapping = GuardMapping2D<Mapping, llama::ArrayDomain<2>, Vector>{arrayDomain};
    std::ofstream{"bufferguard_" + mappingName + ".svg"} << llama::toSvg(mapping);

    auto view1 = allocView(mapping);

    int i = 0;
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
        {
            view1(row, col)(tag::X{}) = ++i;
            view1(row, col)(tag::Y{}) = ++i;
            view1(row, col)(tag::Z{}) = ++i;
        }

    std::cout << "View 1:\n";
    printView(view1, rows, cols);

    auto view2 = allocView(mapping);
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
            view2(row, col) = 0; // broadcast

    std::cout << "\nView 2:\n";
    printView(view2, rows, cols);

    auto copyBlobs = [&](auto& srcView, auto& dstView, auto srcBlobs, auto dstBlobs)
    {
        static_assert(srcBlobs.size() == dstBlobs.size());
        for (auto i = 0; i < srcBlobs.size(); i++)
        {
            const auto src = srcBlobs[i];
            const auto dst = dstBlobs[i];
            assert(mapping.blobSize(src) == mapping.blobSize(dst));
            std::memcpy(&dstView.storageBlobs[dst][0], &srcView.storageBlobs[src][0], mapping.blobSize(src));
        }
    };

    std::cout << "\nCopy view 1 right -> view 2 left:\n";
    copyBlobs(view1, view2, mapping.rightBlobs(), mapping.leftBlobs());

    std::cout << "View 2:\n";
    printView(view2, rows, cols);

    std::cout << "\nCopy view 1 left top -> view 2 right bot:\n";
    copyBlobs(view1, view2, mapping.leftTopBlobs(), mapping.rightBotBlobs());

    std::cout << "View 2:\n";
    printView(view2, rows, cols);

    std::cout << "\nCopy view 2 center -> view 1 center:\n";
    copyBlobs(view2, view1, mapping.centerBlobs(), mapping.centerBlobs());

    std::cout << "View 1:\n";
    printView(view1, rows, cols);
}

auto main() -> int
try
{
    run<llama::mapping::PreconfiguredAoS<>::type>("AoS");
    run<llama::mapping::PreconfiguredSoA<>::type>("SoA");
    run<llama::mapping::PreconfiguredSoA<true>::type>("SoA_MB");
}
catch (const std::exception& e)
{
    std::cerr << "Exception: " << e.what() << '\n';
}

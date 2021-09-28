#include "common.hpp"

#include <catch2/catch.hpp>
#include <llama/llama.hpp>

namespace
{
    using ArrayDims = llama::ArrayDims<2>;
    using RecordDim = Vec3I;

    template<typename SrcMapping, typename DstMapping, typename CopyFunc>
    void testCopy(CopyFunc copy)
    {
        const auto viewSize = ArrayDims{4, 8};
        const auto srcMapping = SrcMapping(viewSize);
        auto srcView = llama::allocViewUninitialized(srcMapping);
        auto value = 0;
        for(auto ad : llama::ArrayDimsIndexRange{srcMapping.arrayDims()})
            llama::forEachLeafCoord<RecordDim>(
                [&](auto coord)
                {
                    srcView(ad)(coord) = value;
                    value++;
                });

        auto dstView = llama::allocViewUninitialized(DstMapping(viewSize));
        copy(srcView, dstView);

        value = 0;
        for(auto ad : llama::ArrayDimsIndexRange{srcMapping.arrayDims()})
            llama::forEachLeafCoord<RecordDim>(
                [&](auto coord)
                {
                    CHECK(dstView(ad)(coord) == value);
                    value++;
                });
    }

    // Do not test all combinations as this exlodes the unit test compile and runtime.
    using AoSMappings = boost::mp11::mp_list<
        llama::mapping::AoS<ArrayDims, RecordDim, false, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::AoS<ArrayDims, RecordDim, false, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::AoS<ArrayDims, RecordDim, true, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::AoS<ArrayDims, RecordDim, true, llama::mapping::LinearizeArrayDimsFortran>>;

    using OtherMappings = boost::mp11::mp_list<
        llama::mapping::SoA<ArrayDims, RecordDim, false, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::SoA<ArrayDims, RecordDim, false, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::SoA<ArrayDims, RecordDim, true, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::SoA<ArrayDims, RecordDim, true, llama::mapping::LinearizeArrayDimsFortran>,
        llama::mapping::AoSoA<ArrayDims, RecordDim, 4, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::AoSoA<ArrayDims, RecordDim, 4, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::AoSoA<ArrayDims, RecordDim, 8, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::AoSoA<ArrayDims, RecordDim, 8, llama::mapping::LinearizeArrayDimsFortran>>;

    using AllMappings = boost::mp11::mp_append<AoSMappings, OtherMappings>;

    using AllMappingsProduct = boost::mp11::mp_product<boost::mp11::mp_list, AllMappings, AllMappings>;

    template<typename List>
    using BothAreSoAOrHaveDifferentLinearizer = std::bool_constant<
        (llama::mapping::isSoA<boost::mp11::mp_first<List>> && llama::mapping::isSoA<boost::mp11::mp_second<List>>)
        || !std::is_same_v<
            typename boost::mp11::mp_first<List>::LinearizeArrayDimsFunctor,
            typename boost::mp11::mp_second<List>::LinearizeArrayDimsFunctor>>;

    using AoSoAMappingsProduct = boost::mp11::mp_remove_if<
        boost::mp11::mp_product<boost::mp11::mp_list, OtherMappings, OtherMappings>,
        BothAreSoAOrHaveDifferentLinearizer>;
} // namespace

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("copy", "", AllMappingsProduct)
{
    using SrcMapping = boost::mp11::mp_first<TestType>;
    using DstMapping = boost::mp11::mp_second<TestType>;
    testCopy<SrcMapping, DstMapping>([](const auto& srcView, auto& dstView) { llama::copy(srcView, dstView); });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("blobMemcpy", "", AllMappings)
{
    testCopy<TestType, TestType>([](const auto& srcView, auto& dstView) { llama::blobMemcpy(srcView, dstView); });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("fieldWiseCopy", "", AllMappingsProduct)
{
    using SrcMapping = boost::mp11::mp_first<TestType>;
    using DstMapping = boost::mp11::mp_second<TestType>;
    testCopy<SrcMapping, DstMapping>([](const auto& srcView, auto& dstView)
                                     { llama::fieldWiseCopy(srcView, dstView); });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("aosoaCommonBlockCopy.readOpt", "", AoSoAMappingsProduct)
{
    using SrcMapping = boost::mp11::mp_first<TestType>;
    using DstMapping = boost::mp11::mp_second<TestType>;
    testCopy<SrcMapping, DstMapping>([](const auto& srcView, auto& dstView)
                                     { llama::aosoaCommonBlockCopy(srcView, dstView, true); });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("aosoaCommonBlockCopy.writeOpt", "", AoSoAMappingsProduct)
{
    using SrcMapping = boost::mp11::mp_first<TestType>;
    using DstMapping = boost::mp11::mp_second<TestType>;
    testCopy<SrcMapping, DstMapping>([](const auto& srcView, auto& dstView)
                                     { llama::aosoaCommonBlockCopy(srcView, dstView, false); });
}

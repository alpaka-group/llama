#include "common.hpp"

namespace
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    using RecordDim = Vec3I;

    template<typename SrcMapping, typename DstMapping, typename CopyFunc>
    void testCopy(CopyFunc copy)
    {
        const auto viewExtents = ArrayExtents{4, 8};
        const auto srcMapping = SrcMapping(viewExtents);
        auto srcView = llama::allocViewUninitialized(srcMapping);
        iotaFillView(srcView);

        auto dstView = llama::allocViewUninitialized(DstMapping(viewExtents));
        copy(srcView, dstView);

        iotaCheckView(dstView);
    }

    // Do not test all combinations as this exlodes the unit test compile and runtime.
    using AoSMappings = boost::mp11::mp_list<
        llama::mapping::AoS<ArrayExtents, RecordDim, false, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::AoS<ArrayExtents, RecordDim, false, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::AoS<ArrayExtents, RecordDim, true, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::AoS<ArrayExtents, RecordDim, true, llama::mapping::LinearizeArrayDimsFortran>>;

    using OtherMappings = boost::mp11::mp_list<
        llama::mapping::SoA<ArrayExtents, RecordDim, false, false, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, false, false, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, true, false, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::SoA<ArrayExtents, RecordDim, true, false, llama::mapping::LinearizeArrayDimsFortran>,
        llama::mapping::SoA<ArrayExtents, RecordDim, false, true, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, false, true, llama::mapping::LinearizeArrayDimsFortran>,
        llama::mapping::AoSoA<ArrayExtents, RecordDim, 4, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::AoSoA<ArrayExtents, RecordDim, 4, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::AoSoA<ArrayExtents, RecordDim, 8, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::AoSoA<ArrayExtents, RecordDim, 8, llama::mapping::LinearizeArrayDimsFortran>>;

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

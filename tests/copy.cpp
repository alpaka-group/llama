// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

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
    using AoSMappings = mp_list<
        llama::mapping::
            AoS<ArrayExtents, RecordDim, llama::mapping::FieldAlignment::Pack, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::AoS<ArrayExtents, RecordDim, llama::mapping::FieldAlignment::Pack,
        // llama::mapping::LinearizeArrayDimsFortran>, llama::mapping::AoS<ArrayExtents, RecordDim,
        // llama::mapping::FieldAlignment::Align, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::AoS<
            ArrayExtents,
            RecordDim,
            llama::mapping::FieldAlignment::Align,
            llama::mapping::LinearizeArrayDimsFortran>>;

    using OtherMappings = mp_list<
        llama::mapping::SoA<
            ArrayExtents,
            RecordDim,
            llama::mapping::Blobs::Single,
            llama::mapping::SubArrayAlignment::Pack,
            llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, llama::mapping::Blobs::Single,
        // llama::mapping::SubArrayAlignment::Pack, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, llama::mapping::Blobs::OnePerField,
        // llama::mapping::SubArrayAlignment::Pack, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::SoA<
            ArrayExtents,
            RecordDim,
            llama::mapping::Blobs::OnePerField,
            llama::mapping::SubArrayAlignment::Pack,
            llama::mapping::LinearizeArrayDimsFortran>,
        llama::mapping::SoA<
            ArrayExtents,
            RecordDim,
            llama::mapping::Blobs::Single,
            llama::mapping::SubArrayAlignment::Align,
            llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, llama::mapping::Blobs::Single,
        // llama::mapping::SubArrayAlignment::Align, llama::mapping::LinearizeArrayDimsFortran>,
        llama::mapping::AoSoA<ArrayExtents, RecordDim, 4, llama::mapping::LinearizeArrayDimsCpp>,
        // llama::mapping::AoSoA<ArrayExtents, RecordDim, 4, llama::mapping::LinearizeArrayDimsFortran>,
        // llama::mapping::AoSoA<ArrayExtents, RecordDim, 8, llama::mapping::LinearizeArrayDimsCpp>,
        llama::mapping::AoSoA<ArrayExtents, RecordDim, 8, llama::mapping::LinearizeArrayDimsFortran>>;

    using AllMappings = mp_append<AoSMappings, OtherMappings>;

    using AllMappingsProduct = mp_product<mp_list, AllMappings, AllMappings>;

    template<typename List>
    using BothAreSoAOrHaveDifferentLinearizer = std::bool_constant<
        (llama::mapping::isSoA<mp_first<List>> && llama::mapping::isSoA<mp_second<List>>)
        || !std::is_same_v<
            typename mp_first<List>::LinearizeArrayDimsFunctor,
            typename mp_second<List>::LinearizeArrayDimsFunctor>>;

    using AoSoAMappingsProduct
        = mp_remove_if<mp_product<mp_list, OtherMappings, OtherMappings>, BothAreSoAOrHaveDifferentLinearizer>;
} // namespace

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("copy", "", AllMappingsProduct)
{
    using SrcMapping = mp_first<TestType>;
    using DstMapping = mp_second<TestType>;
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
    using SrcMapping = mp_first<TestType>;
    using DstMapping = mp_second<TestType>;
    testCopy<SrcMapping, DstMapping>([](const auto& srcView, auto& dstView)
                                     { llama::fieldWiseCopy(srcView, dstView); });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("aosoaCommonBlockCopy.readOpt", "", AoSoAMappingsProduct)
{
    using SrcMapping = mp_first<TestType>;
    using DstMapping = mp_second<TestType>;
    testCopy<SrcMapping, DstMapping>([](const auto& srcView, auto& dstView)
                                     { llama::aosoaCommonBlockCopy(srcView, dstView, true); });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("aosoaCommonBlockCopy.writeOpt", "", AoSoAMappingsProduct)
{
    using SrcMapping = mp_first<TestType>;
    using DstMapping = mp_second<TestType>;
    testCopy<SrcMapping, DstMapping>([](const auto& srcView, auto& dstView)
                                     { llama::aosoaCommonBlockCopy(srcView, dstView, false); });
}

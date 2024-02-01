// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: MPL-2.0

#include "common.hpp"

#include <thread>

namespace
{
    using ArrayExtents = llama::ArrayExtentsDynamic<std::size_t, 2>;
    using RecordDim = llama::Record<llama::Field<int, int>, llama::Field<char, char>, llama::Field<double, double>>;

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
        llama::mapping::AoS<
            ArrayExtents,
            RecordDim,
            llama::mapping::FieldAlignment::Pack,
            llama::mapping::LinearizeArrayIndexRight>,
        // llama::mapping::AoS<ArrayExtents, RecordDim, llama::mapping::FieldAlignment::Pack,
        // llama::mapping::LinearizeArrayIndexLeft>, llama::mapping::AoS<ArrayExtents, RecordDim,
        // llama::mapping::FieldAlignment::Align, llama::mapping::LinearizeArrayIndexRight>,
        llama::mapping::AoS<
            ArrayExtents,
            RecordDim,
            llama::mapping::FieldAlignment::Align,
            llama::mapping::LinearizeArrayIndexLeft>>;

    using OtherMappings = mp_list<
        llama::mapping::SoA<
            ArrayExtents,
            RecordDim,
            llama::mapping::Blobs::Single,
            llama::mapping::SubArrayAlignment::Pack,
            llama::mapping::LinearizeArrayIndexRight>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, llama::mapping::Blobs::Single,
        // llama::mapping::SubArrayAlignment::Pack, llama::mapping::LinearizeArrayIndexLeft>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, llama::mapping::Blobs::OnePerField,
        // llama::mapping::SubArrayAlignment::Pack, llama::mapping::LinearizeArrayIndexRight>,
        llama::mapping::SoA<
            ArrayExtents,
            RecordDim,
            llama::mapping::Blobs::OnePerField,
            llama::mapping::SubArrayAlignment::Pack,
            llama::mapping::LinearizeArrayIndexLeft>,
        llama::mapping::SoA<
            ArrayExtents,
            RecordDim,
            llama::mapping::Blobs::Single,
            llama::mapping::SubArrayAlignment::Align,
            llama::mapping::LinearizeArrayIndexRight>,
        // llama::mapping::SoA<ArrayExtents, RecordDim, llama::mapping::Blobs::Single,
        // llama::mapping::SubArrayAlignment::Align, llama::mapping::LinearizeArrayIndexLeft>,
        llama::mapping::AoSoA<
            ArrayExtents,
            RecordDim,
            4,
            llama::mapping::FieldAlignment::Align,
            llama::mapping::LinearizeArrayIndexRight>,
        // llama::mapping::AoSoA<ArrayExtents, RecordDim, 4, llama::mapping::LinearizeArrayIndexLeft>,
        // llama::mapping::AoSoA<ArrayExtents, RecordDim, 8, llama::mapping::LinearizeArrayIndexRight>,
        llama::mapping::AoSoA<
            ArrayExtents,
            RecordDim,
            8,
            llama::mapping::FieldAlignment::Pack,
            llama::mapping::LinearizeArrayIndexLeft>>;

    using AllMappings = mp_append<AoSMappings, OtherMappings>;

    using AllMappingsProduct = mp_product<mp_list, AllMappings, AllMappings>;

    template<typename List>
    using BothAreSoAOrHaveDifferentLinearizer = std::bool_constant<
        (llama::mapping::isSoA<mp_first<List>> && llama::mapping::isSoA<mp_second<List>>)
        || !std::is_same_v<
            typename mp_first<List>::LinearizeArrayIndexFunctor,
            typename mp_second<List>::LinearizeArrayIndexFunctor>>;

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
TEMPLATE_LIST_TEST_CASE("memcpyBlobs_default", "", AllMappings)
{
    testCopy<TestType, TestType>([](const auto& srcView, auto& dstView) { llama::memcpyBlobs(srcView, dstView); });
}

TEMPLATE_LIST_TEST_CASE("memcpyBlobs_3threads", "", AllMappings)
{
    testCopy<TestType, TestType>(
        [](const auto& srcView, auto& dstView)
        {
            std::thread t1{[&] { llama::memcpyBlobs(srcView, dstView, 0, 3); }};
            std::thread t2{[&] { llama::memcpyBlobs(srcView, dstView, 1, 3); }};
            std::thread t3{[&] { llama::memcpyBlobs(srcView, dstView, 2, 3); }};
            t1.join();
            t2.join();
            t3.join();
        });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("copyBlobs_default", "", AllMappings)
{
    testCopy<TestType, TestType>([](const auto& srcView, auto& dstView) { llama::copyBlobs(srcView, dstView); });
}

// NOLINTNEXTLINE(cert-err58-cpp)
TEMPLATE_LIST_TEST_CASE("copyBlobs_stdcopy", "", AllMappings)
{
    testCopy<TestType, TestType>(
        [](const auto& srcView, auto& dstView)
        {
            llama::copyBlobs(
                srcView,
                dstView,
                [](const auto& srcBlob, auto& dstBlob, std::size_t size)
                { std::copy(&srcBlob[0], &srcBlob[0] + size, &dstBlob[0]); });
        });
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

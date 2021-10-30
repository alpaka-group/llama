#pragma once

#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
        auto partitionRecordDim(Record<Fields...>, RecordCoord<FirstCoord, Coords...>)
        {
            using namespace boost::mp11;
            using Rec = Record<Fields...>;
            if constexpr(sizeof...(Coords) == 0)
            {
                using Part1 = Record<mp_at_c<Rec, FirstCoord>>;
                using Part2 = mp_erase_c<Rec, FirstCoord, FirstCoord + 1>;
                return mp_list<Part1, Part2>{};
            }
            else
            {
                using FieldTag = GetTag<Rec, RecordCoord<FirstCoord>>;
                using FieldType = GetType<Rec, RecordCoord<FirstCoord>>;
                using InnerPartition = decltype(partitionRecordDim(FieldType{}, RecordCoord<Coords...>{}));
                using Part1 = Record<Field<FieldTag, mp_first<InnerPartition>>>;
                using Part2 = mp_replace_at_c<Rec, FirstCoord, Field<FieldTag, mp_second<InnerPartition>>>;
                return mp_list<Part1, Part2>{};
            }
        }

        template<typename Acc, typename TagList>
        struct PartitionFoldOpImpl
        {
            using Part1Before = boost::mp11::mp_first<Acc>;
            using Part2Before = boost::mp11::mp_second<Acc>;
            using R = decltype(partitionRecordDim(Part2Before{}, GetCoordFromTags<Part2Before, TagList>{}));
            using Part1After = boost::mp11::mp_first<R>;
            using Part2After = boost::mp11::mp_second<R>;

            using type = boost::mp11::mp_list<MergedRecordDims<Part1Before, Part1After>, Part2After>;
        };

        template<typename Acc, typename TagList>
        using PartitionFoldOp = typename PartitionFoldOpImpl<Acc, TagList>::type;

        template<typename... Fields, typename... RCs>
        auto partitionRecordDim(Record<Fields...>, boost::mp11::mp_list<RCs...>)
        {
            using namespace boost::mp11;
            using Initial = mp_list<Record<>, Record<Fields...>>; // initially, nothing selected for mapping 1
            return mp_fold<mp_list<GetTags<Record<Fields...>, RCs>...>, Initial, PartitionFoldOp>{};
        }

        // workaround for nvcc 11.3 and below: we cannot put the decltype() directly into the Split class
        template<typename RecordDim, typename RecordCoordForMapping1>
        struct PartionedRecordDim
        {
            using type = decltype(partitionRecordDim(RecordDim{}, RecordCoordForMapping1{}));
        };

        template<typename RC, typename RecordCoordForMapping1>
        inline constexpr bool isSelected = RecordCoordCommonPrefixIsSame<RecordCoordForMapping1, RC>;

        template<typename RC>
        struct IsSelectedPredicate
        {
            template<typename RecordCoordForMapping1>
            using fn = boost::mp11::mp_bool<isSelected<RC, RecordCoordForMapping1>>;
        };

        template<typename RC, typename... RecordCoordsForMapping1>
        inline constexpr bool isSelected<RC, boost::mp11::mp_list<RecordCoordsForMapping1...>> = boost::mp11::
            mp_any_of_q<boost::mp11::mp_list<RecordCoordsForMapping1...>, IsSelectedPredicate<RC>>::value;
    } // namespace internal

    /// Mapping which splits off a part of the record dimension and maps it differently then the rest.
    /// \tparam RecordCoordForMapping1 A \ref RecordCoord or a list of RecordCoords selecting the part of the record
    /// dimension to be mapped differently.
    /// \tparam MappingTemplate1 The mapping used for the selected part of the record dimension.
    /// \tparam MappingTemplate2 The mapping used for the not selected part of the record dimension.
    /// \tparam SeparateBlobs If true, both pieces of the record dimension are mapped to separate blobs.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename RecordCoordForMapping1,
        template<typename...>
        typename MappingTemplate1,
        template<typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct Split
    {
        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;

        using RecordDimPartitions = typename internal::PartionedRecordDim<RecordDim, RecordCoordForMapping1>::type;
        using RecordDim1 = boost::mp11::mp_first<RecordDimPartitions>;
        using RecordDim2 = boost::mp11::mp_second<RecordDimPartitions>;

        using Mapping1 = MappingTemplate1<ArrayExtents, RecordDim1>;
        using Mapping2 = MappingTemplate2<ArrayExtents, RecordDim2>;

        static constexpr std::size_t blobCount = SeparateBlobs ? Mapping1::blobCount + Mapping2::blobCount : 1;
        static_assert(SeparateBlobs || Mapping1::blobCount == 1);
        static_assert(SeparateBlobs || Mapping2::blobCount == 1);

        constexpr Split() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit Split(ArrayExtents extents) : mapping1(extents), mapping2(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return mapping1.extents();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize([[maybe_unused]] std::size_t i) const -> std::size_t
        {
            if constexpr(SeparateBlobs)
            {
                if(i < Mapping1::blobCount)
                    return mapping1.blobSize(i);
                return mapping2.blobSize(i - Mapping1::blobCount);
            }
            else
                return mapping1.blobSize(0) + mapping2.blobSize(0);
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> = {}) const
            -> NrAndOffset
        {
            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;

            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, RecordCoordForMapping1>)
                return mapping1.blobNrAndOffset(ai, GetCoordFromTags<RecordDim1, Tags>{});
            else
            {
                auto nrAndOffset = mapping2.blobNrAndOffset(ai, GetCoordFromTags<RecordDim2, Tags>{});
                if constexpr(SeparateBlobs)
                    nrAndOffset.nr += Mapping1::blobCount;
                else
                {
                    for(std::size_t i = 0; i < Mapping1::blobCount; i++)
                        nrAndOffset.offset += mapping1.blobSize(i);
                }
                return nrAndOffset;
            }
        }

        Mapping1 mapping1;
        Mapping2 mapping2;
    };

    template<
        typename RecordCoordsForMapping1,
        template<typename...>
        typename MappingTemplate1,
        template<typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct PreconfiguredSplit
    {
        template<typename ArrayExtents, typename RecordDim>
        using type = Split<
            ArrayExtents,
            RecordDim,
            RecordCoordsForMapping1,
            MappingTemplate1,
            MappingTemplate2,
            SeparateBlobs>;
    };
} // namespace llama::mapping

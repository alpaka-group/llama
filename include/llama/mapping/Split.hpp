#pragma once

#include "../View.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template<typename... Fields, std::size_t FirstCoord, std::size_t... Coords>
        auto partitionRecordDim(Record<Fields...>, RecordCoord<FirstCoord, Coords...>)
        {
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
            using Part1Before = mp_first<Acc>;
            using Part2Before = mp_second<Acc>;
            using R = decltype(partitionRecordDim(Part2Before{}, GetCoordFromTags<Part2Before, TagList>{}));
            using Part1After = mp_first<R>;
            using Part2After = mp_second<R>;

            using type = mp_list<MergedRecordDims<Part1Before, Part1After>, Part2After>;
        };

        template<typename Acc, typename TagList>
        using PartitionFoldOp = typename PartitionFoldOpImpl<Acc, TagList>::type;

        template<typename... Fields, typename... RCs>
        auto partitionRecordDim(Record<Fields...>, mp_list<RCs...>)
        {
            static_assert((isRecordCoord<RCs> && ...));
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
        inline constexpr bool isSelected = recordCoordCommonPrefixIsSame<RecordCoordForMapping1, RC>;

        template<typename RC, typename... RecordCoordsForMapping1>
        inline constexpr bool isSelected<RC, mp_list<RecordCoordsForMapping1...>>
            = (isSelected<RC, RecordCoordsForMapping1> || ...);

        template<typename RecordDim, typename Selector>
        struct ReplaceTagListsByCoords;

        template<typename RecordDim, std::size_t... RCs>
        struct ReplaceTagListsByCoords<RecordDim, RecordCoord<RCs...>>
        {
            using type = RecordCoord<RCs...>;
        };

        template<typename RecordDim, typename... Args>
        struct ReplaceTagListsByCoords<RecordDim, mp_list<Args...>>
        {
            static auto f()
            {
                if constexpr(((mp_is_list<Args>::value || isRecordCoord<Args>) &&...))
                    // Args is a pack of tag lists/record coords
                    return mp_list<GetCoordFromTags<RecordDim, Args>...>{};
                else
                    // Args is a single tag lists
                    return GetCoordFromTags<RecordDim, Args...>{};
            }

            using type = decltype(f());
        };
    } // namespace internal

    /// Mapping which splits off a part of the record dimension and maps it differently then the rest.
    /// \tparam TSelectorForMapping1 Selects a part of the record dimension to be mapped by MappingTemplate1. Can be a
    /// \ref RecordCoord, a type list of RecordCoords, a type list of tags (selecting one field), or a type list of
    /// type list of tags (selecting one field per sub list). dimension to be mapped differently.
    /// \tparam MappingTemplate1 The mapping used for the selected part of the record dimension.
    /// \tparam MappingTemplate2 The mapping used for the not selected part of the record dimension.
    /// \tparam SeparateBlobs If true, both pieces of the record dimension are mapped to separate blobs.
    template<
        typename TArrayExtents,
        typename TRecordDim,
        typename TSelectorForMapping1,
        template<typename...>
        typename MappingTemplate1,
        template<typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct Split
    {
        static_assert(isRecord<TRecordDim>, "Cannot split a scalar record dim");

        using ArrayExtents = TArrayExtents;
        using ArrayIndex = typename ArrayExtents::Index;
        using RecordDim = TRecordDim;

        using SelectorForMapping1 = TSelectorForMapping1;
        using NormalizedSelectorForMapping1 =
            typename internal::ReplaceTagListsByCoords<RecordDim, SelectorForMapping1>::type;
        using RecordDimPartitions =
            typename internal::PartionedRecordDim<RecordDim, NormalizedSelectorForMapping1>::type;
        using RecordDim1 = mp_first<RecordDimPartitions>;
        using RecordDim2 = mp_second<RecordDimPartitions>;

        using Mapping1 = MappingTemplate1<ArrayExtents, RecordDim1>;
        using Mapping2 = MappingTemplate2<ArrayExtents, RecordDim2>;

        static constexpr std::size_t blobCount = SeparateBlobs ? Mapping1::blobCount + Mapping2::blobCount : 1;
        static_assert(SeparateBlobs || Mapping1::blobCount == 1);
        static_assert(SeparateBlobs || Mapping2::blobCount == 1);

    private:
        using size_type = typename ArrayExtents::value_type;
        static constexpr size_type m1bc = static_cast<size_type>(Mapping1::blobCount);

    public:
        constexpr Split() = default;

        LLAMA_FN_HOST_ACC_INLINE
        constexpr explicit Split(ArrayExtents extents) : mapping1(extents), mapping2(extents)
        {
        }

        LLAMA_FN_HOST_ACC_INLINE
        constexpr Split(Mapping1 mapping1, Mapping2 mapping2)
            : mapping1(std::move(mapping1))
            , mapping2(std::move(mapping2))
        {
        }

        template<typename... Args1, typename... Args2>
        LLAMA_FN_HOST_ACC_INLINE constexpr Split(std::tuple<Args1...> mappingArgs1, std::tuple<Args2...> mappingArgs2)
            : mapping1(std::make_from_tuple<Mapping1>(mappingArgs1))
            , mapping2(std::make_from_tuple<Mapping2>(mappingArgs2))
        {
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> ArrayExtents
        {
            return mapping1.extents();
        }

        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobSize([[maybe_unused]] size_type i) const -> size_type
        {
            if constexpr(SeparateBlobs)
            {
                if(i < m1bc)
                    return mapping1.blobSize(i);
                return mapping2.blobSize(i - m1bc);
            }
            else
                return mapping1.blobSize(0) + mapping2.blobSize(0);
        }

        template<std::size_t... RecordCoords>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto blobNrAndOffset(ArrayIndex ai, RecordCoord<RecordCoords...> = {}) const
            -> NrAndOffset<size_type>
        {
            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;

            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, NormalizedSelectorForMapping1>)
                return mapping1.blobNrAndOffset(ai, GetCoordFromTags<RecordDim1, Tags>{});
            else
            {
                auto nrAndOffset = mapping2.blobNrAndOffset(ai, GetCoordFromTags<RecordDim2, Tags>{});
                if constexpr(SeparateBlobs)
                    nrAndOffset.nr += m1bc;
                else
                {
                    for(size_type i = 0; i < m1bc; i++)
                        nrAndOffset.offset += mapping1.blobSize(i);
                }
                return nrAndOffset;
            }
        }

        template<std::size_t... RecordCoords>
        static constexpr auto isComputed(RecordCoord<RecordCoords...>) -> bool
        {
            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;
            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, NormalizedSelectorForMapping1>)
                return llama::isComputed<Mapping1, GetCoordFromTags<RecordDim1, Tags>>;
            else
                return llama::isComputed<Mapping2, GetCoordFromTags<RecordDim2, Tags>>;
        }

        template<std::size_t... RecordCoords, typename Blobs>
        LLAMA_FN_HOST_ACC_INLINE constexpr auto compute(ArrayIndex ai, RecordCoord<RecordCoords...>, Blobs& blobs)
            const
        {
            using Tags = GetTags<RecordDim, RecordCoord<RecordCoords...>>;
            if constexpr(internal::isSelected<RecordCoord<RecordCoords...>, NormalizedSelectorForMapping1>)
                return mapping1.compute(ai, GetCoordFromTags<RecordDim1, Tags>{}, blobs);
            else
            {
                // only pass on blobs for mapping 2, so it can index starting from 0
                auto* blobs2 = &blobs[0] + m1bc;
                return mapping2.compute(ai, GetCoordFromTags<RecordDim2, Tags>{}, blobs2);
            }
        }

        Mapping1 mapping1;
        Mapping2 mapping2;
    };

    /// Binds parameters to a \ref Split mapping except for array and record dimension, producing a quoted
    /// meta function accepting the latter two. Useful to to prepare this mapping for a meta mapping.
    template<
        typename SelectorForMapping1,
        template<typename...>
        typename MappingTemplate1,
        template<typename...>
        typename MappingTemplate2,
        bool SeparateBlobs = false>
    struct BindSplit
    {
        template<typename ArrayExtents, typename RecordDim>
        using fn
            = Split<ArrayExtents, RecordDim, SelectorForMapping1, MappingTemplate1, MappingTemplate2, SeparateBlobs>;
    };

    template<typename Mapping>
    inline constexpr bool isSplit = false;

    template<
        typename ArrayExtents,
        typename RecordDim,
        typename SelectorForMapping1,
        template<typename...>
        typename MappingTemplate1,
        template<typename...>
        typename MappingTemplate2,
        bool SeparateBlobs>
    inline constexpr bool
        isSplit<Split<ArrayExtents, RecordDim, SelectorForMapping1, MappingTemplate1, MappingTemplate2, SeparateBlobs>>
        = true;
} // namespace llama::mapping

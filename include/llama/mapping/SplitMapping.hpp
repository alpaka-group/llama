#pragma once

#include "../Types.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    namespace internal
    {
        template <typename... DatumElements, std::size_t FirstCoord, std::size_t... Coords>
        auto partitionDatumDomain(DatumStruct<DatumElements...>, DatumCoord<FirstCoord, Coords...>)
        {
            if constexpr (sizeof...(Coords) == 0)
            {
                using With = DatumStruct<boost::mp11::mp_at_c<DatumStruct<DatumElements...>, FirstCoord>>;
                using Without
                    = DatumStruct<boost::mp11::mp_erase_c<DatumStruct<DatumElements...>, FirstCoord, FirstCoord>>;
                return boost::mp11::mp_list<With, Without> {};
            }
            else
            {
                using Result = decltype(partitionDatumDomain(
                    DatumStruct<boost::mp11::mp_at_c<DatumStruct<DatumElements...>, FirstCoord>> {},
                    DatumCoord<Coords...> {}));
                using With = boost::mp11::
                    mp_replace_at_c<DatumStruct<DatumElements...>, FirstCoord, boost::mp11::mp_first<Result>>;
                using Without = boost::mp11::
                    mp_replace_at_c<DatumStruct<DatumElements...>, FirstCoord, boost::mp11::mp_second<Result>>;
                return boost::mp11::mp_list<With, Without> {};
            }
        }
    } // namespace internal

    template <
        typename T_ArrayDomain,
        typename T_DatumDomain,
        typename DatumCoordForMapping1,
        template <typename, typename>
        typename Mapping1,
        template <typename, typename>
        typename Mapping2>
    struct SplitMapping
    {
        using ArrayDomain = T_ArrayDomain;
        using DatumDomain = T_DatumDomain;

        using DatumDomainPartitions
            = decltype(internal::partitionDatumDomain(DatumDomain {}, DatumCoordForMapping1 {}));
        using DatumDomain1 = boost::mp11::mp_first<DatumDomainPartitions>;
        using DatumDomain2 = boost::mp11::mp_second<DatumDomainPartitions>;

        static constexpr std::size_t blobCount = 1;
        // = Mapping1<ArrayDomain, DatumDomain1>::blobCount + Mapping2<ArrayDomain, DatumDomain2>::blobCount;
        static_assert(Mapping1<ArrayDomain, DatumDomain>::blobCount == 1);
        static_assert(Mapping2<ArrayDomain, DatumDomain>::blobCount == 1);

        SplitMapping() = default;

        LLAMA_FN_HOST_ACC_INLINE
        SplitMapping(ArrayDomain size)
            : userDomainSize(size)
            , mapping1(size)
            , mapping2(size)
            , mapping1BlobSize(mapping1.getBlobSize(0))
        {
        }

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const -> std::size_t
        {
            return mapping1BlobSize + mapping2.getBlobSize();
        }

        template <std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(ArrayDomain coord) const -> NrAndOffset
        {
            if constexpr (DatumCoordCommonPrefixIsSame<DatumCoordForMapping1, DatumCoord<DatumDomainCoord...>>)
                return mapping1.template getBlobNrAndOffset<DatumDomainCoord...>(coord);
            else
            {
                auto blobNrAndOffset = mapping2.template getBlobNrAndOffset<DatumDomainCoord...>(coord);
                blobNrAndOffset.offset += mapping1BlobSize;
                return blobNrAndOffset;
            }
        }

        ArrayDomain userDomainSize = {};
        Mapping1<ArrayDomain, DatumDomain> mapping1;
        Mapping2<ArrayDomain, DatumDomain> mapping2;
        std::size_t mapping1BlobSize;
    };
} // namespace llama::mapping

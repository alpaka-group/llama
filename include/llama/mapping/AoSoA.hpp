#pragma once

#include "../Types.hpp"
#include "Common.hpp"

namespace llama::mapping
{
    /// Array of struct of arrays mapping. Used to create a \ref View via \ref
    /// allocView.
    /// \tparam Lanes The size of the inner arrays of this array of struct of
    /// arrays.
    /// \tparam LinearizeUserDomainAdressFunctor Defines how the user
    /// domain should be mapped into linear numbers.
    /// \tparam ExtentUserDomainAdressFunctor Defines how the total number of
    /// \ref UserDomain indices is calculated.
    template<
        typename T_UserDomain,
        typename T_DatumDomain,
        std::size_t Lanes,
        typename LinearizeUserDomainAdressFunctor = LinearizeUserDomainAdress,
        typename ExtentUserDomainAdressFunctor = ExtentUserDomainAdress>
    struct AoSoA
    {
        using UserDomain = T_UserDomain;
        using DatumDomain = T_DatumDomain;
        static constexpr std::size_t blobCount = 1;

        AoSoA() = default;

        LLAMA_FN_HOST_ACC_INLINE
        AoSoA(UserDomain size) : userDomainSize(size) {}

        LLAMA_FN_HOST_ACC_INLINE auto getBlobSize(std::size_t) const
            -> std::size_t
        {
            return ExtentUserDomainAdressFunctor{}(
                userDomainSize)*sizeOf<DatumDomain>;
        }

        template<std::size_t... DatumDomainCoord>
        LLAMA_FN_HOST_ACC_INLINE auto getBlobNrAndOffset(UserDomain coord) const
            -> NrAndOffset
        {
            constexpr auto elementSize
                = sizeof(GetType<DatumDomain, DatumCoord<DatumDomainCoord...>>);
            constexpr auto elementOff
                = offsetOf<DatumDomain, DatumDomainCoord...>;
            constexpr auto datumDomainSize = sizeOf<DatumDomain>;
            LLAMA_FORCE_INLINE_RECURSIVE
            const auto userDomainIndex
                = LinearizeUserDomainAdressFunctor{}(coord, userDomainSize);

            const auto blockIndex = userDomainIndex / Lanes;
            const auto laneIndex = userDomainIndex % Lanes;
            const auto offset = (datumDomainSize * Lanes) * blockIndex
                + elementOff * Lanes + elementSize * laneIndex;
            return {0, offset};
        }

        UserDomain userDomainSize;
    };
}

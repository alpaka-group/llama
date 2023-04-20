// Copyright 2022 Bernhard Manfred Gruber
// SPDX-License-Identifier: LGPL-3.0-or-later

#pragma once

#include <boost/core/demangle.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llama/llama.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <typeinfo>

// make boost::mp11 directly available in all testing code
using namespace boost::mp11; // NOLINT(google-global-names-in-headers)

// NOLINTNEXTLINE(google-runtime-int)
using SizeTypes = mp_list<int, unsigned, long, unsigned long, long long, unsigned long long>;
static_assert(mp_contains<SizeTypes, std::size_t>::value);

// clang-format off
namespace tag
{
    struct Pos {};
    struct Vel {};
    struct X {};
    struct Y {};
    struct Z {};
    struct Mass {};
    struct Flags {};
    struct Id {};
    struct A {};
    struct B {};
    struct C {};
    struct Normal {};
    struct Part1 {};
    struct Part2 {};
} // namespace tag

using Vec2F = llama::Record<
    llama::Field<tag::X, float>,
    llama::Field<tag::Y, float>
>;
using Vec3D = llama::Record<
    llama::Field<tag::X, double>,
    llama::Field<tag::Y, double>,
    llama::Field<tag::Z, double>
>;
using Vec3I = llama::Record<
    llama::Field<tag::X, int>,
    llama::Field<tag::Y, int>,
    llama::Field<tag::Z, int>
>;
using Particle = llama::Record<
    llama::Field<tag::Pos, Vec3D>,
    llama::Field<tag::Mass, float>,
    llama::Field<tag::Vel, Vec3D>,
    llama::Field<tag::Flags, bool[4]>
>;
using ParticleHeatmap = llama::Record<
    llama::Field<tag::Pos, Vec3D>,
    llama::Field<tag::Vel, Vec3D>,
    llama::Field<tag::Mass, float>
>;
using ParticleUnaligned = llama::Record<
    llama::Field<tag::Id, std::uint16_t>,
    llama::Field<tag::Pos, Vec2F>,
    llama::Field<tag::Mass, double>,
    llama::Field<tag::Flags, bool[3]>
>;
// clang-format on

// TODO(bgruber): replace by boost::core::type_name<T>() once released and available
template<typename T>
auto prettyPrintType(const T& t = {}) -> std::string
{
    using llama::mapping::tree::internal::replaceAll;

    auto raw = boost::core::demangle(typeid(t).name());
#ifdef _MSC_VER
    // remove clutter in MSVC
    replaceAll(raw, "struct ", "");
#endif
#ifdef __GNUG__
    // remove clutter in g++
    static const std::regex ulLiteral{"(\\d+)ul"};
    raw = std::regex_replace(raw, ulLiteral, "$1");
#endif

    replaceAll(raw, "<", "<\n");
#ifdef _MSC_VER
    replaceAll(raw, ",", ",\n");
#else
    replaceAll(raw, ", ", ",\n");
#endif
    replaceAll(raw, " >", ">");
    replaceAll(raw, ">", "\n>");

    std::stringstream rawSS(raw);
    std::string token;
    std::string result;
    int indent = 0;
    while(std::getline(rawSS, token, '\n'))
    {
        if(token.back() == '>' || (token.length() > 1 && token[token.length() - 2] == '>'))
            indent -= 4;
        for(int i = 0; i < indent; ++i)
            result += ' ';
        result += token + "\n";
        if(token.back() == '<')
            indent += 4;
    }
    if(result.back() == '\n')
        result.pop_back();
    return result;
}

namespace internal
{
    inline void zeroBlob(std::shared_ptr<std::byte[]>& sp, size_t blobSize)
    {
        std::memset(sp.get(), 0, blobSize);
    }
    template<typename A>
    void zeroBlob(std::vector<std::byte, A>& v, size_t blobSize)
    {
        std::memset(v.data(), 0, blobSize);
    }
} // namespace internal

template<typename View>
void iotaStorage(View& view)
{
    for(auto i = 0; i < View::Mapping::blobCount; i++)
    {
        auto fillFunc = [val = 0]() mutable { return static_cast<typename View::BlobType::PrimType>(val++); };
        std::generate_n(view.blobs()[i].blobs().get(), view.mapping().blobSize(i), fillFunc);
    }
}

template<typename View>
void iotaFillView(View& view)
{
    std::int64_t value = 0;
    using RecordDim = typename View::RecordDim;
    for(auto ai : llama::ArrayIndexRange{view.extents()})
    {
        if constexpr(llama::isRecordDim<RecordDim>)
        {
            llama::forEachLeafCoord<RecordDim>(
                [&](auto rc)
                {
                    using Type = llama::GetType<RecordDim, decltype(rc)>;
                    view(ai)(rc) = static_cast<Type>(value);
                    ++value;
                });
        }
        else
        {
            view(ai) = static_cast<RecordDim>(value);
            ++value;
        }
    }
}

template<typename View>
void iotaCheckView(View& view)
{
    std::int64_t value = 0;
    using RecordDim = typename View::RecordDim;
    for(auto ai : llama::ArrayIndexRange{view.extents()})
    {
        if constexpr(llama::isRecordDim<RecordDim>)
        {
            llama::forEachLeafCoord<RecordDim>(
                [&](auto rc)
                {
                    CAPTURE(ai, rc);
                    using Type = llama::GetType<RecordDim, decltype(rc)>;
                    CHECK(view(ai)(rc) == static_cast<Type>(value));
                    ++value;
                });
        }
        else
        {
            CHECK(view(ai) == static_cast<RecordDim>(value));
            ++value;
        }
    }
}

// maps each element of the record dimension into a separate blobs. Each blob stores Modulus elements. If the array
// dimensions are larger than Modulus, elements are overwritten.
template<typename TArrayExtents, typename TRecordDim, std::size_t Modulus>
struct ModulusMapping : TArrayExtents
{
    using ArrayExtents = TArrayExtents;
    using ArrayIndex = typename ArrayExtents::Index;
    using RecordDim = TRecordDim;
    static constexpr std::size_t blobCount = mp_size<llama::FlatRecordDim<RecordDim>>::value;

    LLAMA_FN_HOST_ACC_INLINE
    constexpr explicit ModulusMapping(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
    {
    }

    LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> const ArrayExtents&
    {
        return *this;
    }

    constexpr auto blobSize(std::size_t) const -> std::size_t
    {
        return Modulus * llama::sizeOf<RecordDim>;
    }

    template<std::size_t... RecordCoords>
    constexpr auto blobNrAndOffset(ArrayIndex ai, llama::RecordCoord<RecordCoords...> = {}) const
        -> llama::NrAndOffset<std::size_t>
    {
        const auto blob = llama::flatRecordCoord<RecordDim, llama::RecordCoord<RecordCoords...>>;
        const auto offset = (llama::mapping::LinearizeArrayDimsCpp{}(ai, extents()) % Modulus)
            * sizeof(llama::GetType<RecordDim, llama::RecordCoord<RecordCoords...>>);
        return {blob, offset};
    }
};

// Maps everything to blob 0, offset 0
template<typename TArrayExtents, typename TRecordDim>
struct MapEverythingToZero : TArrayExtents
{
    using ArrayExtents = TArrayExtents;
    using ArrayIndex = typename ArrayExtents::Index;
    using RecordDim = TRecordDim;
    static constexpr std::size_t blobCount = 1;

    LLAMA_FN_HOST_ACC_INLINE
    constexpr explicit MapEverythingToZero(ArrayExtents extents, RecordDim = {}) : ArrayExtents(extents)
    {
    }

    LLAMA_FN_HOST_ACC_INLINE constexpr auto extents() const -> const ArrayExtents&
    {
        return *this;
    }

    constexpr auto blobSize(std::size_t) const -> std::size_t
    {
        return llama::product(extents()) * llama::sizeOf<RecordDim>;
    }

    template<std::size_t... RecordCoords>
    constexpr auto blobNrAndOffset(ArrayIndex, llama::RecordCoord<RecordCoords...> = {}) const
        -> llama::NrAndOffset<std::size_t>
    {
        return {0, 0};
    }
};

using Triangle = llama::Record<
    llama::Field<tag::A, Vec3D>,
    llama::Field<tag::B, Vec3D>,
    llama::Field<tag::C, Vec3D>,
    llama::Field<tag::Normal, Vec3D>>;

template<typename ArrayExtents, typename RecordDim>
struct TriangleAoSWithComputedNormal : llama::mapping::PackedAoS<ArrayExtents, RecordDim>
{
    using Base = llama::mapping::PackedAoS<ArrayExtents, RecordDim>;
    using typename Base::ArrayIndex;

    using Base::Base;

    template<std::size_t... RecordCoords>
    static constexpr auto isComputed(llama::RecordCoord<RecordCoords...>)
    {
        return llama::recordCoordCommonPrefixIsSame<llama::RecordCoord<RecordCoords...>, llama::RecordCoord<3>>;
    }

    template<std::size_t... RecordCoords, typename Blob>
    constexpr auto compute(
        ArrayIndex ai,
        llama::RecordCoord<RecordCoords...>,
        llama::Array<Blob, Base::blobCount>& blobs) const
    {
        auto fetch = [&](llama::NrAndOffset<std::size_t> nrAndOffset) -> double
        { return *reinterpret_cast<double*>(&blobs[nrAndOffset.nr][nrAndOffset.offset]); };

        const auto ax = fetch(Base::template blobNrAndOffset<0, 0>(ai));
        const auto ay = fetch(Base::template blobNrAndOffset<0, 1>(ai));
        const auto az = fetch(Base::template blobNrAndOffset<0, 2>(ai));
        const auto bx = fetch(Base::template blobNrAndOffset<1, 0>(ai));
        const auto by = fetch(Base::template blobNrAndOffset<1, 1>(ai));
        const auto bz = fetch(Base::template blobNrAndOffset<1, 2>(ai));
        const auto cx = fetch(Base::template blobNrAndOffset<2, 0>(ai));
        const auto cy = fetch(Base::template blobNrAndOffset<2, 1>(ai));
        const auto cz = fetch(Base::template blobNrAndOffset<2, 2>(ai));

        const auto e1x = bx - ax;
        const auto e1y = by - ay;
        const auto e1z = bz - az;
        const auto e2x = cx - ax;
        const auto e2y = cy - ay;
        const auto e2z = cz - az;

        const auto crossx = e1y * e2z - e1z * e2y;
        const auto crossy = -(e1x * e2z - e1z * e2x);
        const auto crossz = e1x * e2y - e1y * e2x;

        const auto length = std::sqrt(crossx * crossx + crossy * crossy + crossz * crossz);

        [[maybe_unused]] const auto normalx = crossx / length;
        [[maybe_unused]] const auto normaly = crossy / length;
        [[maybe_unused]] const auto normalz = crossz / length;

        using DC = llama::RecordCoord<RecordCoords...>;
        if constexpr(std::is_same_v<DC, llama::RecordCoord<3, 0>>)
            return normalx;
        if constexpr(std::is_same_v<DC, llama::RecordCoord<3, 1>>)
            return normaly;
        if constexpr(std::is_same_v<DC, llama::RecordCoord<3, 2>>)
            return normalz;
        // if constexpr (std::is_same_v<DC, llama::RecordCoord<3>>)
        //{
        //    llama::One<llama::GetType<RecordDim, llama::RecordCoord<3>>> normal;
        //    normal(llama::RecordCoord<0>{}) = normalx;
        //    normal(llama::RecordCoord<1>{}) = normaly;
        //    normal(llama::RecordCoord<2>{}) = normalz;
        //    return normal;
        //}
    }
};

// NOLINTBEGIN(readability-identifier-naming)
namespace pmacc
{
    template<typename T_Type, int T_dim>
    struct Vector
    {
        T_Type data[T_dim];
    };

    struct pmacc_void
    {
    };
    struct pmacc_isAlias
    {
    };
    struct multiMask
    {
    };
    struct localCellIdx
    {
    };

    template<unsigned T_Dim>
    struct DataSpace : Vector<int, T_Dim>
    {
    };
} // namespace pmacc

namespace picongpu
{
    template<typename T_Type = pmacc::pmacc_void, typename T_IsAlias = pmacc::pmacc_isAlias>
    struct position
    {
    };
    struct position_pic
    {
    };
    struct momentum
    {
    };
    struct weighting
    {
    };

    using Frame = llama::Record<
        llama::Field<pmacc::multiMask, std::uint8_t>,
        llama::Field<pmacc::localCellIdx, std::int16_t>,
        llama::Field<position<position_pic>, pmacc::Vector<float, 3>>,
        llama::Field<momentum, pmacc::Vector<float, 3>>,
        llama::Field<weighting, float>>;

    struct totalCellIdx
    {
    };

    using FrameOpenPMD = llama::Record<
        llama::Field<position<position_pic>, pmacc::Vector<float, 3>>,
        llama::Field<momentum, pmacc::Vector<float, 3>>,
        llama::Field<weighting, float>,
        llama::Field<totalCellIdx, pmacc::DataSpace<3>>>;
} // namespace picongpu
// NOLINTEND(readability-identifier-naming)

// clang-format off
struct RngState {};
struct Energy {};
struct NumIALeft {};
struct InitialRange {};
struct DynamicRangeFactor {};
struct TlimitMin {};
struct Pos {};
struct Dir {};
struct NavState {};
// clang-format on

struct RanluxppDouble
{
};

namespace vecgeom
{
    using Precision = double;

    template<typename T>
    struct Vector3D
    {
    };

    struct NavStateIndex
    {
    };
} // namespace vecgeom

// from AdePT:
using Track = llama::Record<
    llama::Field<RngState, RanluxppDouble>,
    llama::Field<Energy, double>,
    llama::Field<NumIALeft, double[3]>,
    llama::Field<InitialRange, double>,
    llama::Field<DynamicRangeFactor, double>,
    llama::Field<TlimitMin, double>,
    llama::Field<Pos, vecgeom::Vector3D<vecgeom::Precision>>,
    llama::Field<Dir, vecgeom::Vector3D<vecgeom::Precision>>,
    llama::Field<NavState, vecgeom::NavStateIndex>>;

// from LHCB HEP:
using NTupleSize_t = std::size_t;

// clang-format off
struct H1isMuon{};
struct H2isMuon{};
struct H3isMuon{};

struct H1PX{};
struct H1PY{};
struct H1PZ{};
struct H1ProbK{};
struct H1ProbPi{};

struct H2PX{};
struct H2PY{};
struct H2PZ{};
struct H2ProbK{};
struct H2ProbPi{};

struct H3PX{};
struct H3PY{};
struct H3PZ{};
struct H3ProbK{};
struct H3ProbPi{};
// clang-format on

using LhcbEvent = llama::Record<
    llama::Field<H1isMuon, int>,
    llama::Field<H2isMuon, int>,
    llama::Field<H3isMuon, int>,
    llama::Field<H1PX, double>,
    llama::Field<H1PY, double>,
    llama::Field<H1PZ, double>,
    llama::Field<H1ProbK, double>,
    llama::Field<H1ProbPi, double>,
    llama::Field<H2PX, double>,
    llama::Field<H2PY, double>,
    llama::Field<H2PZ, double>,
    llama::Field<H2ProbK, double>,
    llama::Field<H2ProbPi, double>,
    llama::Field<H3PX, double>,
    llama::Field<H3PY, double>,
    llama::Field<H3PZ, double>,
    llama::Field<H3ProbK, double>,
    llama::Field<H3ProbPi, double>>;

using LhcbCustom4 = llama::mapping::Split<
    llama::ArrayExtentsDynamic<NTupleSize_t, 1>,
    LhcbEvent,
    mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
    llama::mapping::AlignedAoS,
    llama::mapping::BindSplit<
        mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
        llama::mapping::AlignedAoS,
        llama::mapping::AlignedAoS,
        true>::fn,
    true>;

using LhcbCustom8 = llama::mapping::Split<
    llama::ArrayExtentsDynamic<NTupleSize_t, 1>,
    LhcbEvent,
    mp_list<mp_list<H1isMuon>, mp_list<H2isMuon>, mp_list<H3isMuon>>,
    llama::mapping::BindBitPackedIntAoS<llama::Constant<1>, llama::mapping::SignBit::Discard>::fn,
    llama::mapping::BindSplit<
        mp_list<mp_list<H1ProbK>, mp_list<H2ProbK>>,
        llama::mapping::BindChangeType<llama::mapping::BindAoS<>::fn, mp_list<mp_list<double, float>>>::fn,
        llama::mapping::BindBitPackedFloatAoS<llama::Constant<6>, llama::Constant<16>>::template fn,
        true>::fn,
    true>;

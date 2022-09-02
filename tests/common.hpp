#pragma once

#include <boost/core/demangle.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <llama/llama.hpp>
#include <regex>
#include <sstream>
#include <string>
#include <typeinfo>

// NOLINTNEXTLINE(google-runtime-int)
using SizeTypes = boost::mp11::mp_list<int, unsigned, long, unsigned long, long long, unsigned long long>;
static_assert(boost::mp11::mp_contains<SizeTypes, std::size_t>::value);

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
    static std::regex ulLiteral{"(\\d+)ul"};
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
        std::generate_n(view.storageBlobs[i].storageBlobs.get(), view.mapping().blobSize(i), fillFunc);
    }
}

template<typename View>
void iotaFillView(View& view)
{
    std::int64_t value = 0;
    using RecordDim = typename View::RecordDim;
    for(auto ai : llama::ArrayIndexRange{view.mapping().extents()})
        llama::forEachLeafCoord<RecordDim>(
            [&](auto rc)
            {
                using Type = llama::GetType<RecordDim, decltype(rc)>;
                view(ai)(rc) = static_cast<Type>(value);
                ++value;
            });
}

template<typename View>
void iotaCheckView(View& view)
{
    std::int64_t value = 0;
    using RecordDim = typename View::RecordDim;
    for(auto ai : llama::ArrayIndexRange{view.mapping().extents()})
        llama::forEachLeafCoord<RecordDim>(
            [&](auto rc)
            {
                CAPTURE(ai, rc);
                using Type = llama::GetType<RecordDim, decltype(rc)>;
                CHECK(view(ai)(rc) == static_cast<Type>(value));
                ++value;
            });
}

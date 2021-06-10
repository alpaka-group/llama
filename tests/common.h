#pragma once

#include <boost/core/demangle.hpp>
#include <llama/BlobAllocators.hpp>
#include <llama/mapping/tree/toString.hpp>
#include <numeric>
#include <regex>
#include <string>
#include <typeinfo>

using llama::mapping::tree::internal::replace_all;

template <typename T>
std::string prettyPrintType(const T& t = {})
{
    auto raw = boost::core::demangle(typeid(t).name());
#ifdef _MSC_VER
    // remove clutter in MSVC
    replace_all(raw, "struct ", "");
#endif
#ifdef __GNUG__
    // remove clutter in g++
    static std::regex ulLiteral{"(\\d+)ul"};
    raw = std::regex_replace(raw, ulLiteral, "$1");
#endif

    replace_all(raw, "<", "<\n");
#ifdef _MSC_VER
    replace_all(raw, ",", ",\n");
#else
    replace_all(raw, ", ", ",\n");
#endif
    replace_all(raw, " >", ">");
    replace_all(raw, ">", "\n>");

    std::stringstream rawSS(raw);
    std::string token;
    std::string result;
    int indent = 0;
    while (std::getline(rawSS, token, '\n'))
    {
        if (token.back() == '>' || (token.length() > 1 && token[token.length() - 2] == '>'))
            indent -= 4;
        for (int i = 0; i < indent; ++i)
            result += ' ';
        result += token + "\n";
        if (token.back() == '<')
            indent += 4;
    }
    if (result.back() == '\n')
        result.pop_back();
    return result;
}

namespace internal
{
    inline void zeroBlob(std::shared_ptr<std::byte[]>& sp, size_t blobSize)
    {
        std::memset(sp.get(), 0, blobSize);
    }
    template <typename A>
    void zeroBlob(std::vector<std::byte, A>& v, size_t blobSize)
    {
        std::memset(v.data(), 0, blobSize);
    }
} // namespace internal

template <typename View>
void zeroStorage(View& view)
{
    for (auto i = 0; i < View::Mapping::blobCount; i++)
        internal::zeroBlob(view.storageBlobs[i], view.mapping.blobSize(i));
}

template <typename View>
void iotaStorage(View& view)
{
    for (auto i = 0; i < View::Mapping::blobCount; i++)
    {
        auto fillFunc = [val = 0]() mutable { return static_cast<typename View::BlobType::PrimType>(val++); };
        std::generate_n(view.storageBlobs[i].storageBlobs.get(), view.mapping.blobSize(i), fillFunc);
    }
}
